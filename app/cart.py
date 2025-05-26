from fastapi import APIRouter, Depends, HTTPException, Header, Request, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, event
from typing import List, Optional, Dict
import stripe
import os
from datetime import datetime, timedelta, UTC
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential
from stripe.error import CardError, InvalidRequestError, AuthenticationError, APIError

from .database import get_db
from .models import CartItem, Product, Order, OrderItem
from .schemas import CartItemCreate, CartItem as CartItemSchema, OrderCreate
from .auth import get_current_user
from .logging_config import error_log, audit_log
from .rate_limiter import RateLimiter
from .config import get_settings, Settings
from .background_tasks import send_order_confirmation_email

cart_router = APIRouter()

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
STRIPE_ALLOWED_IPS = os.getenv("STRIPE_ALLOWED_IPS", "").split(",")

class OrderStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

def validate_quantity(quantity: int, max_quantity: int = 10) -> bool:
    """Validate if quantity is within acceptable range."""
    return 0 < quantity <= max_quantity

async def reserve_stock(
    db: Session, 
    product_id: int, 
    quantity: int, 
    duration: timedelta
) -> bool:
    """Reserve stock for a product temporarily."""
    try:
        stmt = select(Product).where(
            and_(
                Product.id == product_id,
                Product.stock_quantity >= quantity
            )
        ).with_for_update(skip_locked=True)
        
        result = db.execute(stmt)
        product = result.scalar_one_or_none()
        
        if not product:
            return False
            
        product.reserved_quantity = (product.reserved_quantity or 0) + quantity
        product.reservation_expiry = datetime.now(UTC) + duration
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        error_log(e, {
            "context": "stock_reservation",
            "product_id": product_id,
            "quantity": quantity
        })
        return False

@cart_router.get("/items", response_model=List[CartItemSchema])
async def get_cart_items(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    items = db.query(CartItem).filter(
        CartItem.user_id == current_user.id,
        CartItem.expires_at > datetime.now(UTC)
    ).all()
    return items

@cart_router.post("/items", response_model=CartItemSchema,
                 dependencies=[Depends(RateLimiter(calls=10, period=60))])
async def add_to_cart(
    item: CartItemCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
    settings: Settings = Depends(get_settings)
):
    """Add item to cart with proper validation and stock reservation."""
    try:
        if not validate_quantity(item.quantity, settings.MAX_CART_ITEM_QUANTITY):
            raise HTTPException(
                status_code=400, 
                detail=f"Quantity must be between 1 and {settings.MAX_CART_ITEM_QUANTITY}"
            )

        # Start transaction
        async with db.begin():
            # Lock product row
            stmt = select(Product).where(Product.id == item.product_id).with_for_update()
            result = db.execute(stmt)
            product = result.scalar_one_or_none()

            if not product:
                raise HTTPException(status_code=404, detail="Product not found")
            
            if product.stock_quantity < item.quantity:
                raise HTTPException(status_code=400, detail="Not enough stock available")

            # Check existing cart item
            existing_item = db.query(CartItem).filter(
                and_(
                    CartItem.user_id == current_user.id,
                    CartItem.product_id == item.product_id,
                    CartItem.expires_at > datetime.now(UTC)
                )
            ).with_for_update().first()

            if existing_item:
                new_quantity = existing_item.quantity + item.quantity
                if not validate_quantity(new_quantity, settings.MAX_CART_ITEM_QUANTITY):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Total quantity cannot exceed {settings.MAX_CART_ITEM_QUANTITY}"
                    )
                
                if product.stock_quantity < new_quantity:
                    raise HTTPException(status_code=400, detail="Not enough stock available")
                
                existing_item.quantity = new_quantity
                existing_item.expires_at = datetime.now(UTC) + timedelta(hours=24)
                cart_item = existing_item
            else:
                cart_item = CartItem(
                    user_id=current_user.id,
                    product_id=item.product_id,
                    quantity=item.quantity,
                    expires_at=datetime.now(UTC) + timedelta(hours=24)
                )
                db.add(cart_item)

            # Reserve stock
            if not await reserve_stock(db, item.product_id, item.quantity, timedelta(minutes=15)):
                raise HTTPException(status_code=400, detail="Could not reserve stock")

            db.commit()
            db.refresh(cart_item)

            audit_log(
                action="item_added_to_cart",
                user_id=current_user.id,
                product_id=item.product_id,
                quantity=item.quantity
            )

            # Schedule cleanup task
            background_tasks.add_task(
                cleanup_expired_cart_items,
                user_id=current_user.id,
                db=db
            )

            return cart_item
    except Exception as e:
        db.rollback()
        error_log(e, {
            "context": "add_to_cart",
            "user_id": current_user.id,
            "product_id": item.product_id,
            "quantity": item.quantity
        })
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def create_stripe_session(session_data: Dict) -> stripe.checkout.Session:
    """Create Stripe checkout session with retry logic."""
    try:
        return stripe.checkout.Session.create(**session_data)
    except CardError as e:
        raise HTTPException(
            status_code=400,
            detail={"message": "Payment failed", "code": e.code, "decline_code": e.decline_code}
        )
    except InvalidRequestError as e:
        error_log(e, {"context": "stripe_invalid_request"})
        raise HTTPException(status_code=400, detail="Invalid payment request")
    except AuthenticationError as e:
        error_log(e, {"context": "stripe_auth_error"})
        raise HTTPException(status_code=500, detail="Payment service unavailable")
    except APIError as e:
        error_log(e, {"context": "stripe_api_error"})
        raise HTTPException(status_code=500, detail="Payment service error")

@cart_router.post("/checkout")
async def create_checkout_session(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create mock checkout session for hackathon demo."""
    try:
        async with db.begin():
            # Get cart items
            cart_items = db.query(CartItem).filter(
                and_(
                    CartItem.user_id == current_user.id,
                    CartItem.expires_at > datetime.now(UTC)
                )
            ).with_for_update().all()

            if not cart_items:
                raise HTTPException(status_code=400, detail="Cart is empty")

            # Calculate total and prepare items
            order_metadata = {}
            total_amount = 0

            for item in cart_items:
                product = db.query(Product).filter(Product.id == item.product_id).with_for_update().first()
                if not product:
                    raise HTTPException(status_code=404, detail=f"Product not found: {item.product_id}")
                
                order_metadata[f"item_{product.id}"] = item.quantity
                total_amount += product.price * item.quantity

            # Create mock session ID
            mock_session_id = f"mock_session_{datetime.now(UTC).timestamp()}"

            # Create order directly as completed (for hackathon demo)
            order = Order(
                user_id=current_user.id,
                stripe_session_id=mock_session_id,
                status=OrderStatus.COMPLETED,
                total_amount=total_amount,
                metadata=order_metadata,
                payment_status="paid",
                completed_at=datetime.now(UTC)
            )
            db.add(order)

            # Create order items and update stock
            for item in cart_items:
                product = db.query(Product).filter(Product.id == item.product_id).with_for_update().first()
                
                order_item = OrderItem(
                    order_id=order.id,
                    product_id=item.product_id,
                    quantity=item.quantity,
                    price_at_time=product.price
                )
                db.add(order_item)

                # Update stock
                product.stock_quantity -= item.quantity
                if product.reserved_quantity:
                    product.reserved_quantity -= item.quantity

            # Clear cart
            db.query(CartItem).filter(CartItem.user_id == current_user.id).delete()

            db.commit()

            audit_log(
                action="mock_checkout_completed",
                user_id=current_user.id,
                order_id=order.id,
                amount=total_amount
            )

            # Return success page URL directly
            return {
                "checkout_url": f"{os.getenv('FRONTEND_URL')}/success?order_id={order.id}",
                "order_id": order.id
            }

    except Exception as e:
        db.rollback()
        error_log(e, {
            "context": "mock_checkout",
            "user_id": current_user.id
        })
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail="An error occurred during checkout")

@cart_router.post("/webhook")
async def stripe_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    stripe_signature: str = Header(None)
):
    """Mock webhook handler for hackathon demo."""
    try:
        # For hackathon purposes, we'll just log the webhook call
        audit_log(
            action="mock_webhook_received",
            status="success"
        )
        return {"status": "success"}
    except Exception as e:
        error_log(e, {"context": "mock_webhook"})
        raise HTTPException(status_code=500, detail="Error processing webhook")

async def cleanup_expired_cart_items(user_id: int, db: Session):
    """Cleanup expired cart items and release reserved stock."""
    try:
        async with db.begin():
            expired_items = db.query(CartItem).filter(
                and_(
                    CartItem.user_id == user_id,
                    CartItem.expires_at <= datetime.now(UTC)
                )
            ).with_for_update().all()

            for item in expired_items:
                # Release reserved stock
                product = db.query(Product).filter(
                    Product.id == item.product_id
                ).with_for_update().first()
                
                if product:
                    product.reserved_quantity = max(0, (product.reserved_quantity or 0) - item.quantity)

                db.delete(item)

            db.commit()

    except Exception as e:
        db.rollback()
        error_log(e, {
            "context": "cart_cleanup",
            "user_id": user_id
        })