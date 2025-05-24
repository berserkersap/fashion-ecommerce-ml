"""
Example usage of product operations in the e-commerce system.
This module demonstrates:
1. Creating products with images
2. Deleting products
3. Rollback scenarios

Note: All examples still needs to be tested and may not work.
"""

from fastapi import UploadFile
from PIL import Image
import io
from sqlalchemy.orm import Session
from typing import Dict, Optional
from datetime import datetime, timedelta, UTC
import asyncio

from app.models import Product
from app.vector_store import vector_store
from app.utils import upload_to_gcs, delete_from_gcs
from app.search import get_fashion_image_embedding
from app.logging_config import error_log, audit_log
from app.database import get_db

class ProductOperations:
    def __init__(self, db: Session):
        self.db = db
        self._deleted_products = {}  # Store for potential rollbacks
        self._updated_products = {}  # Store original state for rollbacks
        
    async def create_product_with_image(
        self,
        name: str,
        description: str,
        price: float,
        category: str,
        metadata: Dict,
        image: UploadFile
    ) -> Product:
        """
        Create a new product with image, storing data in both SQL and vector databases.
        
        Args:
            name: Product name
            description: Product description
            price: Product price
            category: Product category
            metadata: Additional product metadata (e.g., brand, size, color)
            image: Uploaded image file
        
        Returns:
            Product: Created product instance
        """
        try:
            # 1. Process the image
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents))
            
            # 2. Upload image to Google Cloud Storage
            image_url = await upload_to_gcs(contents, image.filename)
            
            # 3. Generate image embedding using Fashion-CLIP
            image_embedding = get_fashion_image_embedding(pil_image)
            
            # 4. Create product in SQL database
            product = Product(
                name=name,
                description=description,
                price=price,
                category=category,
                image_url=image_url,
                metadata=metadata,
                image_embedding=image_embedding  # Store embedding in SQL for backup
            )
            
            self.db.add(product)
            self.db.commit()  # This will set the product.id
            self.db.refresh(product)
            
            # 5. Store embedding in vector database
            vector_store.upsert_embedding(product.id, image_embedding)
            
            audit_log(
                action="product_created",
                product_id=product.id,
                category=category,
                has_embedding=True
            )
            
            return product
            
        except Exception as e:
            # Cleanup uploaded image if anything fails
            if 'image_url' in locals():
                await delete_from_gcs(image_url)
            self.db.rollback()
            error_log(e, {
                "context": "product_creation",
                "name": name,
                "category": category
            })
            raise e

    async def delete_product_with_rollback(
        self,
        product_id: int,
        rollback_window: timedelta = timedelta(hours=24)
    ) -> Dict:
        """
        Delete a product with ability to rollback within specified window.
        
        Args:
            product_id: ID of product to delete
            rollback_window: How long to keep product data for potential rollback
            
        Returns:
            Dict with deletion status
        """
        try:
            # 1. Get product
            product = self.db.query(Product).filter(Product.id == product_id).first()
            if not product:
                raise ValueError(f"Product {product_id} not found")
            
            # 2. Store product data for potential rollback
            self._deleted_products[product_id] = {
                "data": {
                    "name": product.name,
                    "description": product.description,
                    "price": product.price,
                    "category": product.category,
                    "metadata": product.metadata,
                    "image_embedding": product.image_embedding
                },
                "image_url": product.image_url,
                "deleted_at": datetime.now(UTC),
                "rollback_window": rollback_window
            }
            
            # 3. Delete from vector store first (can be recreated if needed)
            vector_store.delete_embedding(product_id)
            
            # 4. Delete from SQL database
            self.db.delete(product)
            self.db.commit()
            
            # 5. Schedule cleanup of rollback data
            asyncio.create_task(
                self._cleanup_rollback_data(product_id, rollback_window)
            )
            
            audit_log(
                action="product_deleted",
                product_id=product_id,
                rollback_available_until=(
                    datetime.now(UTC) + rollback_window
                ).isoformat()
            )
            
            return {
                "message": "Product deleted successfully",
                "rollback_available_until": (
                    datetime.now(UTC) + rollback_window
                ).isoformat()
            }
            
        except Exception as e:
            self.db.rollback()
            error_log(e, {
                "context": "product_deletion",
                "product_id": product_id
            })
            raise e

    async def rollback_deletion(self, product_id: int) -> Optional[Product]:
        """
        Rollback a product deletion if within rollback window.
        
        Args:
            product_id: ID of product to restore
            
        Returns:
            Product if restored, None if not possible
        """
        try:
            # 1. Check if product is available for rollback
            deleted_data = self._deleted_products.get(product_id)
            if not deleted_data:
                return None
            
            # 2. Check if within rollback window
            deletion_time = deleted_data["deleted_at"]
            window = deleted_data["rollback_window"]
            if datetime.now(UTC) - deletion_time > window:
                return None
            
            # 3. Recreate product
            product_data = deleted_data["data"]
            product = Product(
                id=product_id,
                **product_data
            )
            
            # 4. Restore in SQL database
            self.db.add(product)
            self.db.commit()
            self.db.refresh(product)
            
            # 5. Restore in vector store
            vector_store.upsert_embedding(
                product_id,
                product_data["image_embedding"]
            )
            
            # 6. Remove from deleted products
            self._deleted_products.pop(product_id)
            
            audit_log(
                action="product_restored",
                product_id=product_id,
                deletion_duration=(
                    datetime.now(UTC) - deletion_time
                ).total_seconds()
            )
            
            return product
            
        except Exception as e:
            self.db.rollback()
            error_log(e, {
                "context": "product_restoration",
                "product_id": product_id
            })
            raise e

    async def _cleanup_rollback_data(
        self,
        product_id: int,
        delay: timedelta
    ):
        """Clean up rollback data after window expires."""
        await asyncio.sleep(delay.total_seconds())
        if product_id in self._deleted_products:
            # Delete stored image
            image_url = self._deleted_products[product_id]["image_url"]
            await delete_from_gcs(image_url)
            # Remove rollback data
            self._deleted_products.pop(product_id)


# Example usage:
async def example_product_lifecycle():
    """Example showing complete product lifecycle with rollback."""
    
    # Setup (assuming you have these configured)
    db_session = next(get_db())  # Get database session
    ops = ProductOperations(db_session)
    
    try:
        # 1. Create a product
        with open("examples/sample_data/denim_jacket.jpg", "rb") as f:
            image = UploadFile(
                filename="denim_jacket.jpg",
                file=f
            )
        
        product = await ops.create_product_with_image(
            name="Premium Denim Jacket",
            description="High-quality denim jacket with vintage wash",
            price=129.99,
            category="jackets",
            metadata={
                "brand": "FashionCo",
                "color": "vintage blue",
                "sizes": ["S", "M", "L", "XL"],
                "material": "100% cotton denim",
                "care": "Machine wash cold"
            },
            image=image
        )
        print(f"Created product: {product.id}")
        
        # 2. Delete the product (with 1 hour rollback window)
        deletion_result = await ops.delete_product_with_rollback(
            product.id,
            rollback_window=timedelta(hours=1)
        )
        print(f"Deletion result: {deletion_result}")
        
        # 3. Rollback the deletion
        restored_product = await ops.rollback_deletion(product.id)
        if restored_product:
            print(f"Successfully restored product {restored_product.id}")
        else:
            print("Product could not be restored (rollback window expired)")
            
    except Exception as e:
        print(f"Error in product lifecycle: {str(e)}")
    finally:
        await image.close()
        db_session.close()

# Run the example
if __name__ == "__main__":
    asyncio.run(example_product_lifecycle()) 