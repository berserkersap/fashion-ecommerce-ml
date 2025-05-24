from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON, DateTime, Boolean, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class Product(Base):
    __tablename__ = "product_main_table"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    price = Column(Float)
    image_url = Column(String)
    category = Column(String, index=True)
    metadata = Column(JSON)  # Store additional product attributes
    stock_quantity = Column(Integer, default=0)
    reserved_quantity = Column(Integer, default=0)
    reservation_expiry = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    cart_items = relationship("CartItem", back_populates="product")

class CartItem(Base):
    __tablename__ = "cart_items"

    id = Column(Integer, primary_key=True, index=True)
    firebase_uid = Column(String, index=True, nullable=False)  # Firebase UID instead of user_id
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)  # Cart items expire after 24 hours if not checked out
    
    product = relationship("Product", back_populates="cart_items")

class SearchHistory(Base):
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)
    firebase_uid = Column(String, index=True, nullable=False)  # Firebase UID instead of user_id
    query_text = Column(Text, nullable=True)
    image_description = Column(Text, nullable=True)
    conversation_history = Column(JSON, default=list)  # Store conversation with agent
    refined_queries = Column(JSON, default=list)  # Store refinement history
    created_at = Column(DateTime, default=datetime.utcnow)

class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    firebase_uid = Column(String, index=True, nullable=False)  # Firebase UID instead of user_id
    status = Column(String)  # pending, completed, failed
    total_amount = Column(Float)
    metadata = Column(JSON)  # Store order items and quantities
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    stripe_session_id = Column(String, unique=True, nullable=True)
    payment_status = Column(String, nullable=True)  # paid, pending, failed
    
    items = relationship("OrderItem", back_populates="order")

class OrderItem(Base):
    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    product_id = Column(Integer, ForeignKey("products.id"))
    quantity = Column(Integer)
    price_at_time = Column(Float)
    
    order = relationship("Order", back_populates="items")
    product = relationship("Product")

class TempUpload(Base):
    __tablename__ = "temp_uploads"

    id = Column(Integer, primary_key=True, index=True)
    firebase_uid = Column(String, index=True, nullable=False)  # Firebase UID instead of user_id
    file_url = Column(String)
    image_description = Column(Text, nullable=True)  # From Moondream vLLM
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)  # 1 hour from creation 