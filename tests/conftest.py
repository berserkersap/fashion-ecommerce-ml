import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
from PIL import Image
import io
import base64
from datetime import datetime
from typing import Tuple, Optional

from app.main import app
from app.database import Base, get_db
from app.models import Product, User
from app.firebase_session import firebase_session_manager

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_test_image(size: Tuple[int, int] = (100, 100), color: str = 'red', format: str = 'JPEG') -> dict:
    """Helper function to create test images with different parameters"""
    img = Image.new('RGB', size, color=color)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    
    return {
        "file": (f"test.{format.lower()}", img_byte_arr, f"image/{format.lower()}"),
        "content": img_byte_arr.getvalue(),
        "size": size,
        "format": format
    }

@pytest.fixture(scope="session")
def db():
    # Create test database tables
    Base.metadata.create_all(bind=engine)
    
    # Create test session
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db):
    # Override the get_db dependency
    def override_get_db():
        try:
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clear database after each test
    for table in reversed(Base.metadata.sorted_tables):
        db.execute(table.delete())
    db.commit()

@pytest.fixture(scope="function")
def test_user(db):
    user = User(
        email="test@example.com",
        firebase_uid="test123",
        is_active=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@pytest.fixture(scope="function")
def test_products(db):
    products = [
        Product(
            name="Blue Jeans",
            description="Classic blue denim jeans",
            price=79.99,
            category="pants",
            image_url="https://storage.googleapis.com/test-bucket/jeans.jpg",
            metadata={
                "brand": "TestBrand",
                "color": "blue",
                "size": ["S", "M", "L"]
            }
        ),
        Product(
            name="White T-Shirt",
            description="Basic white cotton t-shirt",
            price=29.99,
            category="shirts",
            image_url="https://storage.googleapis.com/test-bucket/tshirt.jpg",
            metadata={
                "brand": "TestBrand",
                "color": "white",
                "size": ["XS", "S", "M", "L", "XL"]
            }
        )
    ]
    
    for product in products:
        db.add(product)
    db.commit()
    
    for product in products:
        db.refresh(product)
    
    return products

@pytest.fixture(scope="function")
def test_image():
    """Default test image fixture"""
    return create_test_image()

@pytest.fixture(scope="function")
def test_image_large():
    """Large test image fixture"""
    return create_test_image(size=(2000, 2000))

@pytest.fixture(scope="function")
def test_image_invalid_format():
    """Invalid format test image fixture"""
    return {
        "file": ("test.txt", b"not an image", "text/plain"),
        "content": b"not an image"
    }

@pytest.fixture(scope="function")
def auth_headers(test_user):
    # Mock Firebase token validation
    return {"Authorization": f"Bearer test_token_{test_user.firebase_uid}"} 