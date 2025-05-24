from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict
from datetime import datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    firebase_uid: Optional[str] = None

class User(UserBase):
    id: int
    is_active: bool
    firebase_uid: Optional[str] = None

    class Config:
        from_attributes = True

class ProductBase(BaseModel):
    name: str
    description: str
    price: float
    category: str
    image_url: str
    metadata: Dict

class ProductCreate(ProductBase):
    image_embedding: Optional[List[float]] = None

class Product(ProductBase):
    id: int
    image_embedding: Optional[List[float]] = None

    class Config:
        from_attributes = True

class CartItemBase(BaseModel):
    product_id: int
    quantity: int

class CartItemCreate(CartItemBase):
    pass

class CartItem(CartItemBase):
    id: int
    user_id: int
    created_at: datetime
    product: Product

    class Config:
        from_attributes = True

class SearchQuery(BaseModel):
    query_text: str
    image_url: Optional[str] = None

class SearchRefinement(BaseModel):
    original_query_id: int
    refinement_text: str

class SearchResult(BaseModel):
    products: List[Product]
    similarity_scores: List[float]
    query_understanding: str 