from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from PIL import Image
import io

from .database import get_db
from .models import Product
from .schemas import ProductCreate, Product as ProductSchema
from .auth import get_current_user
from .utils import upload_to_gcs, delete_from_gcs
from .search import get_image_embedding
from .vector_store import vector_store

product_router = APIRouter()

@product_router.post("/", response_model=ProductSchema)
async def create_product(
    name: str = Form(...),
    description: str = Form(...),
    price: float = Form(...),
    category: str = Form(...),
    image: UploadFile = File(...),
    metadata: dict = Form(...),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    try:
        # Process and upload image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Upload to Google Cloud Storage
        image_url = await upload_to_gcs(contents, image.filename)
        
        # Generate image embedding
        image_embedding = get_image_embedding(pil_image)
        
        # Create product
        product = Product(
            name=name,
            description=description,
            price=price,
            category=category,
            image_url=image_url,
            metadata=metadata,
            image_embedding=image_embedding
        )
        
        db.add(product)
        db.commit()
        db.refresh(product)
        
        # Store embedding in vector database
        vector_store.upsert_embedding(product.id, image_embedding)
        
        return product
    except Exception as e:
        # Cleanup uploaded image if product creation fails
        if 'image_url' in locals():
            await delete_from_gcs(image_url)
        raise HTTPException(status_code=400, detail=str(e))

@product_router.get("/", response_model=List[ProductSchema])
async def get_products(
    skip: int = 0,
    limit: int = 100,
    category: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Product)
    if category:
        query = query.filter(Product.category == category)
    return query.offset(skip).limit(limit).all()

@product_router.get("/{product_id}", response_model=ProductSchema)
async def get_product(
    product_id: int,
    db: Session = Depends(get_db)
):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@product_router.put("/{product_id}", response_model=ProductSchema)
async def update_product(
    product_id: int,
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    price: Optional[float] = Form(None),
    category: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    metadata: Optional[dict] = Form(None),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    try:
        if image:
            contents = await image.read()
            pil_image = Image.open(io.BytesIO(contents))
            
            # Upload to Google Cloud Storage
            image_url = await upload_to_gcs(contents, image.filename)
            
            # Generate new image embedding
            image_embedding = get_image_embedding(pil_image)
            
            # Delete old image
            if product.image_url:
                await delete_from_gcs(product.image_url)
            
            product.image_url = image_url
            product.image_embedding = image_embedding
            
            # Update vector store
            vector_store.upsert_embedding(product.id, image_embedding)

        # Update other fields if provided
        if name is not None:
            product.name = name
        if description is not None:
            product.description = description
        if price is not None:
            product.price = price
        if category is not None:
            product.category = category
        if metadata is not None:
            product.metadata = metadata

        db.commit()
        db.refresh(product)
        return product
    except Exception as e:
        # Cleanup uploaded image if update fails
        if 'image_url' in locals():
            await delete_from_gcs(image_url)
        raise HTTPException(status_code=400, detail=str(e))

@product_router.delete("/{product_id}")
async def delete_product(
    product_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    product = db.query(Product).filter(Product.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    try:
        # Delete image from Google Cloud Storage
        if product.image_url:
            await delete_from_gcs(product.image_url)
        
        # Delete from vector store
        vector_store.delete_embedding(product_id)
        
        # Delete product from database
        db.delete(product)
        db.commit()
        return {"message": "Product deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting product: {str(e)}") 