from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Tuple, Set
from PIL import Image
import io
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import os
from concurrent.futures import ThreadPoolExecutor
from structlog import get_logger
from datetime import datetime, timedelta, UTC
import asyncio
from opentelemetry import trace, metrics
import uuid
import base64
from google.cloud import secretmanager
from sqlalchemy.sql import func
from sqlalchemy.sql.functions import plainto_tsquery

from .database import get_db
from .models import Product, SearchHistory, TempUpload
from .schemas import SearchQuery, SearchResult, SearchRefinement
from .auth import get_current_user
from .agent import EcommerceAgent
from .logging_config import log_endpoint_access, log_search_request, audit_log, error_log, app_logger
from .firebase_session import firebase_session_manager
from .utils import upload_to_gcs, delete_from_gcs
from .ml_models import ml_models  # Import singleton instance
from .vector_store import vector_store
from .products import product_router
from .config import Settings, get_settings
from .search.image_processor import ImageProcessor

search_router = APIRouter()

# Initialize Fashion-CLIP model and processor
try:
    model_name = "patrickjohncyh/fashion-clip"
    fashion_processor = CLIPProcessor.from_pretrained(model_name)
    fashion_model = CLIPModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    fashion_model.eval()  # Set to evaluation mode
    app_logger.info(f"Successfully initialized Fashion-CLIP model on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    error_log(e, {"context": "Fashion-CLIP initialization"})
    raise

def get_image_description(image: Image.Image) -> str:
    """Get image description using Moondream vLLM"""
    return ml_models.get_image_description(image)

# Embedding combination weights
IMAGE_WEIGHT = 0.7
TEXT_WEIGHT = 0.3

def combine_embeddings(image_embedding: List[float], text_embedding: List[float], 
                      weights: Tuple[float, float] = (IMAGE_WEIGHT, TEXT_WEIGHT)) -> List[float]:
    """
    Combine image and text embeddings with specified weights
    """
    try:
        image_weight, text_weight = weights
        if abs((image_weight + text_weight) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
            
        # Convert to numpy for efficient computation
        img_emb = np.array(image_embedding)
        txt_emb = np.array(text_embedding)
        
        # Weighted combination
        combined = (image_weight * img_emb) + (text_weight * txt_emb)
        
        # Normalize the combined embedding
        combined_norm = combined / np.linalg.norm(combined)
        
        return combined_norm.tolist()
    except Exception as e:
        error_log(e, {
            "context": "embedding_combination",
            "weights": weights
        })
        raise

def get_fashion_image_embedding(image: Image.Image) -> List[float]:
    """Get fashion-specific image embedding using Fashion-CLIP"""
    try:
        # Preprocess image
        inputs = fashion_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(fashion_model.device) for k, v in inputs.items()}
        
        # Get embedding
        with torch.no_grad():
            image_features = fashion_model.get_image_features(**inputs)
            embedding = image_features[0].cpu().numpy()
            
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    except Exception as e:
        error_log(e, {"context": "fashion_image_embedding"})
        raise

def get_fashion_text_embedding(text: str) -> List[float]:
    """Get fashion-specific text embedding using Fashion-CLIP"""
    try:
        # Preprocess text
        inputs = fashion_processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(fashion_model.device) for k, v in inputs.items()}
        
        # Get embedding
        with torch.no_grad():
            text_features = fashion_model.get_text_features(**inputs)
            embedding = text_features[0].cpu().numpy()
            
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    except Exception as e:
        error_log(e, {"context": "fashion_text_embedding"})
        raise

# Initialize agent per request
def get_agent(db: Session = Depends(get_db)) -> EcommerceAgent:
    return EcommerceAgent(db)

# Start session cleanup task when application starts
@search_router.on_event("startup")
async def startup_event():
    await firebase_session_manager.start_cleanup_task()

# Stop session cleanup task when application shuts down
@search_router.on_event("shutdown")
async def shutdown_event():
    await firebase_session_manager.stop_cleanup_task()

# Image validation constants
MAX_IMAGE_SIZE_MB = 5
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png'}
MIN_IMAGE_SIZE_PIXELS = 100
MAX_IMAGE_SIZE_PIXELS = 4096
MAX_IMAGES_PER_REQUEST = 3  # Maximum number of images allowed per request

async def process_single_image(image: UploadFile, firebase_uid: str, db: Session) -> Tuple[Image.Image, str, str]:
    """Process a single uploaded image"""
    try:
        # Validate and process image using ImageProcessor
        pil_image = await ImageProcessor.validate_and_process_image(image)
        
        # Upload to temporary storage
        contents = await image.read()
        temp_image_url = await upload_to_gcs(
            contents,
            f"temp_search_{firebase_uid}_{uuid.uuid4().hex}_{image.filename}"
        )
        
        # Get image description
        image_description = get_image_description(pil_image)
        
        try:
            # Store temporary upload record
            temp_upload = TempUpload(
                firebase_uid=firebase_uid,
                file_url=temp_image_url,
                image_description=image_description,
                expires_at=datetime.now(UTC) + timedelta(hours=1)
            )
            db.add(temp_upload)
            db.commit()
        except Exception as db_error:
            # If database operation fails, cleanup the uploaded file
            await delete_from_gcs(temp_image_url)
            db.rollback()
            raise db_error
        
        return pil_image, temp_image_url, image_description
    except Exception as e:
        error_log(e, {"context": "image_processing", "user_id": firebase_uid})
        if 'temp_image_url' in locals():
            await delete_from_gcs(temp_image_url)
        raise

@search_router.post("", response_model=SearchResult)
@log_search_request
@log_endpoint_access
async def search_products(
    query: Optional[str] = Form(None),
    images: List[UploadFile] = File(default=[]),
    image_weight: Optional[float] = Form(default=0.7),
    text_weight: Optional[float] = Form(default=0.3),
    filters: Optional[Dict] = Form(default=None),
    page: int = Form(default=1),
    per_page: int = Form(default=20),
    sort_by: str = Form(default="relevance"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
    settings: Settings = Depends(get_settings)
):
    """Enhanced search with both vector and SQL-based filtering"""
    if not query and not images:
        raise HTTPException(status_code=400, detail="Must provide either query text or images")
        
    if image_weight + text_weight != 1.0:
        raise HTTPException(status_code=400, detail="Image and text weights must sum to 1.0")

    if images and len(images) > settings.MAX_IMAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400, 
            detail=f"Maximum {settings.MAX_IMAGES_PER_REQUEST} images can be uploaded at once"
        )

    temp_image_urls = []
    try:
        # Process images if provided
        combined_image_embedding = None
        image_descriptions = []
        
        if images:
            # Process all images concurrently
            processing_results = await asyncio.gather(
                *[process_single_image(image, current_user.id, db) for image in images],
                return_exceptions=True
            )
            
            # Check for any errors in the results
            for result in processing_results:
                if isinstance(result, Exception):
                    raise result
            
            # Separate the results
            pil_images, temp_image_urls, image_descriptions = zip(*processing_results)
            pil_images = list(pil_images)
            temp_image_urls = list(temp_image_urls)
            image_descriptions = list(image_descriptions)

            # Generate embeddings concurrently using ThreadPoolExecutor for CPU-bound operations
            if pil_images:
                with ThreadPoolExecutor() as executor:
                    loop = asyncio.get_event_loop()
                    image_embeddings = await asyncio.gather(
                        *[loop.run_in_executor(executor, get_fashion_image_embedding, img) for img in pil_images]
                    )
                    
                    # Average the embeddings
                    combined_image_embedding = np.mean(image_embeddings, axis=0)
                    # Normalize the combined embedding
                    combined_image_embedding = combined_image_embedding / np.linalg.norm(combined_image_embedding)
                    combined_image_embedding = combined_image_embedding.tolist()

        # Get text embedding if query provided
        text_embedding = None
        if query:
            text_embedding = get_fashion_text_embedding(query)

        # Combine embeddings if both present
        final_embedding = None
        if combined_image_embedding and text_embedding:
            final_embedding = np.array([
                image_weight * i + text_weight * t
                for i, t in zip(combined_image_embedding, text_embedding)
            ])
            final_embedding = final_embedding / np.linalg.norm(final_embedding)
            final_embedding = final_embedding.tolist()
        elif combined_image_embedding:
            final_embedding = combined_image_embedding
        else:
            final_embedding = text_embedding

        # Search vector store first
        search_results = vector_store.search_similar(
            embedding=final_embedding,
            limit=per_page * 3,  # Get more results for post-filtering
            score_threshold=0.5
        )
        
        # Get product IDs from vector search
        product_ids = [result["id"] for result in search_results]
        
        # Build SQL query with filters
        products_query = db.query(Product).filter(Product.id.in_(product_ids))
        
        # Apply enhanced filters
        if filters:
            if 'gender' in filters:
                products_query = products_query.filter(Product.gender == filters['gender'])
            
            if 'category' in filters:
                products_query = products_query.filter(Product.category == filters['category'])
            
            if 'subcategory' in filters:
                products_query = products_query.filter(Product.subcategory == filters['subcategory'])
            
            if 'articletype' in filters:
                products_query = products_query.filter(Product.articletype == filters['articletype'])
            
            if 'basecolour' in filters:
                products_query = products_query.filter(Product.basecolour == filters['basecolour'])
            
            if 'usage' in filters:
                products_query = products_query.filter(Product.usage == filters['usage'])
            
            if 'company_brand' in filters:
                products_query = products_query.filter(Product.company_brand == filters['company_brand'])
            
            if 'size' in filters:
                products_query = products_query.filter(Product.size == filters['size'])
            
            if 'price_range' in filters:
                min_price, max_price = filters['price_range']
                products_query = products_query.filter(Product.price.between(min_price, max_price))

            # Add text search if query exists
            if query:
                search_terms = plainto_tsquery('english', query)
                products_query = products_query.filter(Product.search_vector.op('@@')(search_terms))

        # Apply sorting
        if sort_by == "price_low":
            products_query = products_query.order_by(Product.price.asc())
        elif sort_by == "price_high":
            products_query = products_query.order_by(Product.price.desc())
        elif sort_by == "newest":
            products_query = products_query.order_by(Product.created_at.desc())
        else:  # "relevance"
            # Sort by vector search score
            id_to_score = {result["id"]: result["score"] for result in search_results}
            products = products_query.all()
            products.sort(key=lambda p: id_to_score.get(p.id, 0), reverse=True)
            
            # Apply pagination after sorting
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            products = products[start_idx:end_idx]
            total_count = len(products)
        
        # If not relevance sort, use SQL pagination
        if sort_by != "relevance":
            total_count = products_query.count()
            products = products_query.offset((page - 1) * per_page).limit(per_page).all()

        # Save search history with enhanced metadata
        search_history = SearchHistory(
            firebase_uid=current_user.id,
            query_text=query,
            image_description=", ".join(filter(None, image_descriptions)) if image_descriptions else None,
            metadata={
                "image_weight": image_weight,
                "text_weight": text_weight,
                "temp_image_urls": temp_image_urls,
                "image_count": len(images) if images else 0,
                "filters": filters,
                "sort_by": sort_by,
                "applied_filters": {
                    k: v for k, v in (filters or {}).items() 
                    if k in ['gender', 'category', 'subcategory', 'articletype', 
                            'basecolour', 'usage', 'company_brand', 'size']
                }
            }
        )
        db.add(search_history)
        db.commit()

        return {
            "products": products,
            "total": total_count,
            "page": page,
            "total_pages": (total_count + per_page - 1) // per_page,
            "filters_applied": filters or {},
            "search_type": "combined" if query and images else "image" if images else "text",
            "available_filters": {
                "genders": db.query(Product.gender).distinct().all(),
                "categories": db.query(Product.category).distinct().all(),
                "colours": db.query(Product.basecolour).distinct().all(),
                "brands": db.query(Product.company_brand).distinct().all(),
                "sizes": db.query(Product.size).distinct().all(),
                "price_range": {
                    "min": db.query(func.min(Product.price)).scalar(),
                    "max": db.query(func.max(Product.price)).scalar()
                }
            }
        }
        
    except Exception as e:
        # Cleanup any uploaded files in case of error
        cleanup_tasks = [delete_from_gcs(url) for url in temp_image_urls]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        error_log(e, {
            "context": "search_execution",
            "user_id": current_user.id,
            "query": query,
            "image_count": len(images) if images else 0
        })
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

@search_router.post("/refine", response_model=SearchResult)
@log_endpoint_access
async def refine_search(
    refinement_text: Optional[str] = Form(None),
    original_query_id: int = Form(...),
    images: List[UploadFile] = File(default=[]),
    image_weight: Optional[float] = Form(default=IMAGE_WEIGHT),
    text_weight: Optional[float] = Form(default=TEXT_WEIGHT),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
    agent: EcommerceAgent = Depends(get_agent),
    settings: Settings = Depends(get_settings)
):
    try:
        # Get original search
        original_search = db.query(SearchHistory).filter_by(id=original_query_id).first()
        if not original_search or original_search.user_id != current_user.id:
            error_log(ValueError("Original search not found"), {
                "user_id": current_user.id,
                "original_query_id": original_query_id
            })
            raise HTTPException(status_code=404, detail="Original search not found")

        if not refinement_text and not images:
            raise HTTPException(
                status_code=400, 
                detail="Either refinement text or new images must be provided"
            )

        if images and len(images) > settings.MAX_IMAGES_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {settings.MAX_IMAGES_PER_REQUEST} images can be uploaded at once"
            )

        # Process new images if provided
        pil_images = []
        temp_image_urls = []
        combined_image_embedding = None
        
        if images:
            try:
                # Process all images concurrently
                processing_results = await asyncio.gather(
                    *[process_single_image(image, current_user.id, db) for image in images],
                    return_exceptions=True
                )
                
                # Check for any errors in the results
                for result in processing_results:
                    if isinstance(result, Exception):
                        raise result
                
                # Separate the results
                pil_images, temp_image_urls, image_descriptions = zip(*processing_results)
                pil_images = list(pil_images)
                temp_image_urls = list(temp_image_urls)
                image_descriptions = list(image_descriptions)

                # Generate embeddings concurrently
                if pil_images:
                    with ThreadPoolExecutor() as executor:
                        loop = asyncio.get_event_loop()
                        image_embeddings = await asyncio.gather(
                            *[loop.run_in_executor(executor, get_fashion_image_embedding, img) 
                              for img in pil_images]
                        )
                        
                        # Average the embeddings
                        combined_image_embedding = np.mean(image_embeddings, axis=0)
                        # Normalize the combined embedding
                        combined_image_embedding = combined_image_embedding / np.linalg.norm(combined_image_embedding)
                        combined_image_embedding = combined_image_embedding.tolist()
                    
            except Exception as e:
                # Cleanup any uploaded files in case of error
                cleanup_tasks = [delete_from_gcs(url) for url in temp_image_urls]
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
                
                error_log(e, {
                    "context": "image_processing_refinement",
                    "user_id": current_user.id,
                    "image_count": len(images) if images else 0
                })
                raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")

        # Update user's last activity
        firebase_session_manager.update_user_activity(current_user.id)

        # Combine original search with refinement
        combined_message = f"Original search: {original_search.query_text or 'image search'}"
        if refinement_text:
            combined_message += f"\nRefinement: {refinement_text}"
        
        result = await agent.handle_conversation(
            user_id=current_user.id,
            message=combined_message,
            image_embedding=combined_image_embedding,
            get_text_embedding=get_fashion_text_embedding,
            embedding_weights=(image_weight, text_weight),
            previous_search_id=original_query_id
        )
        
        # Update search history
        search_history = SearchHistory(
            user_id=current_user.id,
            query_text=refinement_text,
            image_description=result.get("image_description"),
            metadata={
                "original_query_id": original_query_id,
                "image_weight": image_weight,
                "text_weight": text_weight,
                "temp_image_urls": temp_image_urls,
                "image_count": len(images) if images else 0
            }
        )
        db.add(search_history)
        db.commit()

        audit_log(
            action="search_refinement_completed",
            user_id=current_user.id,
            original_search_id=original_query_id,
            new_search_id=search_history.id,
            has_results=bool(result.get("products")),
            has_text_refinement=bool(refinement_text),
            image_count=len(images) if images else 0
        )
        
        return result
        
    except Exception as e:
        error_log(e, {
            "context": "search_refinement",
            "user_id": current_user.id,
            "original_query_id": original_query_id,
            "refinement_text": refinement_text,
            "image_count": len(images) if images else 0
        })
        raise HTTPException(status_code=500, detail=f"Error during refinement: {str(e)}")

@search_router.post("/checkout")
@log_endpoint_access
async def checkout_products(
    product_ids: List[int],
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
    agent: EcommerceAgent = Depends(get_agent)
):
    try:
        audit_log(
            action="checkout_started",
            user_id=current_user.id,
            product_ids=product_ids
        )
        
        cart_items = await agent._add_to_cart(current_user.id, product_ids)
        
        audit_log(
            action="checkout_completed",
            user_id=current_user.id,
            cart_items=[item.id for item in cart_items]
        )
        
        return {"message": "Products added to cart successfully", "cart_items": cart_items}
    except Exception as e:
        error_log(e, {
            "context": "checkout",
            "user_id": current_user.id,
            "product_ids": product_ids
        })
        raise HTTPException(status_code=500, detail=f"Error during checkout: {str(e)}")

# Example of accessing secrets in code
def get_credential(credential_name: str) -> str:
    if os.getenv("ENVIRONMENT") == "production":
        # Use Google Cloud Secret Manager
        return get_secret_from_manager(credential_name)
    else:
        # Use local development credentials
        return os.getenv(credential_name)

# Log only significant events at INFO level
app_logger.info("Application startup complete")

class AsyncCloudLogger:
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._buffer = []
        self._buffer_size = 100

    async def log(self, message):
        self._buffer.append(message)
        if len(self._buffer) >= self._buffer_size:
            await self.flush()

    async def flush(self):
        if self._buffer:
            await self._executor.submit(self._batch_upload)
            self._buffer = [] 

logger = get_logger()

def log_business_event(event_type: str, **kwargs):
    logger.info(
        event_type,
        timestamp=datetime.now(UTC).isoformat(),
        environment=os.getenv("ENVIRONMENT"),
        service="fashion-ecommerce",
        **kwargs
    ) 

class SensitiveDataFilter:
    SENSITIVE_FIELDS = {'password', 'credit_card', 'ssn'}
    
    @staticmethod
    def filter_sensitive_data(data: dict) -> dict:
        filtered = data.copy()
        for key in filtered:
            if any(sensitive in key.lower() for sensitive in SensitiveDataFilter.SENSITIVE_FIELDS):
                filtered[key] = '[REDACTED]'
        return filtered

def secure_audit_log(action: str, data: dict):
    filtered_data = SensitiveDataFilter.filter_sensitive_data(data)
    audit_log(action, **filtered_data) 

class ResourceEfficientLogger:
    def __init__(self):
        self._log_queue = asyncio.Queue(maxsize=1000)
        self._batch_size = 100
        self._flush_interval = 5  # seconds
    
    async def process_logs(self):
        while True:
            batch = []
            try:
                while len(batch) < self._batch_size:
                    batch.append(await self._log_queue.get())
            except asyncio.TimeoutError:
                if batch:
                    await self._flush_batch(batch)
            await asyncio.sleep(self._flush_interval) 

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

class MonitoredEndpoint:
    def __init__(self):
        self.request_counter = meter.create_counter(
            "requests",
            description="Number of requests"
        )
        self.latency_histogram = meter.create_histogram(
            "latency",
            description="Request latency"
        )

    @tracer.start_as_current_span("process_request")
    async def process_request(self, request):
        self.request_counter.add(1)
        with self.latency_histogram.record_duration():
            result = await self._process(request)
        return result

def get_secret_from_manager(secret_name: str) -> str:
    """Get secret from Google Cloud Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.getenv('GOOGLE_CLOUD_PROJECT')}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8") 