"""
Module containing FastAPI router for search functionality.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from ..database import get_db
from ..models import Product, SearchHistory
from ..schemas import SearchQuery, SearchResult, SearchRefinement
from ..auth import get_current_user
from ..agent import EcommerceAgent
from ..logging_config import log_endpoint_access, log_search_request, audit_log, error_log
from ..cleanup_manager import cleanup_manager

from .embeddings import EmbeddingGenerator
from .image_processor import ImageProcessor
from .history import SearchHistoryManager

search_router = APIRouter()

# Initialize embedding generator with default weights
embedding_generator = EmbeddingGenerator()

def get_agent(db: Session = Depends(get_db)) -> EcommerceAgent:
    """Get an instance of the e-commerce agent."""
    return EcommerceAgent(db)

@search_router.on_event("startup")
async def startup_event():
    """Start cleanup task when application starts."""
    await cleanup_manager.start()

@search_router.on_event("shutdown")
async def shutdown_event():
    """Stop cleanup task when application shuts down."""
    await cleanup_manager.stop()

@search_router.post("/search", response_model=SearchResult)
@log_search_request
@log_endpoint_access
async def search_products(
    query: Optional[str] = Form(None),
    images: List[UploadFile] = File(default=[]),
    image_weight: Optional[float] = Form(default=0.7),
    text_weight: Optional[float] = Form(default=0.3),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
    agent: EcommerceAgent = Depends(get_agent)
):
    """
    Search for products using text and/or images.
    
    Handles:
    1. Single image upload
    2. Single image + text
    3. Multiple images (up to 3)
    4. Multiple images + text
    
    Each image is processed for both visual features and textual description.
    """
    if not query and not images:
        error_log(ValueError("No query or images provided"), {
            "user_id": current_user.id
        })
        raise HTTPException(
            status_code=400,
            detail="Either query text or at least one image must be provided"
        )
    
    if len(images) > 3:
        raise HTTPException(
            status_code=400,
            detail="Maximum 3 images allowed per search"
        )
    
    try:
        # Process single image more efficiently
        if len(images) == 1:
            pil_image, temp_url = await ImageProcessor.process_image(images[0], current_user.id)
            cleanup_manager.track_upload(current_user.id, temp_url)
            
            # Get image embedding and description
            image_embedding = embedding_generator.get_image_embedding(pil_image)
            image_description = await agent.get_image_description(pil_image)
            
            # Combine text components with equal weights
            if query:
                # When user text exists: 0.7*image + 0.3*avg(description, user_text)
                text_components = [image_description, query]
                effective_query = f"Product search with image showing: {image_description}. User requirements: {query}"
            else:
                # When no user text: 0.7*image + 0.3*description
                text_components = [image_description]
                effective_query = f"Find products similar to image showing: {image_description}"
            
            result = await agent.handle_conversation(
                user_id=current_user.id,
                message=effective_query,
                image_embedding=image_embedding,
                text_components=text_components,  # Pass all text components for equal weighting
                get_text_embedding=embedding_generator.get_text_embedding,
                embedding_weights=(image_weight, text_weight)
            )
            
            # Record search history
            search_history = SearchHistory(
                user_id=current_user.id,
                query_text=query,
                image_urls=[temp_url],
                image_descriptions=[image_description],
                metadata={
                    "image_weight": image_weight,
                    "text_weight": text_weight,
                    "text_components": text_components
                }
            )
            db.add(search_history)
            db.commit()
            
            audit_log(
                action="single_image_search_completed",
                user_id=current_user.id,
                search_history_id=search_history.id,
                has_text_query=bool(query),
                has_description=bool(image_description)
            )
            
            return result
        
        # Process multiple images
        pil_images = []
        temp_image_urls = []
        image_descriptions = []
        image_embeddings = []
        
        try:
            # Process all images concurrently
            processing_results = await asyncio.gather(
                *[ImageProcessor.process_image(image, current_user.id) for image in images],
                return_exceptions=True
            )
            
            # Check for any errors in the results
            for result in processing_results:
                if isinstance(result, Exception):
                    raise result
            
            # Separate the results
            pil_images, temp_image_urls = zip(*processing_results)
            pil_images = list(pil_images)
            temp_image_urls = list(temp_image_urls)
            
            # Track uploads
            for url in temp_image_urls:
                cleanup_manager.track_upload(current_user.id, url)
            
            # Generate embeddings and descriptions concurrently
            with ThreadPoolExecutor() as executor:
                loop = asyncio.get_event_loop()
                
                # Get embeddings
                image_embeddings = await asyncio.gather(
                    *[loop.run_in_executor(executor, embedding_generator.get_image_embedding, img) 
                      for img in pil_images]
                )
                
                # Get descriptions
                image_descriptions = await asyncio.gather(
                    *[agent.get_image_description(img) for img in pil_images]
                )
            
            # Average image embeddings: 0.7 * avg(image_embeddings)
            combined_image_embedding = np.mean(image_embeddings, axis=0)
            
            # Combine text components with equal weights
            text_components = image_descriptions.copy()
            if query:
                text_components.append(query)
            
            # Create effective query combining all descriptions
            descriptions_text = ". ".join(f"Image {i+1} shows: {desc}" 
                                        for i, desc in enumerate(image_descriptions))
            if query:
                effective_query = f"Product search with following images: {descriptions_text}. User requirements: {query}"
            else:
                effective_query = f"Find products similar to images where {descriptions_text}"
            
            # Use agent to handle search with averaged embeddings and text components
            result = await agent.handle_conversation(
                user_id=current_user.id,
                message=effective_query,
                image_embedding=combined_image_embedding.tolist(),
                text_components=text_components,  # Pass all components for equal weighting
                get_text_embedding=embedding_generator.get_text_embedding,
                embedding_weights=(image_weight, text_weight)
            )
            
            # Save search history
            search_history = SearchHistory(
                user_id=current_user.id,
                query_text=query,
                image_urls=temp_image_urls,
                image_descriptions=image_descriptions,
                metadata={
                    "image_weight": image_weight,
                    "text_weight": text_weight,
                    "image_count": len(images),
                    "text_components": text_components
                }
            )
            db.add(search_history)
            db.commit()
            
            audit_log(
                action="multi_image_search_completed",
                user_id=current_user.id,
                search_history_id=search_history.id,
                has_text_query=bool(query),
                image_count=len(images),
                text_component_count=len(text_components)
            )
            
            return result
            
        except Exception as e:
            # Cleanup any uploaded files in case of error
            cleanup_tasks = [ImageProcessor.delete_from_gcs(url) for url in temp_image_urls]
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            error_log(e, {
                "context": "image_processing",
                "user_id": current_user.id,
                "image_count": len(images)
            })
            raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")
            
    except Exception as e:
        error_log(e, {
            "context": "search_products",
            "user_id": current_user.id,
            "query": query,
            "image_count": len(images)
        })
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@search_router.post("/refine", response_model=SearchResult)
@log_endpoint_access
async def refine_search(
    refinement_text: Optional[str] = Form(None),
    original_query_id: int = Form(...),
    images: List[UploadFile] = File(default=[]),
    image_weight: Optional[float] = Form(default=0.7),
    text_weight: Optional[float] = Form(default=0.3),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
    agent: EcommerceAgent = Depends(get_agent)
):
    """
    Refine an existing search with additional criteria.
    
    Args:
        refinement_text (Optional[str]): Additional text to refine the search
        original_query_id (int): ID of the original search to refine
        images (List[UploadFile]): Additional images for refinement
        image_weight (float): Weight for image embeddings (default: 0.7)
        text_weight (float): Weight for text embeddings (default: 0.3)
        db (Session): Database session
        current_user: Current authenticated user
        agent (EcommerceAgent): E-commerce agent instance
    
    Returns:
        SearchResult: Refined search results
        
    Raises:
        HTTPException: If validation fails or refinement fails
    """
    history_manager = SearchHistoryManager(db)
    original_search = history_manager.get_original_search(original_query_id, current_user.id)
    
    if not refinement_text and not images:
        raise HTTPException(
            status_code=400,
            detail="Either refinement text or images must be provided"
        )
    
    try:
        # Process new images
        processed_images = []
        new_image_urls = []
        if images:
            for image in images:
                pil_image, temp_url = await ImageProcessor.process_image(image, current_user.id)
                processed_images.append(pil_image)
                new_image_urls.append(temp_url)
                # Track temporary upload
                cleanup_manager.track_upload(current_user.id, temp_url)
        
        # Combine original and new criteria
        combined_query = f"{original_search.query_text or ''} {refinement_text or ''}"
        combined_image_urls = original_search.image_urls + new_image_urls
        
        # Generate new embeddings
        embeddings = []
        if combined_query.strip():
            text_embedding = embedding_generator.get_text_embedding(combined_query)
            embeddings.append(text_embedding)
        
        for pil_image in processed_images:
            image_embedding = embedding_generator.get_image_embedding(pil_image)
            embeddings.append(image_embedding)
        
        # Combine embeddings
        if len(embeddings) > 1:
            final_embedding = embedding_generator.combine_embeddings(
                embeddings[0],
                embeddings[1],
                weights=(image_weight, text_weight)
            )
        else:
            final_embedding = embeddings[0]
        
        # Get refined recommendations
        products = agent.get_product_recommendations(final_embedding)
        
        # Record refined search
        search_record = history_manager.record_search(
            user_id=current_user.id,
            query_text=combined_query,
            image_urls=combined_image_urls,
            results=products,
            image_weight=image_weight,
            text_weight=text_weight
        )
        
        # Track user activity
        cleanup_manager.track_user_activity(current_user.id)
        
        return SearchResult(
            search_id=search_record.id,
            products=products,
            query_text=combined_query,
            image_urls=combined_image_urls
        )
    
    except Exception as e:
        error_log(e, {
            "context": "search_refinement",
            "user_id": current_user.id,
            "original_query_id": original_query_id,
            "refinement_text": refinement_text,
            "image_count": len(images)
        })
        raise HTTPException(
            status_code=500,
            detail=f"Search refinement failed: {str(e)}"
        ) 