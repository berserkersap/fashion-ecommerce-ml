"""
Module for managing vector-based product search.

This module handles:
- Vector store initialization
- Fashion image embedding storage
- Similarity search
"""

from typing import List, Dict, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from structlog import get_logger

from .models import Product
from .logging_config import error_log, audit_log

logger = get_logger(__name__)

class VectorStore:
    """
    Manager for vector-based fashion image search using Qdrant.
    """
    
    def __init__(self, url: str = None, port: int = None):
        """
        Initialize the vector store.
        
        Args:
            url (str, optional): Qdrant server URL, defaults to config
            port (int, optional): Qdrant server port, defaults to config
        """
        from .config import get_settings
        
        settings = get_settings()
        
        url = url or settings.QDRANT_URL
        port = port or settings.QDRANT_PORT
        api_key = settings.QDRANT_API_KEY
        
        try:
            # Initialize client with API key if provided (for Qdrant Cloud)
            if api_key:
                self.client = QdrantClient(url=url, port=port, api_key=api_key)
            else:
                self.client = QdrantClient(url=url, port=port)
                
            self._ensure_collection_exists()
            logger.info("vector_store_initialized", url=url, port=port)
        except Exception as e:
            error_log(e, {"context": "vector_store_initialization"})
            raise
    
    def _ensure_collection_exists(self):
        """Ensure the fashion_image_embeddings collection exists with proper configuration."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if not any(c.name == "fashion_image_embeddings" for c in collections):
                # Create collection with proper configuration
                self.client.create_collection(
                    collection_name="fashion_image_embeddings",
                    vectors_config=models.VectorParams(
                        size=512,  # Fashion-CLIP embedding size
                        distance=models.Distance.COSINE
                    )
                )
                logger.info("fashion_image_embeddings_collection_created")
        except Exception as e:
            error_log(e, {"context": "collection_creation"})
            raise
    
    def upsert_embedding(self, product_id: int, embedding: List[float]):
        """
        Insert or update a product's embedding in the vector store.
        
        Args:
            product_id (int): Product ID
            embedding (List[float]): Fashion-CLIP embedding
        """
        try:
            if not embedding:
                raise ValueError(f"Product {product_id} has no embedding")
            
            self.client.upsert(
                collection_name="fashion_image_embeddings",
                points=[
                    models.PointStruct(
                        id=product_id,
                        vector=embedding
                    )
                ]
            )
            
            audit_log(
                action="fashion_embedding_stored",
                product_id=product_id
            )
        except Exception as e:
            error_log(e, {
                "context": "fashion_embedding_storage",
                "product_id": product_id
            })
            raise
    
    def search_similar(
        self,
        embedding: List[float],
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Search for similar fashion items using embedding.
        
        Args:
            embedding (List[float]): Query embedding
            limit (int): Maximum number of results
            score_threshold (float): Minimum similarity score
            
        Returns:
            List[Dict]: List of product IDs and similarity scores
        """
        try:
            results = self.client.search(
                collection_name="fashion_image_embeddings",
                query_vector=embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            return [
                {
                    "id": hit.id,
                    "score": hit.score
                }
                for hit in results
            ]
        except Exception as e:
            error_log(e, {"context": "vector_search"})
            raise
    
    def delete_embedding(self, product_id: int):
        """
        Delete a product's embedding from the vector store.
        
        Args:
            product_id (int): ID of product to delete
        """
        try:
            self.client.delete(
                collection_name="fashion_image_embeddings",
                points_selector=models.PointIdsList(
                    points=[product_id]
                )
            )
            
            audit_log(
                action="fashion_embedding_deleted",
                product_id=product_id
            )
        except Exception as e:
            error_log(e, {
                "context": "fashion_embedding_deletion",
                "product_id": product_id
            })
            raise

    def count_similar(
        self,
        embedding: List[float],
        score_threshold: float = 0.7
    ) -> int:
        """
        Count number of similar fashion items above threshold.
        
        Args:
            embedding (List[float]): Query embedding
            score_threshold (float): Minimum similarity score
            
        Returns:
            int: Count of similar items
        """
        try:
            count = self.client.count(
                collection_name="fashion_image_embeddings",
                vector=embedding,
                score_threshold=score_threshold
            )
            return count.count
        except Exception as e:
            error_log(e, {"context": "vector_count"})
            raise

# Global instance
vector_store = VectorStore() 