"""
Module for handling search history and refinement functionality.
"""

from typing import List, Optional
from datetime import datetime, UTC
from sqlalchemy.orm import Session
from fastapi import HTTPException

from ..models import SearchHistory, Product
from ..schemas import SearchQuery
from ..logging_config import audit_log

class SearchHistoryManager:
    """
    Class for managing search history and refinements.
    
    This class provides methods to:
    - Record search queries
    - Track search results
    - Handle search refinements
    - Retrieve search history
    """
    
    def __init__(self, db: Session):
        """
        Initialize the search history manager.
        
        Args:
            db (Session): Database session
        """
        self.db = db
    
    def record_search(
        self,
        user_id: int,
        query_text: Optional[str],
        image_urls: List[str],
        results: List[Product],
        image_weight: float,
        text_weight: float
    ) -> SearchHistory:
        """
        Record a search query and its results.
        
        Args:
            user_id (int): ID of the user performing the search
            query_text (Optional[str]): Text query, if any
            image_urls (List[str]): List of image URLs used in the search
            results (List[Product]): List of product results
            image_weight (float): Weight given to image embeddings
            text_weight (float): Weight given to text embeddings
            
        Returns:
            SearchHistory: Created search history record
        """
        try:
            search_record = SearchHistory(
                user_id=user_id,
                query_text=query_text,
                image_urls=image_urls,
                result_product_ids=[p.id for p in results],
                image_weight=image_weight,
                text_weight=text_weight,
                timestamp=datetime.now(UTC)
            )
            
            self.db.add(search_record)
            self.db.commit()
            self.db.refresh(search_record)
            
            audit_log(
                action="search_recorded",
                user_id=user_id,
                search_id=search_record.id,
                query_text=query_text,
                image_count=len(image_urls),
                result_count=len(results)
            )
            
            return search_record
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to record search: {str(e)}"
            )
    
    def get_original_search(self, search_id: int, user_id: int) -> SearchHistory:
        """
        Retrieve the original search for refinement.
        
        Args:
            search_id (int): ID of the original search
            user_id (int): ID of the user requesting the search
            
        Returns:
            SearchHistory: Original search record
            
        Raises:
            HTTPException: If search not found or unauthorized
        """
        search = self.db.query(SearchHistory).filter(
            SearchHistory.id == search_id
        ).first()
        
        if not search:
            raise HTTPException(
                status_code=404,
                detail="Original search not found"
            )
        
        if search.user_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="Not authorized to access this search"
            )
        
        return search
    
    def get_user_history(
        self,
        user_id: int,
        limit: int = 10,
        offset: int = 0
    ) -> List[SearchHistory]:
        """
        Retrieve search history for a user.
        
        Args:
            user_id (int): ID of the user
            limit (int): Maximum number of records to return
            offset (int): Number of records to skip
            
        Returns:
            List[SearchHistory]: List of search history records
        """
        return self.db.query(SearchHistory).filter(
            SearchHistory.user_id == user_id
        ).order_by(
            SearchHistory.timestamp.desc()
        ).offset(offset).limit(limit).all()
    
    def clear_user_history(self, user_id: int) -> None:
        """
        Clear all search history for a user.
        
        Args:
            user_id (int): ID of the user
        """
        try:
            self.db.query(SearchHistory).filter(
                SearchHistory.user_id == user_id
            ).delete()
            
            self.db.commit()
            
            audit_log(
                action="search_history_cleared",
                user_id=user_id
            )
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Failed to clear search history: {str(e)}"
            ) 