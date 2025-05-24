"""
E-commerce agent for handling product recommendations and user interactions.
"""

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch
import os
from typing import List, Dict, Optional, Callable, Tuple
from sqlalchemy.orm import Session
from .models import Product, SearchHistory, CartItem
from .sql_agent import natural_language_query
from .cart import add_to_cart
from .vector_store import vector_store
from .logging_config import error_log, audit_log
from .ml_models import ml_models  # Import singleton instance
import numpy as np

class EcommerceAgent:
    def __init__(self, db: Session):
        """
        Initialize the e-commerce agent.
        
        Args:
            db (Session): Database session
        """
        self.db = db
        
        # Initialize conversation LLM
        self.conversation_llm = HuggingFacePipeline(
            pipeline=pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                torch_dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        )
        
        # Initialize memory with metadata about weights
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Define tools
        self.tools = [
            Tool(
                name="search_products",
                func=self._search_products,
                description="Search for products using text and image embeddings with customizable weights"
            ),
            Tool(
                name="refine_search",
                func=self._refine_search,
                description="Refine search results based on user preferences"
            ),
            Tool(
                name="add_to_cart",
                func=self._add_to_cart,
                description="Add products to user's cart"
            )
        ]
        
        # Create agent with fashion-specific prompt
        prompt = PromptTemplate.from_template("""
        You are a helpful fashion e-commerce assistant. Help users find fashion products and make style recommendations.
        When processing image and text inputs together, I prioritize image features (70%) while considering text modifications (30%).
        
        Current conversation:
        {chat_history}
        
        User: {input}
        Assistant: Let me help you find the perfect fashion items for you. I'll use my fashion-specific tools to find the best products.
        {agent_scratchpad}
        """)
        
        self.agent = create_react_agent(
            llm=self.conversation_llm,
            tools=self.tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )
    
    def get_image_description(self, image) -> str:
        """
        Get description of image using Moondream model.
        
        Args:
            image: Input image
            
        Returns:
            str: Image description
        """
        return ml_models.get_image_description(image)
    
    async def _search_products(
        self, 
        query: Optional[str] = None,
        image_embedding: Optional[List[float]] = None,
        get_text_embedding: Optional[Callable] = None,
        embedding_weights: Tuple[float, float] = (0.7, 0.3),
        filters: Optional[Dict] = None,
        page: int = 1,
        per_page: int = 20,
        sort_by: str = "relevance"
    ) -> Dict:
        """
        Search products using text and/or image embeddings with custom weights.
        Supports traditional e-commerce search features like filtering, pagination and sorting.
        
        Args:
            query (Optional[str]): Text query, optional for image-only search
            image_embedding (Optional[List[float]]): Image embedding
            get_text_embedding (Optional[Callable]): Function to get text embedding
            embedding_weights (Tuple[float, float]): Weights for (image, text)
            filters (Optional[Dict]): Filter conditions (price range, categories, sizes, etc.)
            page (int): Page number for pagination
            per_page (int): Number of items per page
            sort_by (str): Sorting method ("relevance", "price_low", "price_high", "newest")
            
        Returns:
            Dict: {
                "products": List of products,
                "total": Total number of matching products,
                "page": Current page,
                "total_pages": Total number of pages,
                "filters_applied": Applied filters,
                "suggested_filters": Suggested filters based on results
            }
        """
        try:
            if not query and not image_embedding:
                raise ValueError("Either query text or image embedding must be provided")

            # Step 1: Handle exact keyword matches first
            if query:
                # Split query into keywords
                keywords = query.lower().split()
                
                # Start with SQL query for exact matches
                base_query = self.db.query(Product)
                
                # Apply keyword filters
                for keyword in keywords:
                    base_query = base_query.filter(
                        (Product.name.ilike(f"%{keyword}%")) |
                        (Product.description.ilike(f"%{keyword}%")) |
                        (Product.category.ilike(f"%{keyword}%")) |
                        (Product.brand.ilike(f"%{keyword}%"))
                    )
                
                # Apply user filters if provided
                if filters:
                    if 'price_range' in filters:
                        min_price, max_price = filters['price_range']
                        base_query = base_query.filter(
                            Product.price.between(min_price, max_price)
                        )
                    if 'categories' in filters:
                        base_query = base_query.filter(
                            Product.category.in_(filters['categories'])
                        )
                    if 'brands' in filters:
                        base_query = base_query.filter(
                            Product.brand.in_(filters['brands'])
                        )
                    if 'sizes' in filters:
                        base_query = base_query.filter(
                            Product.available_sizes.overlap(filters['sizes'])
                        )
                
                # Get exact matches
                exact_matches = base_query.all()
                exact_match_ids = {p.id for p in exact_matches}
                
                # If we have enough exact matches, prioritize them
                if len(exact_matches) >= per_page:
                    total_exact = len(exact_matches)
                    start_idx = (page - 1) * per_page
                    end_idx = start_idx + per_page
                    
                    # Apply sorting
                    if sort_by == "price_low":
                        exact_matches.sort(key=lambda p: p.price)
                    elif sort_by == "price_high":
                        exact_matches.sort(key=lambda p: -p.price)
                    elif sort_by == "newest":
                        exact_matches.sort(key=lambda p: p.created_at, reverse=True)
                    
                    paginated_products = exact_matches[start_idx:end_idx]
                    
                    # Get suggested filters from the full result set
                    suggested_filters = self._get_suggested_filters(exact_matches)
                    
                    return {
                        "products": paginated_products,
                        "total": total_exact,
                        "page": page,
                        "total_pages": (total_exact + per_page - 1) // per_page,
                        "filters_applied": filters or {},
                        "suggested_filters": suggested_filters,
                        "search_type": "exact_match"
                    }

            # Step 2: If not enough exact matches, use vector search
            text_embedding = None
            if query and get_text_embedding:
                text_embedding = get_text_embedding(query)
            
            final_embedding = text_embedding
            if image_embedding:
                if text_embedding:
                    # Combine embeddings with weights
                    image_weight, text_weight = embedding_weights
                    final_embedding = [
                        (image_weight * i + text_weight * t)
                        for i, t in zip(image_embedding, text_embedding)
                    ]
                    # Normalize
                    norm = sum(x * x for x in final_embedding) ** 0.5
                    final_embedding = [x / norm for x in final_embedding]
                else:
                    final_embedding = image_embedding

            # Get similar products from vector store with pagination
            search_results = vector_store.search_similar(
                embedding=final_embedding,
                limit=per_page,
                offset=(page - 1) * per_page,
                score_threshold=0.5  # Lower threshold for more results
            )
            
            # Get full product details
            product_ids = [result["id"] for result in search_results]
            products = self.db.query(Product).filter(Product.id.in_(product_ids))
            
            # Apply filters to vector search results
            if filters:
                for key, value in filters.items():
                    if key == 'price_range':
                        min_price, max_price = value
                        products = products.filter(Product.price.between(min_price, max_price))
                    elif key == 'categories':
                        products = products.filter(Product.category.in_(value))
                    elif key == 'brands':
                        products = products.filter(Product.brand.in_(value))
                    elif key == 'sizes':
                        products = products.filter(Product.available_sizes.overlap(value))
            
            products = products.all()
            
            # Sort products
            if sort_by == "price_low":
                products.sort(key=lambda p: p.price)
            elif sort_by == "price_high":
                products.sort(key=lambda p: -p.price)
            elif sort_by == "newest":
                products.sort(key=lambda p: p.created_at, reverse=True)
            else:  # "relevance"
                # Sort by search score
                id_to_score = {result["id"]: result["score"] for result in search_results}
                products.sort(key=lambda p: id_to_score[p.id], reverse=True)
            
            # Get total count for pagination
            total_count = vector_store.count_similar(
                embedding=final_embedding,
                score_threshold=0.5
            )
            
            # Get suggested filters
            suggested_filters = self._get_suggested_filters(products)
            
            audit_log(
                action="product_search",
                query=query,
                has_image=bool(image_embedding),
                result_count=len(products),
                page=page,
                filters=filters,
                sort_by=sort_by
            )
            
            return {
                "products": products,
                "total": total_count,
                "page": page,
                "total_pages": (total_count + per_page - 1) // per_page,
                "filters_applied": filters or {},
                "suggested_filters": suggested_filters,
                "search_type": "vector_similarity"
            }
            
        except Exception as e:
            error_log(e, {
                "context": "product_search",
                "query": query,
                "has_image": bool(image_embedding),
                "page": page,
                "filters": filters,
                "sort_by": sort_by
            })
            raise

    def _get_suggested_filters(self, products: List[Product]) -> Dict:
        """
        Generate suggested filters based on the result set.
        """
        categories = set()
        brands = set()
        sizes = set()
        min_price = float('inf')
        max_price = 0
        
        for product in products:
            categories.add(product.category)
            brands.add(product.brand)
            sizes.update(product.available_sizes)
            min_price = min(min_price, product.price)
            max_price = max(max_price, product.price)
            
        return {
            "categories": list(categories),
            "brands": list(brands),
            "sizes": list(sizes),
            "price_range": {
                "min": min_price,
                "max": max_price,
                "steps": [
                    min_price,
                    min_price + (max_price - min_price) * 0.25,
                    min_price + (max_price - min_price) * 0.5,
                    min_price + (max_price - min_price) * 0.75,
                    max_price
                ]
            }
        }
    
    async def _refine_search(
        self, 
        original_results: List[Product], 
        refinement: str,
        get_text_embedding: Optional[Callable] = None,
        embedding_weights: Tuple[float, float] = (0.7, 0.3)
    ) -> List[Product]:
        """
        Refine search results based on user preferences.
        
        Args:
            original_results (List[Product]): Original search results
            refinement (str): Refinement query
            get_text_embedding (Optional[Callable]): Function to get text embedding
            embedding_weights (Tuple[float, float]): Weights for (image, text)
            
        Returns:
            List[Product]: Refined product list
        """
        try:
            if not get_text_embedding:
                raise ValueError("Text embedding function must be provided")
                
            # Combine original results with refinement query
            refined_query = f"From these products: {[p.name for p in original_results]}, {refinement}"
            return await self._search_products(
                refined_query, 
                get_text_embedding=get_text_embedding,
                embedding_weights=embedding_weights
            )
        except Exception as e:
            error_log(e, {
                "context": "search_refinement",
                "original_count": len(original_results),
                "refinement": refinement
            })
            raise
    
    async def _add_to_cart(self, user_id: int, product_ids: List[int]) -> List[CartItem]:
        """
        Add products to user's cart.
        
        Args:
            user_id (int): User ID
            product_ids (List[int]): List of product IDs to add
            
        Returns:
            List[CartItem]: Added cart items
        """
        try:
            cart_items = []
            for product_id in product_ids:
                cart_item = add_to_cart(self.db, user_id, product_id, 1)
                cart_items.append(cart_item)
                
            audit_log(
                action="products_added_to_cart",
                user_id=user_id,
                product_ids=product_ids
            )
            
            return cart_items
        except Exception as e:
            error_log(e, {
                "context": "add_to_cart",
                "user_id": user_id,
                "product_ids": product_ids
            })
            raise
    
    async def handle_conversation(
        self, 
        user_id: int, 
        message: str, 
        image_embedding: Optional[List[float]] = None,
        text_components: Optional[List[str]] = None,
        get_text_embedding: Optional[Callable] = None,
        embedding_weights: Tuple[float, float] = (0.7, 0.3)
    ) -> Dict:
        """
        Handle user conversation and return recommendations.
        
        Args:
            user_id (int): User ID
            message (str): Combined message for context
            image_embedding (Optional[List[float]]): Combined image embedding
            text_components (Optional[List[str]]): List of text components (descriptions, query)
            get_text_embedding (Optional[Callable]): Function to get text embedding
            embedding_weights (Tuple[float, float]): Weights for (image, text)
            
        Returns:
            Dict: Response with recommendations
        """
        try:
            # Process text components if provided
            text_embedding = None
            if text_components:
                if not get_text_embedding:
                    raise ValueError("Text embedding function must be provided for text components")
                
                # Get embeddings for all text components
                component_embeddings = [get_text_embedding(text) for text in text_components]
                
                # Average the text embeddings with equal weights
                text_embedding = np.mean(component_embeddings, axis=0).tolist()
            
            # Add weights to memory for context
            self.memory.save_context(
                {"input": f"Using weights: {embedding_weights} for search. Text components: {len(text_components) if text_components else 0}"},
                {"output": "Weights and components saved for context"}
            )
            
            # Run agent with combined embeddings
            response = await self.agent_executor.arun(
                input=message,
                image_embedding=image_embedding,
                text_embedding=text_embedding,
                embedding_weights=embedding_weights
            )
            
            return {
                "response": response,
                "weights_used": {
                    "image_weight": embedding_weights[0],
                    "text_weight": embedding_weights[1]
                },
                "components_used": {
                    "image_count": 1 if image_embedding else 0,
                    "text_components": len(text_components) if text_components else 0
                }
            }
        except Exception as e:
            error_log(e, {
                "context": "conversation_handling",
                "user_id": user_id,
                "message": message,
                "has_image": bool(image_embedding),
                "text_component_count": len(text_components) if text_components else 0
            })
            raise 