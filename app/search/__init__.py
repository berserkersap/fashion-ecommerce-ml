"""
Search package for the e-commerce backend.

This package handles all search-related functionality including:
- Text and image-based product search
- Search refinement
- Search history tracking
- Embedding generation and combination
"""

from .router import search_router

__all__ = ['search_router'] 