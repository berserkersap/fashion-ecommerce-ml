from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn
from datetime import datetime, UTC

from .database import get_db, engine, Base
from .auth import auth_router
from .products import product_router
from .search import search_router
from .cart import cart_router
from .config import Settings, get_settings
from .firebase_session import firebase_session_manager
from .logging_config import app_logger
from .middleware import RequestLoggingMiddleware, ErrorHandlingMiddleware, RateLimitMiddleware

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="E-commerce API with Image Search")

# CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "X-Total-Count"],
)

# Rate limiting middleware
app.add_middleware(
    RateLimitMiddleware,
    rate_limit=settings.RATE_LIMIT,
    time_window=settings.RATE_LIMIT_WINDOW
)

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(product_router, prefix="/products", tags=["Products"])
app.include_router(search_router, prefix="/search", tags=["Search"])
app.include_router(cart_router, prefix="/cart", tags=["Cart"])

# Add middlewares
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

@app.on_event("startup")
async def startup_event():
    # Start the session cleanup task
    await firebase_session_manager.start_cleanup_task()
    app_logger.info("Firebase session cleanup task started")
    app_logger.info("Database tables created")

@app.on_event("shutdown")
async def shutdown_event():
    # Stop the session cleanup task
    await firebase_session_manager.stop_cleanup_task()
    app_logger.info("Firebase session cleanup task stopped")

@app.get("/")
async def root():
    return {"message": "Welcome to E-commerce API with Image Search"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    Returns basic application health metrics.
    """
    settings = get_settings()
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 