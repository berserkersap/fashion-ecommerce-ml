from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, Set
import os
from pathlib import Path
import torch

class Settings(BaseSettings):
    # Database (from Secret Manager in production)
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str
    DB_HOST: str

    # Firebase
    FIREBASE_PROJECT_ID: str
    FIREBASE_STORAGE_BUCKET: str
    # In Cloud Run, this will be mounted from Secret Manager
    FIREBASE_CREDENTIALS_PATH: str = "/secrets/firebase-credentials.json"
    
    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION_NAME: str = "fashion_image_embeddings"  # Match the collection name in vector_store.py
    QDRANT_API_KEY: Optional[str] = None  # Required for Qdrant Cloud
    QDRANT_ALLOW_CORS: bool = True

    # Google Cloud
    GOOGLE_CLOUD_PROJECT: str
    BUCKET_NAME: str

    # Frontend
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # Image Processing Settings
    MAX_IMAGE_SIZE_MB: int = 5
    MAX_IMAGE_SIZE_BYTES: int = MAX_IMAGE_SIZE_MB * 1024 * 1024
    ALLOWED_EXTENSIONS: Set[str] = {'.jpg', '.jpeg', '.png'}
    MIN_IMAGE_SIZE_PIXELS: int = 100
    MAX_IMAGE_SIZE_PIXELS: int = 4096
    MAX_IMAGES_PER_REQUEST: int = 3

    # Search Settings
    IMAGE_WEIGHT: float = 0.7
    TEXT_WEIGHT: float = 0.3

    # Session Settings
    SESSION_TIMEOUT: int = 600  # 10 minutes in seconds
    CLEANUP_INTERVAL: int = 60  # 1 minute in seconds

    # Model Settings
    FASHION_CLIP_MODEL: str = "patrickjohncyh/fashion-clip"
    # Automatically determine CUDA availability
    USE_CUDA: bool = torch.cuda.is_available()

    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    SAMPLE_RATE: float = 0.1  # Log 10% of requests

    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """
    Create cached instance of settings.
    This way we don't have to load the environment every time we need settings.
    """
    return Settings()

# Create .env.example file if it doesn't exist
def create_env_example():
    env_example = """# Database
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
DB_HOST=your_cloud_sql_instance_connection_name

# Firebase
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_STORAGE_BUCKET=your_bucket_name
FIREBASE_CREDENTIALS_PATH=/secrets/firebase-credentials.json

# Qdrant
QDRANT_URL=http://localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=fashion_image_embeddings
QDRANT_API_KEY=your_qdrant_api_key  # Required for Qdrant Cloud
QDRANT_ALLOW_CORS=true

# Google Cloud
GOOGLE_CLOUD_PROJECT=your_project_id
BUCKET_NAME=your_bucket_name

# Frontend
FRONTEND_URL=http://localhost:3000

# Image Processing Settings
MAX_IMAGE_SIZE_MB=5
MAX_IMAGE_SIZE_BYTES=MAX_IMAGE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS=.jpg,.jpeg,.png
MIN_IMAGE_SIZE_PIXELS=100
MAX_IMAGE_SIZE_PIXELS=4096
MAX_IMAGES_PER_REQUEST=3

# Search Settings
IMAGE_WEIGHT=0.7
TEXT_WEIGHT=0.3

# Session Settings
SESSION_TIMEOUT=600 # 10 minutes in seconds
CLEANUP_INTERVAL=60 # 1 minute in seconds

# Model Settings
FASHION_CLIP_MODEL=patrickjohncyh/fashion-clip

# Logging Settings
LOG_LEVEL=INFO
SAMPLE_RATE=0.1 # Log 10% of requests

# Environment
ENVIRONMENT=development
"""
    example_path = Path(".env.example")
    if not example_path.exists():
        with open(example_path, "w") as f:
            f.write(env_example)

if __name__ == "__main__":
    create_env_example() 