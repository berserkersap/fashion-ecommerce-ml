from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional, Set
import os
from pathlib import Path

class Settings(BaseSettings):
    # API Keys
    STRIPE_SECRET_KEY: str
    STRIPE_PUBLISHABLE_KEY: str

    # Database
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str
    DB_HOST: str

    # Firebase
    FIREBASE_PROJECT_ID: str
    FIREBASE_STORAGE_BUCKET: str
    FIREBASE_CREDENTIALS_PATH: str
    
    # Qdrant
    QDRANT_HOST: str
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "products"

    # Google Cloud
    GOOGLE_CLOUD_PROJECT: str
    BUCKET_NAME: str

    # JWT (for backup auth)
    SECRET_KEY: str
    ALGORITHM: str = "HS256"

    # Frontend
    FRONTEND_URL: str = "http://localhost:3000"

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
    USE_CUDA: bool = True

    # Logging Settings
    LOG_LEVEL: str = "INFO"
    SAMPLE_RATE: float = 0.1  # Log 10% of requests

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
    env_example = """# API Keys
STRIPE_SECRET_KEY=your_stripe_secret_key
STRIPE_PUBLISHABLE_KEY=your_stripe_publishable_key

# Database
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
DB_HOST=your_cloud_sql_instance_connection_name

# Firebase
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_STORAGE_BUCKET=your_bucket_name
FIREBASE_CREDENTIALS_PATH=path/to/firebase-credentials.json

# Qdrant
QDRANT_HOST=your_qdrant_host
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=products

# Google Cloud
GOOGLE_CLOUD_PROJECT=your_project_id
BUCKET_NAME=your_bucket_name

# JWT (for backup auth)
SECRET_KEY=your_secret_key_min_32_chars
ALGORITHM=HS256

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
USE_CUDA=True

# Logging Settings
LOG_LEVEL=INFO
SAMPLE_RATE=0.1 # Log 10% of requests
"""
    example_path = Path(".env.example")
    if not example_path.exists():
        with open(example_path, "w") as f:
            f.write(env_example)

if __name__ == "__main__":
    create_env_example() 