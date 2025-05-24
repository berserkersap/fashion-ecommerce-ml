"""
Utility functions for file handling and data formatting.

This module provides utility functions for:
- Google Cloud Storage operations (upload, download, delete)
- Data formatting and validation
"""

from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
import os
from datetime import datetime, UTC
import uuid
import aiohttp
from typing import Union, Optional
from fastapi import HTTPException

# Initialize Google Cloud Storage client
storage_client: storage.Client = storage.Client()
bucket_name: str = os.getenv("BUCKET_NAME")
if not bucket_name:
    raise ValueError("BUCKET_NAME environment variable not set")
bucket: Bucket = storage_client.bucket(bucket_name)

class StorageError(Exception):
    """Base exception for storage-related errors."""
    pass

class UploadError(StorageError):
    """Exception raised when file upload fails."""
    pass

class DownloadError(StorageError):
    """Exception raised when file download fails."""
    pass

class DeleteError(StorageError):
    """Exception raised when file deletion fails."""
    pass

async def upload_to_gcs(
    file_content: Union[bytes, str],
    original_filename: str,
    content_type: Optional[str] = None
) -> str:
    """
    Upload a file to Google Cloud Storage and return its public URL.
    
    Args:
        file_content (Union[bytes, str]): Content of the file to upload
        original_filename (str): Original name of the file
        content_type (Optional[str]): Content type of the file. If not provided,
                                    will be inferred from file extension.
    
    Returns:
        str: Public URL of the uploaded file
        
    Raises:
        UploadError: If the upload fails
        ValueError: If the filename is invalid
    """
    try:
        # Validate filename
        if not original_filename or '.' not in original_filename:
            raise ValueError("Invalid filename")
        
        # Generate a unique filename
        extension = original_filename.split('.')[-1].lower()
        timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        unique_filename = f"{timestamp}_{unique_id}.{extension}"
        
        # Determine content type
        if content_type is None:
            content_type = f"image/{extension}"
        
        # Create and configure blob
        blob: Blob = bucket.blob(f"products/{unique_filename}")
        
        # Upload the file
        blob.upload_from_string(
            file_content,
            content_type=content_type
        )
        
        # Make the blob publicly readable
        blob.make_public()
        
        return blob.public_url
    
    except ValueError as e:
        raise ValueError(f"Invalid input: {str(e)}")
    except Exception as e:
        raise UploadError(f"Failed to upload file: {str(e)}")

async def download_from_gcs(url: str) -> bytes:
    """
    Download a file from Google Cloud Storage using its public URL.
    
    Args:
        url (str): Public URL of the file to download
    
    Returns:
        bytes: Content of the downloaded file
        
    Raises:
        DownloadError: If the download fails
        ValueError: If the URL is invalid
    """
    if not url.startswith('http'):
        raise ValueError("Invalid URL format")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                elif response.status == 404:
                    raise DownloadError("File not found")
                else:
                    raise DownloadError(f"Download failed with status {response.status}")
    except aiohttp.ClientError as e:
        raise DownloadError(f"Network error: {str(e)}")
    except Exception as e:
        raise DownloadError(f"Failed to download file: {str(e)}")

def delete_from_gcs(url: str) -> None:
    """
    Delete a file from Google Cloud Storage using its public URL.
    
    Args:
        url (str): Public URL of the file to delete
    
    Raises:
        DeleteError: If the deletion fails
        ValueError: If the URL is invalid
    """
    if not url or bucket_name not in url:
        raise ValueError("Invalid GCS URL")
    
    try:
        # Extract blob name from URL
        blob_name = url.split(f"{bucket_name}/")[-1]
        if not blob_name:
            raise ValueError("Could not extract blob name from URL")
        
        blob: Blob = bucket.blob(blob_name)
        
        # Check if blob exists before deleting
        if not blob.exists():
            raise DeleteError("File not found")
        
        blob.delete()
    except ValueError as e:
        raise ValueError(f"Invalid input: {str(e)}")
    except Exception as e:
        raise DeleteError(f"Failed to delete file: {str(e)}")

def format_price(
    price: float,
    currency_symbol: str = "₹",
    decimal_places: int = 2
) -> str:
    """
    Format a price with currency symbol and specified decimal places.
    
    Args:
        price (float): Price to format
        currency_symbol (str, optional): Currency symbol to use. Defaults to "₹"
        decimal_places (int, optional): Number of decimal places. Defaults to 2
    
    Returns:
        str: Formatted price string
        
    Raises:
        ValueError: If price is negative or decimal_places is invalid
    """
    if price < 0:
        raise ValueError("Price cannot be negative")
    if decimal_places < 0:
        raise ValueError("Decimal places cannot be negative")
    
    return f"{currency_symbol}{price:.{decimal_places}f}" 