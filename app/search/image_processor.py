"""
Module for handling image validation, processing, and storage.
"""

import os
from typing import Set, Tuple
from PIL import Image
import io
from fastapi import HTTPException, UploadFile

from ..logging_config import audit_log
from ..utils import upload_to_gcs
from ..firebase_session import firebase_session_manager
from ..config import get_settings

class ImageProcessor:
    """
    Class for handling image validation, processing, and storage.
    
    This class provides methods to:
    - Validate image files (size, format, dimensions)
    - Process images for search
    - Handle temporary storage
    """
    
    @classmethod
    async def validate_and_process_image(cls, image: UploadFile) -> Image.Image:
        """
        Validate and process an uploaded image without storage.
        
        Args:
            image (UploadFile): The uploaded image file
            
        Returns:
            PIL.Image.Image: Processed image
            
        Raises:
            HTTPException: If image validation fails
        """
        settings = get_settings()
        
        # Validate file extension
        file_ext = os.path.splitext(image.filename)[1].lower()
        if not file_ext or file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format '{file_ext}'. Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Read and validate image size
        contents = await image.read()
        file_size = len(contents)
        if file_size > settings.MAX_IMAGE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Image size ({file_size / 1024 / 1024:.1f}MB) exceeds maximum allowed size ({settings.MAX_IMAGE_SIZE_MB}MB)"
            )
        
        try:
            pil_image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Convert RGBA to RGB if needed
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        
        # Validate image dimensions
        width, height = pil_image.size
        if width < settings.MIN_IMAGE_SIZE_PIXELS or height < settings.MIN_IMAGE_SIZE_PIXELS:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions ({width}x{height}) too small. Minimum size is {settings.MIN_IMAGE_SIZE_PIXELS}x{settings.MIN_IMAGE_SIZE_PIXELS} pixels"
            )
        
        if width > settings.MAX_IMAGE_SIZE_PIXELS or height > settings.MAX_IMAGE_SIZE_PIXELS:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions ({width}x{height}) too large. Maximum size is {settings.MAX_IMAGE_SIZE_PIXELS}x{settings.MAX_IMAGE_SIZE_PIXELS} pixels"
            )
        
        # Reset file pointer for future reads
        await image.seek(0)
        
        return pil_image
    
    @classmethod
    async def process_image(cls, image: UploadFile, user_id: int) -> Tuple[Image.Image, str]:
        """
        Process and validate an uploaded image, including storage.
        
        Args:
            image (UploadFile): The uploaded image file
            user_id (int): ID of the user uploading the image
            
        Returns:
            Tuple[PIL.Image.Image, str]: Processed image and temporary storage URL
            
        Raises:
            HTTPException: If image validation fails
        """
        # First validate and process the image
        pil_image = await cls.validate_and_process_image(image)
        
        # Then handle storage
        contents = await image.read()
        temp_image_url = await upload_to_gcs(contents, f"temp/{user_id}_{image.filename}")
        firebase_session_manager.track_user_upload(user_id, temp_image_url)
        
        # Log processing
        audit_log(
            action="image_processing",
            user_id=user_id,
            image_name=image.filename,
            image_size=len(contents),
            image_dimensions=f"{pil_image.size[0]}x{pil_image.size[1]}",
            temp_url=temp_image_url
        )
        
        return pil_image, temp_image_url 