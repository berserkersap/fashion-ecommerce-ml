from fastapi import HTTPException
from typing import Any, Dict, Optional

class AppBaseException(HTTPException):
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)

class ImageProcessingError(AppBaseException):
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=f"Image processing error: {detail}")

class InvalidImageError(AppBaseException):
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=f"Invalid image: {detail}")

class StorageError(AppBaseException):
    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=f"Storage error: {detail}")

class SearchError(AppBaseException):
    def __init__(self, detail: str):
        super().__init__(status_code=500, detail=f"Search error: {detail}")

class AuthenticationError(AppBaseException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=401,
            detail=f"Authentication error: {detail}",
            headers={"WWW-Authenticate": "Bearer"}
        )

class ResourceNotFoundError(AppBaseException):
    def __init__(self, resource_type: str, resource_id: Any):
        super().__init__(
            status_code=404,
            detail=f"{resource_type} with id {resource_id} not found"
        ) 