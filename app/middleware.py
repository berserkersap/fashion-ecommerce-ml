from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
from typing import Callable
from .logging_config import app_logger
from .config import get_settings
from .exceptions import AppBaseException

settings = get_settings()

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", str(time.time()))
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            
            # Log only sample of successful requests
            if request_id[-1] in "05":  # 20% sampling
                process_time = time.time() - start_time
                app_logger.info(
                    "Request processed",
                    request_id=request_id,
                    method=request.method,
                    url=str(request.url),
                    status_code=response.status_code,
                    processing_time=f"{process_time:.3f}s"
                )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            app_logger.error(
                "Request failed",
                request_id=request_id,
                method=request.method,
                url=str(request.url),
                error=str(e),
                processing_time=f"{process_time:.3f}s"
            )
            raise

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except AppBaseException as e:
            # Log custom exceptions
            app_logger.warning(
                "Application error",
                error_type=e.__class__.__name__,
                detail=e.detail,
                status_code=e.status_code
            )
            raise
        except Exception as e:
            # Log unexpected exceptions
            app_logger.error(
                "Unexpected error",
                error=str(e),
                error_type=e.__class__.__name__
            )
            raise 