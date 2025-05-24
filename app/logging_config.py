import logging
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Dict
import os
from google.cloud import logging as cloud_logging
from functools import wraps

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create loggers
app_logger = logging.getLogger('app')
audit_logger = logging.getLogger('audit')
error_logger = logging.getLogger('error')

# Set up file handlers
app_handler = logging.FileHandler('logs/app.log')
audit_handler = logging.FileHandler('logs/audit.log')
error_handler = logging.FileHandler('logs/error.log')

# Create formatters
standard_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
audit_formatter = logging.Formatter(
    '%(asctime)s - AUDIT - %(message)s'
)
error_formatter = logging.Formatter(
    '%(asctime)s - ERROR - %(name)s - %(message)s\nStack Trace: %(stack_trace)s'
)

# Set formatters for handlers
app_handler.setFormatter(standard_formatter)
audit_handler.setFormatter(audit_formatter)
error_handler.setFormatter(error_formatter)

# Add handlers to loggers
app_logger.addHandler(app_handler)
audit_logger.addHandler(audit_handler)
error_logger.addHandler(error_handler)

class CloudLoggingHandler:
    def __init__(self):
        if os.getenv("ENVIRONMENT") == "production":
            self.client = cloud_logging.Client()
            self.logger = self.client.logger('fashion-ecommerce')
        else:
            self.client = None
            self.logger = None

    def log_to_cloud(self, severity: str, message: Dict[str, Any]):
        if self.logger:
            self.logger.log_struct(
                message,
                severity=severity
            )

cloud_handler = CloudLoggingHandler()

def audit_log(action: str, user_id: int = None, **kwargs):
    """
    Log audit events with user context
    """
    audit_data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "action": action,
        "user_id": user_id,
        **kwargs
    }
    
    # Log locally
    audit_logger.info(json.dumps(audit_data))
    
    # Log to Cloud Logging in production
    cloud_handler.log_to_cloud('INFO', audit_data)

def error_log(error: Exception, context: Dict[str, Any] = None):
    """
    Log errors with context
    """
    error_data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context or {}
    }
    
    # Log locally
    error_logger.error(
        json.dumps(error_data),
        extra={"stack_trace": logging.traceback.format_exc()}
    )
    
    # Log to Cloud Logging in production
    cloud_handler.log_to_cloud('ERROR', error_data)

def log_endpoint_access(func):
    """
    Decorator to log API endpoint access
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now(UTC)
        
        try:
            result = await func(*args, **kwargs)
            
            # Log successful request
            audit_log(
                action="endpoint_access",
                endpoint=func.__name__,
                status="success",
                duration=(datetime.now(UTC) - start_time).total_seconds(),
                **kwargs
            )
            
            return result
            
        except Exception as e:
            # Log error
            error_log(
                e,
                context={
                    "endpoint": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
            )
            raise
            
    return wrapper

def log_search_request(func):
    """
    Decorator specifically for search endpoints
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now(UTC)
        
        try:
            result = await func(*args, **kwargs)
            
            # Log search details
            audit_log(
                action="search_request",
                endpoint=func.__name__,
                query=kwargs.get("query"),
                has_image=bool(kwargs.get("image")),
                image_weight=kwargs.get("image_weight"),
                text_weight=kwargs.get("text_weight"),
                duration=(datetime.now(UTC) - start_time).total_seconds(),
                user_id=kwargs.get("current_user").id if kwargs.get("current_user") else None
            )
            
            return result
            
        except Exception as e:
            # Log search error
            error_log(
                e,
                context={
                    "endpoint": func.__name__,
                    "query": kwargs.get("query"),
                    "has_image": bool(kwargs.get("image")),
                    "user_id": kwargs.get("current_user").id if kwargs.get("current_user") else None
                }
            )
            raise
            
    return wrapper 