"""
Module for managing cleanup of temporary resources.

This module handles:
- Cleanup of temporary image uploads
- Cleanup of temporary embeddings
- Session inactivity management
"""

import asyncio
from datetime import datetime, timedelta, UTC
from typing import Dict, Set
from structlog import get_logger

from .logging_config import audit_log, error_log
from .utils import delete_from_gcs, DeleteError

logger = get_logger(__name__)

class CleanupManager:
    """
    Manager for cleaning up temporary resources.
    
    Handles automatic cleanup of:
    - Temporary image uploads after session inactivity
    - Temporary embeddings from memory
    """
    
    def __init__(self, cleanup_interval: int = 60, session_timeout: int = 600):
        """
        Initialize the cleanup manager.
        
        Args:
            cleanup_interval (int): Interval between cleanup checks in seconds (default: 60)
            session_timeout (int): Session timeout in seconds (default: 600 = 10 minutes)
        """
        self.user_sessions: Dict[int, datetime] = {}  # user_id -> last_activity
        self.user_uploads: Dict[int, Set[str]] = {}  # user_id -> set of uploaded file URLs
        self.cleanup_interval = cleanup_interval
        self.session_timeout = session_timeout
        self.cleanup_task = None
        logger.info(
            "cleanup_manager_initialized",
            cleanup_interval=cleanup_interval,
            session_timeout=session_timeout
        )
    
    async def start(self):
        """Start the cleanup task."""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        audit_log(action="cleanup_task_started")
    
    async def stop(self):
        """Stop the cleanup task."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        audit_log(action="cleanup_task_stopped")
    
    async def _cleanup_loop(self):
        """Background task to cleanup inactive sessions."""
        while True:
            try:
                await self._cleanup_inactive_sessions()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                error_log(e, {"context": "cleanup_loop"})
                await asyncio.sleep(self.cleanup_interval)
    
    async def _cleanup_inactive_sessions(self):
        """Clean up sessions that have been inactive for more than session_timeout"""
        current_time = datetime.now(UTC)
        inactive_users = []
        
        for user_id, last_activity in self.user_sessions.items():
            if (current_time - last_activity) > timedelta(seconds=self.session_timeout):
                inactive_users.append(user_id)
        
        for user_id in inactive_users:
            await self.cleanup_user_resources(user_id)
    
    async def cleanup_user_resources(self, user_id: int):
        """
        Clean up all resources for a specific user.
        
        Args:
            user_id (int): ID of the user whose resources to clean up
        """
        try:
            # Get user's uploaded files
            user_files = self.user_uploads.get(user_id, set())
            
            # Delete each file from GCS
            for file_url in user_files:
                try:
                    delete_from_gcs(file_url)
                    audit_log(
                        action="temp_file_deleted",
                        user_id=user_id,
                        file_url=file_url,
                        reason="session_timeout"
                    )
                except DeleteError as e:
                    error_log(e, {
                        "context": "temp_file_deletion",
                        "user_id": user_id,
                        "file_url": file_url
                    })
            
            # Clean up session data
            self.user_sessions.pop(user_id, None)
            self.user_uploads.pop(user_id, None)
            
            audit_log(
                action="user_resources_cleaned",
                user_id=user_id
            )
        except Exception as e:
            error_log(e, {
                "context": "user_cleanup",
                "user_id": user_id
            })
    
    def update_user_activity(self, user_id: int):
        """Update user's last activity time."""
        self.user_sessions[user_id] = datetime.now(UTC)
    
    def track_upload(self, user_id: int, file_url: str):
        """
        Track a temporary file upload.
        
        Args:
            user_id (int): ID of the user who uploaded the file
            file_url (str): URL of the uploaded file
        """
        if user_id not in self.user_uploads:
            self.user_uploads[user_id] = set()
        self.user_uploads[user_id].add(file_url)
        self.update_user_activity(user_id)
    
    def remove_upload(self, user_id: int, file_url: str):
        """
        Remove a tracked file upload.
        
        Args:
            user_id (int): ID of the user who uploaded the file
            file_url (str): URL of the uploaded file
        """
        if user_id in self.user_uploads:
            self.user_uploads[user_id].discard(file_url)
            self.update_user_activity(user_id)

# Global instance
cleanup_manager = CleanupManager() 