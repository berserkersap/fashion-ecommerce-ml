# DEPRECATED: This file is no longer in use. The session management has been migrated to Firebase Auth.
# Please use firebase_session.py instead.
# This file is kept for reference purposes only.

from datetime import datetime, timedelta
import asyncio
from typing import Dict, Set
import logging
from .utils import delete_from_gcs
from .logging_config import error_log, audit_log

class SessionManager:
    def __init__(self, cleanup_interval: int = 60, session_timeout: int = 600):
        self.user_sessions: Dict[int, datetime] = {}  # user_id -> last_activity
        self.user_uploads: Dict[int, Set[str]] = {}  # user_id -> set of uploaded file URLs
        self.cleanup_interval = cleanup_interval  # Check every minute
        self.session_timeout = session_timeout  # 10 minutes timeout
        self.cleanup_task = None

    async def start_cleanup_task(self):
        """Start the background cleanup task"""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        audit_log(action="session_cleanup_started")

    async def stop_cleanup_task(self):
        """Stop the cleanup task"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        audit_log(action="session_cleanup_stopped")

    async def _cleanup_loop(self):
        """Background task to cleanup inactive sessions"""
        while True:
            try:
                await self._cleanup_inactive_sessions()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                error_log(e, {"context": "session_cleanup_loop"})
                await asyncio.sleep(self.cleanup_interval)

    async def _cleanup_inactive_sessions(self):
        """Clean up sessions that have been inactive for more than session_timeout"""
        current_time = datetime.utcnow()
        inactive_users = []

        for user_id, last_activity in self.user_sessions.items():
            if (current_time - last_activity) > timedelta(seconds=self.session_timeout):
                inactive_users.append(user_id)

        for user_id in inactive_users:
            await self.cleanup_user_session(user_id)

    async def cleanup_user_session(self, user_id: int):
        """Clean up a specific user's session"""
        try:
            # Get user's uploaded files
            user_files = self.user_uploads.get(user_id, set())
            
            # Delete each file from GCS
            for file_url in user_files:
                try:
                    delete_from_gcs(file_url)
                    audit_log(
                        action="file_deleted",
                        user_id=user_id,
                        file_url=file_url,
                        reason="session_timeout"
                    )
                except Exception as e:
                    error_log(e, {
                        "context": "file_deletion",
                        "user_id": user_id,
                        "file_url": file_url
                    })

            # Clean up session data
            self.user_sessions.pop(user_id, None)
            self.user_uploads.pop(user_id, None)

            audit_log(
                action="session_cleaned",
                user_id=user_id
            )
        except Exception as e:
            error_log(e, {
                "context": "session_cleanup",
                "user_id": user_id
            })

    def update_user_activity(self, user_id: int):
        """Update user's last activity timestamp"""
        self.user_sessions[user_id] = datetime.utcnow()

    def track_user_upload(self, user_id: int, file_url: str):
        """Track a file uploaded by a user"""
        if user_id not in self.user_uploads:
            self.user_uploads[user_id] = set()
        self.user_uploads[user_id].add(file_url)
        self.update_user_activity(user_id)

    def remove_user_upload(self, user_id: int, file_url: str):
        """Remove a file from user's tracked uploads"""
        if user_id in self.user_uploads:
            self.user_uploads[user_id].discard(file_url)
            self.update_user_activity(user_id)

# Create global session manager instance
session_manager = SessionManager() 