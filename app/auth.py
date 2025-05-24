"""
Firebase authentication module.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional
import firebase_admin
from firebase_admin import credentials, auth
import os
from datetime import datetime
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from functools import lru_cache
from .logging_config import error_log, audit_log

from .database import get_db
from .models import User
from .schemas import UserCreate, Token

auth_router = APIRouter()

# Initialize Firebase Admin
cred = credentials.Certificate(os.getenv("FIREBASE_ADMIN_CREDENTIALS"))
firebase_admin.initialize_app(cred)

security = HTTPBearer()

class FirebaseUser:
    def __init__(self, uid: str, email: str, email_verified: bool):
        self.uid = uid
        self.email = email
        self.email_verified = email_verified

@lru_cache(maxsize=1000)
def verify_token(token: str) -> FirebaseUser:
    """
    Verify Firebase ID token and return user info.
    Uses LRU cache to reduce Firebase API calls.
    """
    try:
        decoded_token = auth.verify_id_token(token)
        return FirebaseUser(
            uid=decoded_token['uid'],
            email=decoded_token.get('email'),
            email_verified=decoded_token.get('email_verified', False)
        )
    except Exception as e:
        error_log(e, {"context": "firebase_token_verification"})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> FirebaseUser:
    """
    Get current authenticated user from Firebase token.
    """
    try:
        token = credentials.credentials
        firebase_user = verify_token(token)
        
        audit_log(
            action="user_authenticated",
            firebase_uid=firebase_user.uid,
            email=firebase_user.email
        )
        
        return firebase_user
    except Exception as e:
        error_log(e, {"context": "user_authentication"})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

@auth_router.post("/verify-token", response_model=Token)
async def verify_firebase_token(
    authorization: str,
    db: Session = Depends(get_db)
):
    """
    Verify Firebase token and return our custom token
    """
    user = await get_current_user(authorization, db)
    return {"access_token": authorization.replace("Bearer ", ""), "token_type": "bearer"}

@auth_router.post("/register", response_model=Token)
async def register_user(
    authorization: str,
    db: Session = Depends(get_db)
):
    """
    Register a new user using Firebase token
    """
    user = await get_current_user(authorization, db)
    return {"access_token": authorization.replace("Bearer ", ""), "token_type": "bearer"} 