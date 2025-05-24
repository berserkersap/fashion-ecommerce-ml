import pytest
from unittest.mock import patch, MagicMock
from firebase_admin import auth as firebase_auth

class BaseAuthTest:
    """Base class for authentication tests"""
    
    @pytest.fixture
    def protected_url(self):
        """URL of a protected endpoint to test"""
        return "/search/history"
    
    def assert_unauthorized(self, response, expected_detail: str):
        """Helper method to check unauthorized responses"""
        assert response.status_code == 401
        assert expected_detail in response.json()["detail"]

class TestAuthenticationFlow(BaseAuthTest):
    """Test basic authentication flow"""
    
    def test_protected_route_without_token(self, client, protected_url):
        """Test that protected routes require authentication"""
        response = client.get(protected_url)
        self.assert_unauthorized(response, "Not authenticated")
    
    def test_missing_auth_header(self, client, protected_url):
        """Test request without authorization header"""
        response = client.get(protected_url)
        self.assert_unauthorized(response, "Not authenticated")
    
    def test_malformed_auth_header(self, client, protected_url):
        """Test malformed authorization header"""
        headers = {"Authorization": "InvalidFormat token"}
        response = client.get(protected_url, headers=headers)
        self.assert_unauthorized(response, "Invalid authentication credentials")

class TestFirebaseTokenValidation(BaseAuthTest):
    """Test Firebase token validation"""
    
    @patch("firebase_admin.auth.verify_id_token")
    def test_valid_firebase_token(self, mock_verify_token, client, test_user, protected_url):
        """Test authentication with valid Firebase token"""
        mock_verify_token.return_value = {
            "uid": test_user.firebase_uid,
            "email": test_user.email
        }
        
        headers = {"Authorization": "Bearer valid_token"}
        response = client.get(protected_url, headers=headers)
        
        assert response.status_code == 200
        mock_verify_token.assert_called_once_with("valid_token")
    
    @patch("firebase_admin.auth.verify_id_token")
    def test_invalid_firebase_token(self, mock_verify_token, client, protected_url):
        """Test authentication with invalid Firebase token"""
        mock_verify_token.side_effect = firebase_auth.InvalidIdTokenError("Invalid token")
        
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get(protected_url, headers=headers)
        
        self.assert_unauthorized(response, "Invalid authentication credentials")

class TestUserValidation(BaseAuthTest):
    """Test user-related authentication scenarios"""
    
    @patch("firebase_admin.auth.verify_id_token")
    def test_user_not_found(self, mock_verify_token, client, protected_url):
        """Test authentication with valid token but non-existent user"""
        mock_verify_token.return_value = {
            "uid": "non_existent_uid",
            "email": "nonexistent@example.com"
        }
        
        headers = {"Authorization": "Bearer valid_token"}
        response = client.get(protected_url, headers=headers)
        
        self.assert_unauthorized(response, "User not found")
    
    @patch("firebase_admin.auth.verify_id_token")
    def test_inactive_user(self, mock_verify_token, client, test_user, db, protected_url):
        """Test authentication with inactive user account"""
        test_user.is_active = False
        db.commit()
        
        mock_verify_token.return_value = {
            "uid": test_user.firebase_uid,
            "email": test_user.email
        }
        
        headers = {"Authorization": "Bearer valid_token"}
        response = client.get(protected_url, headers=headers)
        
        self.assert_unauthorized(response, "Inactive user account")

class TestTokenErrors(BaseAuthTest):
    """Test various token error scenarios"""
    
    @patch("firebase_admin.auth.verify_id_token")
    def test_token_expired(self, mock_verify_token, client, protected_url):
        """Test authentication with expired token"""
        mock_verify_token.side_effect = firebase_auth.ExpiredIdTokenError("Token expired")
        
        headers = {"Authorization": "Bearer expired_token"}
        response = client.get(protected_url, headers=headers)
        
        self.assert_unauthorized(response, "Token has expired")
    
    @patch("firebase_admin.auth.verify_id_token")
    def test_revoked_token(self, mock_verify_token, client, protected_url):
        """Test authentication with revoked token"""
        mock_verify_token.side_effect = firebase_auth.RevokedIdTokenError("Token revoked")
        
        headers = {"Authorization": "Bearer revoked_token"}
        response = client.get(protected_url, headers=headers)
        
        self.assert_unauthorized(response, "Token has been revoked") 