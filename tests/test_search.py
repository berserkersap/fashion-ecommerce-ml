import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from typing import List, Optional

@pytest.fixture
def mock_embeddings():
    """Fixture to provide consistent mock embeddings"""
    return np.random.rand(512).tolist()

@pytest.fixture
def search_url():
    """Base search URL"""
    return "/search/search"

def create_image_files(test_image: dict, count: int) -> List[tuple]:
    """Helper function to create multiple image files"""
    return [("images", test_image["file"]) for _ in range(count)]

def test_search_without_auth(client, search_url):
    """Test that search requires authentication"""
    response = client.post(search_url)
    assert response.status_code == 401

def test_search_no_query_no_images(client, auth_headers, search_url):
    """Test that search requires either query or images"""
    response = client.post(search_url, headers=auth_headers)
    assert response.status_code == 400
    assert "Either query text or at least one image must be provided" in response.json()["detail"]

def test_search_too_many_images(client, auth_headers, test_image, search_url):
    """Test that search has a limit on number of images"""
    files = create_image_files(test_image, 4)
    response = client.post(search_url, headers=auth_headers, files=files)
    assert response.status_code == 400
    assert "Maximum 3 images" in response.json()["detail"]

@patch("app.search.get_fashion_text_embedding")
def test_text_search(mock_text_embedding, client, auth_headers, test_products, mock_embeddings, search_url):
    """Test text-based search"""
    mock_text_embedding.return_value = mock_embeddings
    
    response = client.post(
        search_url,
        headers=auth_headers,
        data={"query": "blue jeans"}
    )
    
    assert response.status_code == 200
    assert "products" in response.json()
    assert len(response.json()["products"]) > 0

@patch("app.search.get_fashion_image_embedding")
def test_image_search(mock_image_embedding, client, auth_headers, test_image, test_products, mock_embeddings, search_url):
    """Test image-based search"""
    mock_image_embedding.return_value = mock_embeddings
    
    response = client.post(
        search_url,
        headers=auth_headers,
        files=[("images", test_image["file"])]
    )
    
    assert response.status_code == 200
    assert "products" in response.json()
    assert len(response.json()["products"]) > 0

@patch("app.search.get_fashion_text_embedding")
@patch("app.search.get_fashion_image_embedding")
def test_combined_search(
    mock_image_embedding,
    mock_text_embedding,
    client,
    auth_headers,
    test_image,
    test_products,
    mock_embeddings,
    search_url
):
    """Test combined text and image search"""
    mock_text_embedding.return_value = mock_embeddings
    mock_image_embedding.return_value = mock_embeddings
    
    response = client.post(
        search_url,
        headers=auth_headers,
        data={"query": "blue jeans", "image_weight": 0.6, "text_weight": 0.4},
        files=[("images", test_image["file"])]
    )
    
    assert response.status_code == 200
    assert "products" in response.json()
    assert len(response.json()["products"]) > 0

class TestSearchHistory:
    """Group search history related tests"""
    
    @pytest.fixture
    def history_url(self):
        return "/search/history"
    
    @patch("app.search.get_fashion_text_embedding")
    def test_search_history_creation(self, mock_text_embedding, client, auth_headers, test_products, mock_embeddings, search_url, history_url):
        """Test that search history is recorded"""
        mock_text_embedding.return_value = mock_embeddings
        
        # Perform a search
        search_response = client.post(
            search_url,
            headers=auth_headers,
            data={"query": "blue jeans"}
        )
        assert search_response.status_code == 200
        
        # Check search history
        history_response = client.get(history_url, headers=auth_headers)
        assert history_response.status_code == 200
        assert len(history_response.json()) > 0
        assert history_response.json()[0]["query_text"] == "blue jeans"
    
    @patch("app.search.get_fashion_text_embedding")
    def test_search_refinement(self, mock_text_embedding, client, auth_headers, test_products, mock_embeddings, search_url):
        """Test search refinement"""
        mock_text_embedding.return_value = mock_embeddings
        
        # Perform initial search
        response = client.post(
            search_url,
            headers=auth_headers,
            data={"query": "blue jeans"}
        )
        
        assert response.status_code == 200
        search_id = response.json()["search_id"]
        
        # Refine search
        refinement_response = client.post(
            "/search/refine",
            headers=auth_headers,
            data={
                "refinement_text": "dark blue",
                "original_query_id": search_id
            }
        )
        
        assert refinement_response.status_code == 200
        assert "products" in refinement_response.json()

class TestImageValidation:
    """Group image validation related tests"""
    
    def test_invalid_image_format(self, client, auth_headers, test_image_invalid_format, search_url):
        """Test handling of invalid image format"""
        response = client.post(
            search_url,
            headers=auth_headers,
            files=[("images", test_image_invalid_format["file"])]
        )
        
        assert response.status_code == 400
        assert "Invalid file format" in response.json()["detail"]
    
    def test_large_image(self, client, auth_headers, test_image_large, search_url):
        """Test handling of large images"""
        response = client.post(
            search_url,
            headers=auth_headers,
            files=[("images", test_image_large["file"])]
        )
        
        assert response.status_code == 400
        assert "Image dimensions" in response.json()["detail"] 