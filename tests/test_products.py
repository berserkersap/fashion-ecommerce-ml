import pytest
from unittest.mock import patch, MagicMock

class BaseProductTest:
    """Base class for product tests"""
    
    @pytest.fixture
    def base_url(self):
        """Base URL for product endpoints"""
        return "/products"
    
    @pytest.fixture
    def sample_product_data(self):
        """Sample product data for creation/update tests"""
        return {
            "name": "Test Product",
            "description": "A test product",
            "price": "49.99",
            "category": "test",
            "metadata": '{"brand": "TestBrand", "color": "red"}'
        }
    
    def assert_product_response(self, product, expected_name, expected_price):
        """Helper method to check product response"""
        assert product["name"] == expected_name
        assert product["price"] == expected_price

class TestProductListing(BaseProductTest):
    """Test product listing functionality"""
    
    def test_list_products(self, client, test_products, base_url):
        """Test listing all products"""
        response = client.get(f"{base_url}/")
        
        assert response.status_code == 200
        products = response.json()
        assert len(products) == 2
        self.assert_product_response(products[0], "Blue Jeans", 79.99)
        self.assert_product_response(products[1], "White T-Shirt", 29.99)
    
    def test_list_products_with_category(self, client, test_products, base_url):
        """Test listing products filtered by category"""
        response = client.get(f"{base_url}/?category=pants")
        
        assert response.status_code == 200
        products = response.json()
        assert len(products) == 1
        self.assert_product_response(products[0], "Blue Jeans", 79.99)
        assert products[0]["category"] == "pants"
    
    def test_list_products_pagination(self, client, test_products, base_url):
        """Test product listing pagination"""
        response = client.get(f"{base_url}/?skip=1&limit=1")
        
        assert response.status_code == 200
        products = response.json()
        assert len(products) == 1
        self.assert_product_response(products[0], "White T-Shirt", 29.99)

class TestProductRetrieval(BaseProductTest):
    """Test product retrieval functionality"""
    
    def test_get_product_by_id(self, client, test_products, base_url):
        """Test getting a single product by ID"""
        product_id = test_products[0].id
        response = client.get(f"{base_url}/{product_id}")
        
        assert response.status_code == 200
        self.assert_product_response(response.json(), "Blue Jeans", 79.99)
    
    def test_get_nonexistent_product(self, client, base_url):
        """Test getting a product that doesn't exist"""
        response = client.get(f"{base_url}/999")
        
        assert response.status_code == 404
        assert "Product not found" in response.json()["detail"]

class TestProductManagement(BaseProductTest):
    """Test product management functionality"""
    
    @patch("app.utils.upload_to_gcs")
    def test_create_product(self, mock_upload, client, auth_headers, test_image, base_url, sample_product_data):
        """Test creating a new product"""
        mock_upload.return_value = "https://storage.googleapis.com/test-bucket/new-product.jpg"
        
        files = [("image", test_image["file"])]
        response = client.post(
            f"{base_url}/",
            headers=auth_headers,
            data=sample_product_data,
            files=files
        )
        
        assert response.status_code == 201
        product = response.json()
        self.assert_product_response(product, sample_product_data["name"], float(sample_product_data["price"]))
        assert product["image_url"] == mock_upload.return_value
    
    def test_create_product_without_auth(self, client, base_url):
        """Test that product creation requires authentication"""
        response = client.post(f"{base_url}/")
        assert response.status_code == 401
    
    @patch("app.utils.delete_from_gcs")
    def test_delete_product(self, mock_delete, client, auth_headers, test_products, base_url):
        """Test deleting a product"""
        product_id = test_products[0].id
        
        response = client.delete(
            f"{base_url}/{product_id}",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        assert "Product deleted" in response.json()["message"]
        mock_delete.assert_called_once_with(test_products[0].image_url)
    
    def test_delete_nonexistent_product(self, client, auth_headers, base_url):
        """Test deleting a product that doesn't exist"""
        response = client.delete(
            f"{base_url}/999",
            headers=auth_headers
        )
        
        assert response.status_code == 404
        assert "Product not found" in response.json()["detail"]
    
    @patch("app.utils.upload_to_gcs")
    def test_update_product(self, mock_upload, client, auth_headers, test_products, test_image, base_url):
        """Test updating a product"""
        product_id = test_products[0].id
        mock_upload.return_value = "https://storage.googleapis.com/test-bucket/updated-product.jpg"
        
        update_data = {
            "name": "Updated Jeans",
            "price": "89.99",
            "metadata": '{"brand": "TestBrand", "color": "dark-blue"}'
        }
        
        files = [("image", test_image["file"])]
        response = client.patch(
            f"{base_url}/{product_id}",
            headers=auth_headers,
            data=update_data,
            files=files
        )
        
        assert response.status_code == 200
        product = response.json()
        self.assert_product_response(product, "Updated Jeans", 89.99)
        assert product["image_url"] == mock_upload.return_value

class TestProductValidation(BaseProductTest):
    """Test product validation"""
    
    def test_invalid_product_data(self, client, auth_headers, base_url):
        """Test creating a product with invalid data"""
        invalid_data = {
            "name": "",  # Empty name
            "price": "invalid",  # Invalid price
            "category": "test"
        }
        
        response = client.post(
            f"{base_url}/",
            headers=auth_headers,
            data=invalid_data
        )
        
        assert response.status_code == 422  # Validation error 