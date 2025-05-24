# E-commerce Product Management Examples

This directory contains example code demonstrating how to use the e-commerce platform's product management features.

## Features Demonstrated

### 1. Product Creation
- Creating products with images
- Generating and storing fashion embeddings
- Handling both SQL and vector database storage
- Error handling and cleanup

### 2. Product Deletion
- Safe deletion with rollback capability
- Cleanup of associated resources (images, embeddings)
- Proper database transaction handling

### 3. Rollback Support
- Rolling back product deletions within a time window
- Restoring product data and images
- Automatic cleanup of rollback data

## Example Files

- `product_operations.py`: Complete example of product lifecycle management
- `sample_data/`: Directory containing sample images and data for examples

## Usage

To run the examples:

1. Ensure your environment is properly configured:
   ```bash
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
   ```

2. Run the example:
   ```bash
   python -m examples.product_operations
   ```

## Key Features

### Product Creation
```python
product = await ops.create_product_with_image(
    name="Premium Denim Jacket",
    description="High-quality denim jacket with vintage wash",
    price=129.99,
    category="jackets",
    metadata={
        "brand": "FashionCo",
        "color": "vintage blue",
        "sizes": ["S", "M", "L", "XL"]
    },
    image=image_file
)
```

### Product Deletion with Rollback
```python
# Delete with 24-hour rollback window
result = await ops.delete_product_with_rollback(
    product_id=123,
    rollback_window=timedelta(hours=24)
)

# Rollback if needed
restored = await ops.rollback_deletion(product_id=123)
```

## Important Notes

1. **Image Handling**:
   - Images are automatically uploaded to Google Cloud Storage
   - Fashion embeddings are generated using Fashion-CLIP
   - Both SQL and vector databases are updated atomically

2. **Rollback Support**:
   - Deletions can be rolled back within the specified window
   - Rollback data is automatically cleaned up after window expires
   - Both databases and cloud storage are handled consistently

3. **Error Handling**:
   - All operations include proper cleanup on failure
   - Database transactions ensure data consistency
   - Detailed error logging for debugging

## Best Practices Demonstrated

1. **Resource Management**:
   - Proper cleanup of temporary files
   - Automatic deletion of unused cloud storage
   - Memory-efficient handling of large files

2. **Data Consistency**:
   - Atomic operations across multiple storage systems
   - Transaction management
   - Proper error handling and rollback

3. **Security**:
   - Secure file handling
   - Proper authentication checks
   - Safe cleanup of sensitive data 