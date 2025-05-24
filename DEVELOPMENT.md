# Development Guide for ML Hackathon

## Quick Start

1. **Clone and Setup**
```bash
git clone [repository-url]
cd fashion-ecommerce
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r Requirements.txt
```

2. **Environment Setup**
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. **Run with Docker Compose**
```bash
docker-compose up
```

## ML Model Development

### 1. Fashion-CLIP Integration
- Model location: `app/models/fashion_clip/`
- Embedding dimension: 512
- Input size: 224x224 pixels
- Supported formats: RGB images

Example usage:
```python
from app.search import get_fashion_image_embedding, get_fashion_text_embedding

# Get image embedding
image_emb = get_fashion_image_embedding(pil_image)

# Get text embedding
text_emb = get_fashion_text_embedding("blue floral dress")
```

### 2. MoonDream Image Description
- Model location: `app/models/moondream/`
- ONNX optimized for CPU inference
- Batch processing supported

Example usage:
```python
from app.ml_models import ml_models

description = ml_models.get_image_description(pil_image)
```

### 3. Vector Search (Qdrant)
- Collection: `fashion_products`
- Vector dimension: 512
- Distance metric: Cosine similarity

Example usage:
```python
from app.vector_store import vector_store

# Search by embedding
results = vector_store.search_similar(
    embedding=combined_embedding,
    limit=20,
    score_threshold=0.7
)
```

## Testing

1. **Unit Tests**
```bash
pytest tests/unit/
```

2. **Integration Tests**
```bash
pytest tests/integration/
```

3. **ML Model Tests**
```bash
pytest tests/ml/
```

## Debugging Tips

1. **Model Loading Issues**
- Check CUDA availability: `torch.cuda.is_available()`
- Verify model paths in `.env`
- Monitor memory usage with `nvidia-smi`

2. **Vector Search Issues**
- Check Qdrant connection: `curl http://localhost:6333/health`
- Verify collection exists: `curl http://localhost:6333/collections/fashion_products`
- Monitor collection stats

3. **Performance Optimization**
- Use batched inference where possible
- Monitor GPU memory with `torch.cuda.memory_summary()`
- Profile inference time:
```python
import time

start = time.time()
embedding = get_fashion_image_embedding(image)
print(f"Inference time: {time.time() - start:.2f}s")
```

## Common Issues

1. **Out of Memory**
- Reduce batch size
- Use model quantization
- Enable gradient checkpointing

2. **Slow Inference**
- Use ONNX Runtime
- Enable TensorRT if available
- Implement caching

3. **Poor Search Results**
- Check embedding normalization
- Verify distance metric
- Adjust score threshold

## Development Workflow

1. **Feature Development**
```bash
git checkout -b feature/my-feature
# Make changes
pytest tests/  # Run tests
black app/ tests/  # Format code
git commit -m "Add feature description"
git push origin feature/my-feature
```

2. **Model Updates**
```bash
# Update model
python scripts/download_models.py
python scripts/optimize_models.py
python scripts/test_models.py
```

## Monitoring

1. **ML Metrics**
- Model inference time
- Embedding quality scores
- Search relevance metrics

2. **System Metrics**
- GPU utilization
- Memory usage
- Request latency

3. **Business Metrics**
- Search success rate
- Click-through rate
- Conversion rate

## Resources

1. **Documentation**
- [Fashion-CLIP Paper](https://arxiv.org/abs/2204.03972)
- [MoonDream Documentation](https://github.com/vikhyat/moondream)
- [Qdrant Docs](https://qdrant.tech/documentation/)

2. **Tutorials**
- [Vector Search Best Practices](https://qdrant.tech/documentation/tutorials/search-basics/)
- [ONNX Optimization Guide](https://onnxruntime.ai/docs/performance/model-optimizations.html)
- [FastAPI ML Deployment](https://fastapi.tiangolo.com/advanced/using-dependencies/) 