# Fashion E-commerce ML 

A machine learning-powered backend for fashion e-commerce, built for the ML Hackathon. This project uses Fashion-CLIP for visual search and recommendation.

Live Link (Only live for few days)- https://ecommerce-recommendation-website-871670877574.us-central1.run.app/

## Features

- 🔍 Visual Search: Search products using images with Fashion-CLIP embeddings
- 🎯 Text Search: Natural language search with semantic understanding
- 🔄 Hybrid Search: Combine image and text queries for better results
- 🛍️ Product Management: CRUD operations with image processing
- 🔐 Authentication: Firebase-based user authentication
- 💳 Cart & Checkout: Mock checkout system for hackathon demo
- 🚀 Vector Search: Fast similarity search using Qdrant
- 📊 Analytics: Basic search and user behavior tracking

## Tech Stack

- **Framework**: FastAPI
- **ML Models**: Fashion-CLIP
- **Vector Store**: Qdrant
- **Authentication**: Firebase
- **Storage**: Google Cloud Storage
- **Database**: SQLAlchemy with PostgreSQL
- **Image Processing**: Pillow
- **Deployment**: Google Cloud Run

## Quick Start

1. Clone the repository:
```bash
git clone [your-repo-url]
cd fashion-ecommerce
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

3. Install dependencies:
```bash
pip install -r Requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

5. Run the development server:
```bash
uvicorn app.main:app --reload
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development Guide

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed setup instructions.

## Deployment

See [docs/deployment.md](docs/deployment.md) for deployment instructions.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Fashion-CLIP model by [patrickjohncyh](https://github.com/patrickjohncyh/fashion-clip)
- FastAPI framework
- Qdrant vector database
- Google Cloud Platform 
