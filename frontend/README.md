# Fashion E-commerce Frontend

A modern, responsive frontend for the fashion e-commerce platform with image-based search capabilities.

## Features

- Text and image-based product search
- Multiple image upload support (up to 3 images)
- Real-time image preview
- Shopping cart management
- Stripe checkout integration
- Responsive design using Tailwind CSS

## Tech Stack

- Python 3.9+
- Flask
- JavaScript (ES6+)
- Tailwind CSS
- Google Cloud Run ready

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create `.env` file:
   ```bash
   cp .env.example .env
   ```
   Update the environment variables as needed.

5. Run the development server:
   ```bash
   python app.py
   ```

## Docker Deployment

Build and run the Docker container:

```bash
docker build -t fashion-ecommerce-frontend .
docker run -p 8080:8080 fashion-ecommerce-frontend
```

## Google Cloud Run Deployment

1. Build the container:
   ```bash
   gcloud builds submit --tag gcr.io/[PROJECT_ID]/fashion-ecommerce-frontend
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy fashion-ecommerce-frontend \
     --image gcr.io/[PROJECT_ID]/fashion-ecommerce-frontend \
     --platform managed \
     --allow-unauthenticated \
     --region [REGION] \
     --set-env-vars "BACKEND_URL=[YOUR_BACKEND_URL]"
   ```

## Development

The frontend is structured as follows:

- `app.py`: Flask application and API routes
- `templates/`: HTML templates
- `static/`: Static assets (CSS, JavaScript)
  - `css/style.css`: Custom styles
  - `js/main.js`: Frontend functionality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 