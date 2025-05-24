# Deployment Guide for E-commerce ML Hackathon Project

This guide explains how to deploy the e-commerce website with ML-powered recommendations to Google Cloud Run.

## Prerequisites

1. Install Google Cloud SDK
2. Enable required APIs in Google Cloud Console:
   - Cloud Run
   - Cloud SQL
   - Cloud Storage
   - Cloud Build
   - Container Registry

## Environment Variables

Create a `.env` file with the following variables:

```env
# Firebase
FIREBASE_API_KEY=your_api_key
FIREBASE_AUTH_DOMAIN=your_project.firebaseapp.com
FIREBASE_PROJECT_ID=your_project_id

# Google Cloud
GOOGLE_CLOUD_PROJECT=your_project_id
CLOUD_SQL_CONNECTION_NAME=your_connection_name
CLOUD_STORAGE_BUCKET=your_bucket_name

# Database
DB_USER=your_db_user
DB_PASS=your_db_password
DB_NAME=your_db_name

# Stripe (Mock)
STRIPE_SECRET_KEY=your_test_key
STRIPE_WEBHOOK_SECRET=your_test_webhook_secret

# Qdrant
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Frontend
FRONTEND_URL=https://your-frontend-url
```

## Setup Steps

1. Create a Google Cloud SQL instance:
```bash
gcloud sql instances create ecommerce-db \
  --database-version=POSTGRES_13 \
  --tier=db-f1-micro \
  --region=us-central1
```

2. Create database and user:
```bash
gcloud sql databases create ecommerce_db --instance=ecommerce-db
gcloud sql users create ecommerce_user --instance=ecommerce-db --password=your_password
```

3. Create a Cloud Storage bucket:
```bash
gsutil mb -l us-central1 gs://your-bucket-name
```

4. Set up Qdrant free tier:
- Go to cloud.qdrant.io
- Create a free cluster
- Get the cluster URL and API key

5. Build and deploy backend:
```bash
# Build container
gcloud builds submit ./app --tag gcr.io/$PROJECT_ID/ecommerce-backend

# Deploy to Cloud Run
gcloud run deploy ecommerce-backend \
  --image gcr.io/$PROJECT_ID/ecommerce-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "$(cat .env | xargs)"
```

6. Build and deploy frontend:
```bash
# Build container
gcloud builds submit ./frontend --tag gcr.io/$PROJECT_ID/ecommerce-frontend

# Deploy to Cloud Run
gcloud run deploy ecommerce-frontend \
  --image gcr.io/$PROJECT_ID/ecommerce-frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "BACKEND_URL=https://ecommerce-backend-xxxxx-uc.a.run.app"
```

## Post-Deployment Steps

1. Set up Firebase Authentication:
   - Go to Firebase Console
   - Enable Email/Password authentication
   - Add your deployed domain to authorized domains

2. Set up Cloud SQL connection:
   - Create a service account for Cloud Run
   - Grant it Cloud SQL Client role
   - Update the connection string in the backend service

3. Configure Cloud Storage:
   - Grant the Cloud Run service account Storage Object Viewer role
   - Update CORS configuration for your frontend domain

4. Set up cleanup job:
```bash
# Create a Cloud Scheduler job for cleanup
gcloud scheduler jobs create http cleanup-job \
  --schedule="*/10 * * * *" \
  --uri="https://ecommerce-backend-xxxxx-uc.a.run.app/cleanup" \
  --http-method=POST
```

## Monitoring and Maintenance

1. View logs:
```bash
# Backend logs
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=ecommerce-backend"

# Frontend logs
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=ecommerce-frontend"
```

2. Monitor costs:
   - Set up budget alerts in Google Cloud Console
   - Monitor Cloud SQL usage
   - Monitor Cloud Storage usage
   - Monitor Cloud Run invocations

3. Backup database:
```bash
# Create backup
gcloud sql backups create --instance=ecommerce-db

# List backups
gcloud sql backups list --instance=ecommerce-db
```

## Troubleshooting

1. If the backend can't connect to Cloud SQL:
   - Check the connection string
   - Verify service account permissions
   - Check network configuration

2. If image uploads fail:
   - Check Cloud Storage permissions
   - Verify bucket CORS configuration
   - Check file size limits

3. If vector search is slow:
   - Check Qdrant instance size
   - Verify index configuration
   - Consider upgrading to a paid tier

4. If ML models are slow:
   - Check instance memory allocation
   - Consider using larger instance sizes
   - Monitor GPU quotas if using GPUs

## Security Notes

1. Always use HTTPS
2. Keep API keys secure
3. Regularly rotate database passwords
4. Monitor Firebase authentication logs
5. Set up rate limiting
6. Configure CORS properly
7. Use minimum required permissions for service accounts 