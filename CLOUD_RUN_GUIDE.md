# Cloud Run Deployment Guide

This guide provides instructions for deploying the ecommerce recommendation website to Google Cloud Run.

## Prerequisites

1. Google Cloud SDK installed and configured
2. Docker installed
3. Firebase project created with Authentication enabled
4. Google Cloud SQL instance created with PostgreSQL
5. Google Cloud project with necessary APIs enabled:
   - Cloud Run
   - Cloud SQL Admin
   - Secret Manager
   - Cloud Build
   - Qdrant Cloud (or self-hosted Qdrant instance)

## Environment Files for Cloud Run

Our application uses special `.env.cloud` files that are automatically used when deploying to Cloud Run:

1. `frontend/.env.cloud` - Contains frontend configuration for Cloud Run
2. `app/.env.cloud` - Contains backend configuration for Cloud Run

These files are copied to the appropriate locations in the Dockerfile during the build process.

## Step 1: Set Firebase Environment Variables

Create a `.env` file in the `frontend` directory with your Firebase configuration:

```bash
# Get these values from your Firebase console
FIREBASE_API_KEY=your-api-key
FIREBASE_AUTH_DOMAIN=your-project-id.firebaseapp.com
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_STORAGE_BUCKET=your-project-id.appspot.com
FIREBASE_MESSAGING_SENDER_ID=your-sender-id
FIREBASE_APP_ID=your-app-id
```

## Step 2: Set Google Cloud SQL Connection Variables

Create a `.env` file in the root directory with your Cloud SQL configuration:

```bash
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_NAME=your-db-name
DB_HOST=your-cloud-sql-connection-name
GOOGLE_CLOUD_PROJECT=your-project-id
```

## Step 3: Build and Deploy to Cloud Run

Run the following commands to build and deploy:

```bash
# Build the container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ecommerce-app

# Deploy to Cloud Run
gcloud run deploy ecommerce-app \
  --image gcr.io/YOUR_PROJECT_ID/ecommerce-app \
  --platform managed \
  --region YOUR_REGION \
  --allow-unauthenticated \
  --add-cloudsql-instances=YOUR_CLOUD_SQL_INSTANCE_CONNECTION_NAME \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,FIREBASE_API_KEY=YOUR_API_KEY,FIREBASE_AUTH_DOMAIN=YOUR_AUTH_DOMAIN,FIREBASE_PROJECT_ID=YOUR_FIREBASE_PROJECT_ID,FIREBASE_STORAGE_BUCKET=YOUR_STORAGE_BUCKET,FIREBASE_MESSAGING_SENDER_ID=YOUR_SENDER_ID,FIREBASE_APP_ID=YOUR_APP_ID,DB_USER=YOUR_DB_USER,DB_PASSWORD=YOUR_DB_PASSWORD,DB_NAME=YOUR_DB_NAME,DB_HOST=YOUR_CLOUD_SQL_CONNECTION_NAME"
```

## Troubleshooting

### Connection Issues Between Frontend and Backend

When running in Cloud Run, both the frontend and backend are in the same container. The frontend should connect to the backend at `http://127.0.0.1:8000`. If you encounter connection issues:

1. Check the environment variables in the frontend `.env` file:
   ```
   BACKEND_URL=http://127.0.0.1:8000
   CLOUD_RUN_BACKEND_URL=http://127.0.0.1:8000
   ```

2. Check logs for connection errors:
   ```bash
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ecommerce-app"
   ```

### Firebase Authentication Issues

1. Ensure Firebase Authentication is enabled in your Firebase project
2. Verify the Firebase API Key and other credentials are correctly set
3. Make sure proper CORS settings are configured in Firebase Console

### Database Connection Issues

1. Verify Cloud SQL instance is running
2. Check that the Cloud SQL Proxy is correctly set up

### Qdrant Vector Database Issues

1. For self-hosted Qdrant in Cloud Run:
   
   The application is now configured to use Qdrant running in the same container or at a configured URL.
   
   In `app/.env.cloud`, we set:
   ```
   QDRANT_URL=http://127.0.0.1
   QDRANT_PORT=6333
   QDRANT_COLLECTION_NAME=fashion_image_embeddings
   ```

2. For Qdrant Cloud:
   
   Update `app/.env.cloud` with your Qdrant Cloud configuration:
   ```
   QDRANT_URL=https://your-instance.qdrant.tech
   QDRANT_PORT=6333
   QDRANT_API_KEY=your-qdrant-cloud-api-key
   QDRANT_COLLECTION_NAME=fashion_image_embeddings
   ```

3. Qdrant Health Check:
   
   You can test the Qdrant connection from Cloud Run by checking logs:
   ```bash
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=ecommerce-app AND textPayload:vector_store"
   ```
2. Ensure service account has necessary permissions
3. Check connection string format is correct

## Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Cloud SQL for PostgreSQL Documentation](https://cloud.google.com/sql/docs/postgres)
- [Firebase Authentication Documentation](https://firebase.google.com/docs/auth)
