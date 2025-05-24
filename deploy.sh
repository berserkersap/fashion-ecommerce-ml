#!/bin/bash

# Load environment variables
set -a
source app/.env
set +a

# Configuration
PROJECT_ID=$GOOGLE_CLOUD_PROJECT
REGION="us-central1"  # Change as needed
BACKEND_SERVICE_NAME="fashion-store-backend"
FRONTEND_SERVICE_NAME="fashion-store-frontend"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Starting deployment to Google Cloud Run...${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
if ! command_exists gcloud; then
    echo -e "${RED}Error: Google Cloud SDK (gcloud) is not installed${NC}"
    exit 1
fi

# Ensure we're logged in and using the right project
echo "Configuring Google Cloud project..."
gcloud config set project $PROJECT_ID

# Create a secret for Firebase credentials
echo "Setting up Firebase credentials in Secret Manager..."
SECRET_NAME="firebase-credentials"
gcloud secrets create $SECRET_NAME --replication-policy="automatic" --data-file="./firebase-credentials.json" || \
gcloud secrets versions add $SECRET_NAME --data-file="./firebase-credentials.json"

# Build and deploy backend
echo "Building and deploying backend service..."
cd app
gcloud builds submit --tag gcr.io/$PROJECT_ID/$BACKEND_SERVICE_NAME

# Deploy backend service
gcloud run deploy $BACKEND_SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$BACKEND_SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT_ID" \
    --set-env-vars "DB_HOST=$DB_HOST" \
    --set-env-vars "DB_NAME=$DB_NAME" \
    --set-env-vars "ENVIRONMENT=production" \
    --set-env-vars "FRONTEND_URL=https://$FRONTEND_SERVICE_NAME-xxxxx-uc.a.run.app" \
    --set-secrets "DB_USER=db-user:latest" \
    --set-secrets "DB_PASSWORD=db-password:latest" \
    --set-secrets "FIREBASE_CREDENTIALS=/secrets/firebase-credentials.json" \
    --set-secrets "STRIPE_SECRET_KEY=stripe-secret:latest" \
    --set-secrets "STRIPE_WEBHOOK_SECRET=stripe-webhook-secret:latest" \
    --mount-secrets "/secrets/firebase-credentials.json=firebase-credentials:latest"

BACKEND_URL=$(gcloud run services describe $BACKEND_SERVICE_NAME --region $REGION --format 'value(status.url)')

# Build and deploy frontend
echo "Building and deploying frontend service..."
cd ../frontend
gcloud builds submit --tag gcr.io/$PROJECT_ID/$FRONTEND_SERVICE_NAME

# Deploy frontend service
gcloud run deploy $FRONTEND_SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$FRONTEND_SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars "BACKEND_URL=$BACKEND_URL" \
    --set-env-vars "NODE_ENV=production"

FRONTEND_URL=$(gcloud run services describe $FRONTEND_SERVICE_NAME --region $REGION --format 'value(status.url)')

# Update backend CORS settings with frontend URL
gcloud run services update $BACKEND_SERVICE_NAME \
    --region $REGION \
    --set-env-vars "FRONTEND_URL=$FRONTEND_URL"

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo "Backend URL: $BACKEND_URL"
echo "Frontend URL: $FRONTEND_URL" 