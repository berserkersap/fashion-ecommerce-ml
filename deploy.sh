#!/bin/bash

# Exit on error
set -e

# Load environment variables
set -a
source .env
set +a

# Configuration
PROJECT_ID=$GOOGLE_CLOUD_PROJECT
REGION="us-central1"
BACKEND_SERVICE_NAME="ecommerce-backend"
INSTANCE_NAME="ecommerce-db"
DB_NAME="ecommerce"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting deployment to Google Cloud...${NC}"

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

# Create Cloud SQL instance if it doesn't exist
echo "Setting up Cloud SQL..."
if ! gcloud sql instances describe $INSTANCE_NAME > /dev/null 2>&1; then
    echo "Creating Cloud SQL instance..."
    gcloud sql instances create $INSTANCE_NAME \
        --database-version=POSTGRES_13 \
        --tier=db-f1-micro \
        --region=$REGION \
        --storage-size=10GB \
        --storage-type=SSD \
        --backup-start-time=23:00 \
        --availability-type=zonal \
        --root-password=$DB_ROOT_PASSWORD

    # Create database
    gcloud sql databases create $DB_NAME --instance=$INSTANCE_NAME
fi

# Set up secrets in Secret Manager
echo "Setting up secrets in Secret Manager..."
declare -a secrets=(
    "db-user:$DB_USER"
    "db-password:$DB_PASSWORD"
    "firebase-credentials:$(cat firebase-credentials.json)"
)

for secret in "${secrets[@]}"; do
    SECRET_NAME="${secret%%:*}"
    SECRET_VALUE="${secret#*:}"
    
    if ! gcloud secrets describe $SECRET_NAME > /dev/null 2>&1; then
        echo "Creating secret: $SECRET_NAME"
        echo -n "$SECRET_VALUE" | gcloud secrets create $SECRET_NAME --data-file=-
    else
        echo "Updating secret: $SECRET_NAME"
        echo -n "$SECRET_VALUE" | gcloud secrets versions add $SECRET_NAME --data-file=-
    fi
done

# Set up Cloud Storage bucket
BUCKET_NAME="$PROJECT_ID-ecommerce-storage"
if ! gsutil ls -b "gs://$BUCKET_NAME" > /dev/null 2>&1; then
    echo "Creating Cloud Storage bucket..."
    gsutil mb -l $REGION "gs://$BUCKET_NAME"
    gsutil uniformbucketlevelaccess set on "gs://$BUCKET_NAME"
fi

# Build and deploy backend
echo "Building and deploying backend service..."
gcloud builds submit --config cloudbuild.yaml

# Set up Qdrant on Cloud Run
echo "Deploying Qdrant service..."
gcloud run deploy qdrant \
    --image qdrant/qdrant:latest \
    --platform managed \
    --region $REGION \
    --memory 2Gi \
    --cpu 2 \
    --port 6333 \
    --min-instances 1 \
    --set-env-vars "QDRANT_ALLOW_CORS=true"

QDRANT_URL=$(gcloud run services describe qdrant --region $REGION --format 'value(status.url)')
BACKEND_URL=$(gcloud run services describe $BACKEND_SERVICE_NAME --region $REGION --format 'value(status.url)')

# Update backend with Qdrant URL
echo "Updating backend configuration..."
gcloud run services update $BACKEND_SERVICE_NAME \
    --region $REGION \
    --set-env-vars "QDRANT_URL=$QDRANT_URL"

echo -e "${GREEN}Deployment completed successfully!${NC}"
echo -e "${YELLOW}Important URLs:${NC}"
echo "Backend URL: $BACKEND_URL"
echo "Qdrant URL: $QDRANT_URL"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Update your frontend .env file with the backend URL"
echo "2. Deploy your frontend application"
echo "3. Configure your domain and SSL certificates if needed" 