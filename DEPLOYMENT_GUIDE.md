# E-commerce Recommendation System Deployment Guide

This guide provides step-by-step instructions for deploying the e-commerce recommendation system on Google Cloud Platform.

## Prerequisites

1. Google Cloud Account with billing enabled
2. Google Cloud SDK installed locally
3. Docker installed locally
4. Firebase project created
5. Qdrant Cloud account (free tier)

## Local Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and configure environment files:
```bash
# Copy environment template
cp .env.example .env
# Edit .env with your configuration
```

3. Set up Google Cloud credentials:
```bash
# Create credentials directory
mkdir -p credentials/development
# Create service account and download key
gcloud iam service-accounts create ecommerce-dev
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ecommerce-dev@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/editor"
gcloud iam service-accounts keys create credentials/development/google_cloud.json \
    --iam-account=ecommerce-dev@$PROJECT_ID.iam.gserviceaccount.com
```

4. Start local development environment:
```bash
docker-compose up --build
```

## Production Deployment

### 1. Initial Setup

1. Enable required Google Cloud APIs:
```bash
gcloud services enable \
    run.googleapis.com \
    sql-component.googleapis.com \
    sqladmin.googleapis.com \
    secretmanager.googleapis.com \
    cloudbuild.googleapis.com \
    storage.googleapis.com
```

2. Create service account for production:
```bash
# Create service account
gcloud iam service-accounts create ecommerce-backend \
    --display-name="E-commerce Backend Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ecommerce-backend@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/cloudsql.client"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ecommerce-backend@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ecommerce-backend@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### 2. Infrastructure Setup

1. Create Cloud SQL instance:
```bash
gcloud sql instances create ecommerce-db \
    --database-version=POSTGRES_13 \
    --tier=db-f1-micro \
    --region=us-central1 \
    --storage-size=10GB \
    --storage-type=SSD \
    --backup-start-time=23:00 \
    --availability-type=zonal
```

2. Create Cloud Storage bucket:
```bash
gsutil mb -l us-central1 gs://$PROJECT_ID-ecommerce-storage
gsutil uniformbucketlevelaccess set on gs://$PROJECT_ID-ecommerce-storage
```

3. Set up Secret Manager secrets:
```bash
# Create secrets
echo -n "your-db-user" | gcloud secrets create db-user --data-file=-
echo -n "your-db-password" | gcloud secrets create db-password --data-file=-
gcloud secrets create firebase-credentials --data-file=firebase-credentials.json
```

### 3. Deployment

1. Deploy using the deployment script:
```bash
# Make script executable
chmod +x deploy.sh
# Run deployment
./deploy.sh
```

### 4. Post-Deployment

1. Configure custom domain (optional):
```bash
gcloud beta run domain-mappings create \
    --service=ecommerce-backend \
    --domain=api.yourdomain.com \
    --region=us-central1
```

2. Set up SSL certificates (if using custom domain):
```bash
# Cloud Run handles this automatically when using domain-mappings
```

## Infrastructure Components

- **Backend API**: Cloud Run
- **Database**: Cloud SQL (PostgreSQL)
- **File Storage**: Cloud Storage
- **Vector Database**: Qdrant
- **Authentication**: Firebase Auth
- **Secrets**: Secret Manager
- **Monitoring**: Cloud Monitoring
- **Logging**: Cloud Logging

## Environment Variables

The application requires the following environment variables:

```env
# Required environment variables listed in .env.example
```

## Monitoring and Maintenance

1. View logs:
```bash
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=ecommerce-backend"
```

2. Monitor performance:
```bash
# Open Cloud Monitoring dashboard
gcloud monitoring dashboards list
```

3. Database maintenance:
```bash
# Connect to database
gcloud sql connect ecommerce-db --user=postgres
```

## Troubleshooting

1. Check service status:
```bash
gcloud run services describe ecommerce-backend --region=us-central1
```

2. View deployment history:
```bash
gcloud run revisions list --service=ecommerce-backend --region=us-central1
```

3. Common issues:
- **Database connection issues**: Check Cloud SQL proxy and credentials
- **Storage access issues**: Verify IAM permissions
- **Memory issues**: Check Cloud Run instance configuration

## Security Considerations

1. Always use Secret Manager for sensitive data
2. Keep service account keys secure
3. Regularly rotate database passwords
4. Use minimum required IAM permissions
5. Enable audit logging
6. Configure appropriate CORS settings
7. Use Cloud Armor for DDoS protection (optional)

## Cost Optimization

1. Use Cloud SQL autopause for non-production environments
2. Configure appropriate instance sizes
3. Use Cloud Storage lifecycle policies
4. Monitor and optimize Cloud Run instance count
5. Use Cloud Monitoring budget alerts

## Backup and Recovery

1. Database backups:
```bash
# Create on-demand backup
gcloud sql backups create --instance=ecommerce-db
```

2. Storage backups:
```bash
# Set up bucket versioning
gsutil versioning set on gs://$PROJECT_ID-ecommerce-storage
```

## Scaling Considerations

1. Cloud Run auto-scales based on traffic
2. Configure min and max instances appropriately
3. Monitor database connection pooling
4. Use caching where appropriate
5. Configure appropriate resource limits

## Useful Commands

```bash
# View service logs
gcloud logging tail "resource.type=cloud_run_revision"

# Check service health
gcloud run services describe ecommerce-backend

# Update service configuration
gcloud run services update ecommerce-backend --memory=2Gi

# Roll back deployment
gcloud run services rollback ecommerce-backend
``` 