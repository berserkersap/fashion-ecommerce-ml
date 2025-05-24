# CI/CD Deployment Guide

> **Note**: This guide focuses on setting up automated deployment with GitHub Actions. For manual deployment, see [Quick Start Guide](../DEPLOYMENT.md). For detailed configuration, see [Comprehensive Guide](../DEPLOYMENT_GUIDE.md).

This guide explains how to set up and configure the CI/CD pipeline for the Fashion E-commerce application.

## Prerequisites

1. A Google Cloud Platform (GCP) account
2. A GitHub account
3. A Snyk account (for security scanning)
4. Firebase project
5. Stripe account

## Setting Up Google Cloud Platform

1. Create a new GCP project or use an existing one:
   ```bash
   gcloud projects create [PROJECT_ID] --name="Fashion E-commerce"
   gcloud config set project [PROJECT_ID]
   ```

2. Enable required APIs:
   ```bash
   gcloud services enable \
     run.googleapis.com \
     containerregistry.googleapis.com \
     secretmanager.googleapis.com \
     cloudbuild.googleapis.com
   ```

3. Create a service account for GitHub Actions:
   ```bash
   gcloud iam service-accounts create github-actions \
     --description="Service account for GitHub Actions" \
     --display-name="GitHub Actions"
   ```

4. Grant necessary permissions:
   ```bash
   gcloud projects add-iam-policy-binding [PROJECT_ID] \
     --member="serviceAccount:github-actions@[PROJECT_ID].iam.gserviceaccount.com" \
     --role="roles/run.admin"

   gcloud projects add-iam-policy-binding [PROJECT_ID] \
     --member="serviceAccount:github-actions@[PROJECT_ID].iam.gserviceaccount.com" \
     --role="roles/storage.admin"

   gcloud projects add-iam-policy-binding [PROJECT_ID] \
     --member="serviceAccount:github-actions@[PROJECT_ID].iam.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"
   ```

5. Create and download service account key:
   ```bash
   gcloud iam service-accounts keys create key.json \
     --iam-account=github-actions@[PROJECT_ID].iam.gserviceaccount.com
   ```

## Setting Up Secrets

### GitHub Secrets

1. Go to your GitHub repository → Settings → Secrets and variables → Actions
2. Add the following secrets:
   - `GCP_PROJECT_ID`: Your Google Cloud project ID
   - `GCP_SA_KEY`: The entire content of the `key.json` file created earlier
   - `SNYK_TOKEN`: Your Snyk API token (get from Snyk account settings)

### Google Cloud Secrets

1. Store Firebase credentials:
   ```bash
   gcloud secrets create FIREBASE_CREDENTIALS \
     --data-file="path/to/firebase-credentials.json"
   ```

2. Store Stripe secret key:
   ```bash
   echo -n "your-stripe-secret-key" | \
   gcloud secrets create STRIPE_SECRET_KEY --data-file=-
   ```

## Environment Variables

The application uses the following environment variables in production:

```env
ENVIRONMENT=production
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_STORAGE_BUCKET=your-bucket-name
GOOGLE_CLOUD_PROJECT=your-project-id
FRONTEND_URL=https://your-frontend-url
```

These are set in the Cloud Run deployment command in the CI/CD pipeline.

## Deployment Process

The CI/CD pipeline automatically:

1. Runs on every push to the main branch
2. Executes tests, linting, and security checks
3. Builds and pushes Docker image to Google Container Registry
4. Deploys to Google Cloud Run

### Manual Deployment

If needed, you can deploy manually:

```bash
# Build Docker image
docker build -t gcr.io/[PROJECT_ID]/fashion-ecommerce:latest .

# Push to Container Registry
docker push gcr.io/[PROJECT_ID]/fashion-ecommerce:latest

# Deploy to Cloud Run
gcloud run deploy fashion-ecommerce \
  --image gcr.io/[PROJECT_ID]/fashion-ecommerce:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="ENVIRONMENT=production" \
  --set-secrets="FIREBASE_CREDENTIALS=FIREBASE_CREDENTIALS:latest" \
  --set-secrets="STRIPE_SECRET_KEY=STRIPE_SECRET_KEY:latest"
```

## Monitoring and Logging

1. View Cloud Run service logs:
   ```bash
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=fashion-ecommerce"
   ```

2. Monitor service metrics in Google Cloud Console:
   - Go to Cloud Run → fashion-ecommerce → Metrics
   - View request counts, latency, and memory usage

## Troubleshooting

1. If deployment fails:
   - Check Cloud Build logs in Google Cloud Console
   - Verify all secrets are properly set
   - Ensure service account has necessary permissions

2. If application fails to start:
   - Check Cloud Run logs
   - Verify environment variables and secrets
   - Check Firebase credentials are valid

3. Common issues:
   - Missing or invalid secrets
   - Insufficient permissions
   - Resource limits exceeded
   - Network connectivity issues

## Security Best Practices

1. Regularly rotate service account keys
2. Monitor Snyk security reports
3. Keep dependencies updated
4. Review Cloud Audit Logs
5. Use least privilege principle for service accounts
6. Enable Cloud Run security features:
   - Container scanning
   - Binary Authorization
   - VPC Service Controls (if needed)

## Support and Maintenance

1. Set up monitoring alerts:
   ```bash
   gcloud monitoring channels create \
     --display-name="DevOps Team Email" \
     --type=email \
     --email-address=devops@your-company.com
   ```

2. Create uptime checks:
   ```bash
   gcloud monitoring uptime-checks create http \
     --display-name="Fashion E-commerce API" \
     --uri="https://your-service-url/health"
   ``` 