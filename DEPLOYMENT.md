# Quick Start Deployment Guide

> **Note**: This is the quick start guide for manual deployment. For CI/CD setup, see [CI/CD Guide](docs/deployment.md). For detailed configuration, see [Comprehensive Guide](DEPLOYMENT_GUIDE.md).

## Pre-deployment Checklist

### 1. Local Setup
- [ ] Install Google Cloud SDK
- [ ] Install Firebase CLI
- [ ] Clone repository
- [ ] Create `.env` files for both frontend and backend

### 2. Google Cloud Setup
- [ ] Create new Google Cloud Project (or use existing)
- [ ] Enable billing
- [ ] Enable required APIs:
  ```bash
  gcloud services enable \
    run.googleapis.com \
    secretmanager.googleapis.com \
    cloudbuild.googleapis.com \
    cloudresourcemanager.googleapis.com \
    sqladmin.googleapis.com
  ```

### 3. Secret Manager Setup (REQUIRED BEFORE CLOUD RUN)
- [ ] Enable Secret Manager API
- [ ] Create required secrets
- [ ] Store sensitive values in secrets
- [ ] Grant access to Cloud Run service account

### 4. Firebase Setup
- [ ] Create Firebase project
- [ ] Download Firebase credentials (`firebase-credentials.json`)
- [ ] Enable Authentication in Firebase Console
- [ ] Configure Authentication providers (Email/Password, Google, etc.)

### 5. Database Setup
- [ ] Create Cloud SQL instance
- [ ] Create database and user
- [ ] Note down connection details
- [ ] Run database migrations

## Environment Files

### Backend (.env)
```env
# Required Environment Variables
GOOGLE_CLOUD_PROJECT=your-project-id
DB_HOST=your-cloud-sql-instance
DB_USER=your-db-user
DB_PASSWORD=your-db-password
DB_NAME=fashion_store
FIREBASE_CREDENTIALS_PATH=/secrets/firebase-credentials.json

# Optional Environment Variables
ENVIRONMENT=production
RATE_LIMIT=100
RATE_LIMIT_WINDOW=3600
```

### Frontend (.env)
```env
BACKEND_URL=https://your-backend-url
NODE_ENV=production
PORT=8080
```

## Secret Manager Configuration

This must be completed BEFORE deploying to Cloud Run:

1. **Enable Secret Manager API**:
   ```bash
   gcloud services enable secretmanager.googleapis.com
   ```

2. **Create Required Secrets**:
   ```bash
   # Create secrets
   gcloud secrets create db-user --replication-policy="automatic"
   gcloud secrets create db-password --replication-policy="automatic"
   gcloud secrets create firebase-credentials --replication-policy="automatic"
   gcloud secrets create stripe-secret --replication-policy="automatic"
   gcloud secrets create stripe-webhook-secret --replication-policy="automatic"

   # Store values
   echo -n "your-db-username" | gcloud secrets versions add db-user --data-file=-
   echo -n "your-db-password" | gcloud secrets versions add db-password --data-file=-
   gcloud secrets versions add firebase-credentials --data-file=./firebase-credentials.json
   echo -n "your-stripe-secret-key" | gcloud secrets versions add stripe-secret --data-file=-
   echo -n "your-stripe-webhook-secret" | gcloud secrets versions add stripe-webhook-secret --data-file=-
   ```

3. **Grant Access to Cloud Run**:
   ```bash
   # Get project number
   PROJECT_NUMBER=$(gcloud projects describe $GOOGLE_CLOUD_PROJECT --format='value(projectNumber)')

   # Grant access to each secret
   for SECRET in db-user db-password firebase-credentials stripe-secret stripe-webhook-secret; do
       gcloud secrets add-iam-policy-binding $SECRET \
           --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
           --role="roles/secretmanager.secretAccessor"
   done
   ```

## Deployment Steps

1. **Initial Setup**
```bash
# Login to Google Cloud
gcloud auth login

# Set project
gcloud config set project your-project-id

# Authenticate Docker with Google Cloud
gcloud auth configure-docker
```

2. **Deploy Application**
```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment (this will use secrets configured above)
./deploy.sh
```

3. **Verify Deployment**
```bash
# Check services
gcloud run services list

# View logs
gcloud run services logs read fashion-store-backend
gcloud run services logs read fashion-store-frontend
```

## Post-deployment Tasks

1. **Update DNS (if using custom domain)**
```bash
# Map custom domain
gcloud beta run domain-mappings create \
  --service fashion-store-frontend \
  --domain your-domain.com
```

2. **Set up monitoring**
```bash
# Create uptime check
gcloud monitoring uptime-check-configs create fashion-store-backend \
  --display-name="Fashion Store Backend" \
  --http-check=host=$(gcloud run services describe fashion-store-backend \
  --format='value(status.url)' --region=us-central1),path=/health
```

3. **Configure SSL (if using custom domain)**
- [ ] Verify domain ownership
- [ ] Wait for SSL certificate provisioning
- [ ] Test HTTPS access

## Common Issues & Solutions

### CORS Issues
1. Check FRONTEND_URL in backend service matches actual frontend URL
2. Verify CORS configuration in `main.py`
3. Test with curl:
```bash
curl -I -H "Origin: your-frontend-url" your-backend-url/health
```

### Database Connection Issues
1. Check Cloud SQL instance is running
2. Verify connection string format
3. Test connection:
```bash
gcloud sql connect your-instance-name --user=your-user
```

### Firebase Authentication Issues
1. Verify Firebase credentials are mounted correctly
2. Check Firebase project settings
3. Test authentication flow:
```bash
curl -X POST your-backend-url/auth/test
```

## Rollback Procedure

If deployment fails or issues are found:

1. **Rollback to Previous Version**
```bash
# List revisions
gcloud run revisions list --service fashion-store-backend

# Rollback to previous revision
gcloud run services update-traffic fashion-store-backend \
  --to-revision=previous-revision-name
```

2. **Emergency Rollback**
```bash
# Stop traffic to new version
gcloud run services update-traffic fashion-store-backend \
  --to-revisions=previous-revision-name=100
```

## Useful Commands

### View Service Status
```bash
gcloud run services describe fashion-store-backend
gcloud run services describe fashion-store-frontend
```

### View Logs
```bash
# View backend logs
gcloud run services logs read fashion-store-backend

# View frontend logs
gcloud run services logs read fashion-store-frontend
```

### Update Environment Variables
```bash
# Update backend env vars
gcloud run services update fashion-store-backend \
  --set-env-vars KEY=VALUE

# Update frontend env vars
gcloud run services update fashion-store-frontend \
  --set-env-vars KEY=VALUE
```

### Manage Secrets
```bash
# Update a secret
echo -n "new-value" | gcloud secrets versions add secret-name --data-file=-

# List secrets
gcloud secrets list

# View secret versions
gcloud secrets versions list secret-name
``` 