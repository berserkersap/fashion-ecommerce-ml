# Use Python 3.11 slim image as base (consistent with root Dockerfile)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (include all necessary dependencies)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080  \
    BACKEND_PORT=8000 
    # Define a separate port for the backend

# Install Python dependencies (combine all requirements)
COPY requirements.txt ./
COPY frontend/requirements.txt ./frontend/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Copy application code
COPY . .

# Prepare Cloud Run-specific environment
# COPY frontend/.env.cloud frontend/.env
# COPY app/.env.cloud app/.env

# Create directory for secrets
RUN mkdir -p /secrets

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser
RUN chown -R appuser:appuser /app /secrets
USER appuser

# Expose one port (only frontend and not backend)
EXPOSE 8000
# EXPOSE 8080

# Start both applications using a process manager (e.g., supervisord)
RUN pip install supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/home/appuser/.local/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
