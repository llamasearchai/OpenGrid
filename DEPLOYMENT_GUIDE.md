# OpenGrid Deployment Guide

This guide provides comprehensive instructions for deploying the OpenGrid platform in various environments.

## Quick Deployment Options

### 1. Local Development (Fastest)

```bash
# Clone and setup
git clone https://github.com/llamasearchai/OpenGrid.git
cd OpenGrid
pip install -r requirements.txt

# Set environment (optional)
export OPENAI_API_KEY="your-api-key-here"

# Start server
python main.py
```

**Access:** http://localhost:8000  
**Documentation:** http://localhost:8000/docs

### 2. Docker (Recommended)

```bash
# Build and run
docker build -t opengrid .
docker run -p 8000:8000 -e OPENAI_API_KEY="your-key" opengrid
```

### 3. Docker Compose (Full Stack)

```bash
# Production deployment
docker-compose up -d

# Development with Redis
docker-compose --profile dev up -d

# Full stack with database
docker-compose --profile full up -d
```

## Detailed Deployment Scenarios

### Local Development Environment

#### Prerequisites
- Python 3.11 or higher
- Git
- 4GB+ RAM
- 2GB free disk space

#### Step-by-Step Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/llamasearchai/OpenGrid.git
   cd OpenGrid
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your-key-here" > .env
   echo "LOG_LEVEL=INFO" >> .env
   echo "RELOAD=true" >> .env
   ```

5. **Verify Installation**
   ```bash
   # Test CLI
   python main.py network list-samples
   
   # Start server
   python main.py
   ```

### Docker Development Environment

#### Using Pre-built Development Image

```bash
# Development with hot reload
docker run -it --rm \
  -p 8000:8000 \
  -v $(pwd):/app \
  -e OPENAI_API_KEY="your-key" \
  -e RELOAD=true \
  opengrid:dev
```

#### Custom Development Build

```dockerfile
# Dockerfile.dev (already included)
FROM python:3.11-slim
# ... (see existing Dockerfile.dev)
```

```bash
# Build development image
docker build -f Dockerfile.dev -t opengrid:dev .

# Run with volume mounting for live editing
docker run -it --rm \
  -p 8000:8000 \
  -p 5678:5678 \  # Debug port
  -v $(pwd):/app \
  -e OPENAI_API_KEY="your-key" \
  opengrid:dev
```

### Production Deployment

#### Docker Production

```bash
# Build production image
docker build -t opengrid:latest .

# Run production container
docker run -d \
  --name opengrid-prod \
  --restart unless-stopped \
  -p 80:8000 \
  -e OPENAI_API_KEY="your-key" \
  -e WORKERS=4 \
  -e HOST=0.0.0.0 \
  -e PORT=8000 \
  -e RELOAD=false \
  opengrid:latest
```

#### Docker Compose Production Stack

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  opengrid:
    image: opengrid:latest
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - WORKERS=4
      - HOST=0.0.0.0
      - PORT=8000
      - RELOAD=false
    volumes:
      - opengrid_data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - opengrid

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data

volumes:
  opengrid_data:
  redis_data:
```

### Cloud Platform Deployments

#### AWS ECS/Fargate

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com

docker build -t opengrid .
docker tag opengrid:latest your-account.dkr.ecr.us-east-1.amazonaws.com/opengrid:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/opengrid:latest
```

**ECS Task Definition:**
```json
{
  "family": "opengrid-task",
  "cpu": "512",
  "memory": "1024",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "containerDefinitions": [
    {
      "name": "opengrid",
      "image": "your-account.dkr.ecr.us-east-1.amazonaws.com/opengrid:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAI_API_KEY",
          "value": "your-key"
        },
        {
          "name": "WORKERS",
          "value": "2"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/opengrid",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Run

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/your-project/opengrid

# Deploy to Cloud Run
gcloud run deploy opengrid \
  --image gcr.io/your-project/opengrid \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY="your-key",WORKERS=2 \
  --memory 2Gi \
  --cpu 2
```

#### Azure Container Instances

```bash
# Create resource group
az group create --name opengrid-rg --location eastus

# Deploy container
az container create \
  --resource-group opengrid-rg \
  --name opengrid \
  --image opengrid:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --dns-name-label opengrid-demo \
  --environment-variables OPENAI_API_KEY="your-key" WORKERS=2
```

#### Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opengrid
spec:
  replicas: 3
  selector:
    matchLabels:
      app: opengrid
  template:
    metadata:
      labels:
        app: opengrid
    spec:
      containers:
      - name: opengrid
        image: opengrid:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: opengrid-secrets
              key: openai-api-key
        - name: WORKERS
          value: "2"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: opengrid-service
spec:
  selector:
    app: opengrid
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: Secret
metadata:
  name: opengrid-secrets
type: Opaque
data:
  openai-api-key: eW91ci1iYXNlNjQtZW5jb2RlZC1rZXk=  # base64 encoded
```

```bash
# Deploy to Kubernetes
kubectl apply -f k8s-deployment.yaml
```

## Environment Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key for AI features | None | No |
| `HOST` | Server bind address | 127.0.0.1 | No |
| `PORT` | Server port | 8000 | No |
| `WORKERS` | Number of worker processes | 1 | No |
| `RELOAD` | Enable auto-reload (dev only) | false | No |
| `LOG_LEVEL` | Logging level | INFO | No |
| `DATABASE_URL` | Database connection string | None | No |
| `REDIS_URL` | Redis connection string | None | No |

### Configuration Files

#### `.env` File
```bash
# Core settings
OPENAI_API_KEY=sk-your-key-here
HOST=0.0.0.0
PORT=8000
WORKERS=4
RELOAD=false

# Logging
LOG_LEVEL=INFO
STRUCTLOG_PROCESSOR=json

# Database (optional)
DATABASE_URL=sqlite:///opengrid.db
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com
```

#### `config.yml` (Future Enhancement)
```yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  reload: false

ai:
  provider: openai
  model: gpt-4
  max_tokens: 2048
  temperature: 0.1

analysis:
  default_tolerance: 1e-6
  max_iterations: 100
  parallel_processing: true

security:
  cors_enabled: true
  rate_limiting: true
  api_key_required: false
```

## Security Configuration

### Production Security Checklist

- [ ] Set secure `SECRET_KEY`
- [ ] Enable HTTPS/TLS
- [ ] Configure CORS properly
- [ ] Enable rate limiting
- [ ] Set up API authentication
- [ ] Use environment variables for secrets
- [ ] Enable security headers
- [ ] Set up monitoring/logging
- [ ] Configure firewall rules
- [ ] Regular security updates

### HTTPS Configuration

#### Nginx SSL Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/ssl/certs/your-domain.pem;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://opengrid:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### API Security

#### Rate Limiting
```python
# In production, add rate limiting middleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/networks")
@limiter.limit("10/minute")
async def list_networks(request: Request):
    # API endpoint with rate limiting
    pass
```

## Monitoring and Logging

### Health Checks

The application includes built-in health check endpoints:

- `/health` - Basic health status
- `/health/detailed` - Detailed system status
- `/metrics` - Application metrics (Prometheus format)

### Logging Configuration

```bash
# Structured logging with JSON output
export STRUCTLOG_PROCESSOR=json
export LOG_LEVEL=INFO

# Log to file
export LOG_FILE=/app/logs/opengrid.log
```

### Monitoring Stack

#### Prometheus + Grafana
```yaml
# monitoring/docker-compose.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Check what's using port 8000
lsof -i :8000

# Kill process using port
kill -9 <PID>

# Or use different port
export PORT=8001
python main.py
```

#### Memory Issues
```bash
# Increase memory limits for Docker
docker run -m 4g opengrid:latest

# For Kubernetes
resources:
  limits:
    memory: "4Gi"
```

#### Permission Errors
```bash
# Fix file permissions
chmod +x main.py

# For Docker volume permissions
docker run --user $(id -u):$(id -g) opengrid:latest
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Enable development mode
export RELOAD=true
export DEBUG=true

# Run with debugger
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py
```

### Performance Tuning

#### Production Optimization
```bash
# Increase worker processes
export WORKERS=4

# Optimize Python
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1

# Increase ulimits
ulimit -n 65536
```

## Scaling Considerations

### Horizontal Scaling

- Use load balancer (nginx, HAProxy, AWS ALB)
- Scale worker processes: `WORKERS=CPU_CORES`
- Use Redis for session storage
- Consider async processing for heavy computations

### Vertical Scaling

- Minimum: 2 CPU cores, 4GB RAM
- Recommended: 4 CPU cores, 8GB RAM
- Heavy workloads: 8+ CPU cores, 16GB+ RAM

### Database Scaling

- PostgreSQL for persistent storage
- Redis for caching and sessions
- Consider read replicas for heavy read workloads

## Updates and Maintenance

### Updating OpenGrid

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart services
docker-compose restart
```

### Backup Procedures

```bash
# Backup data volumes
docker run --rm -v opengrid_data:/data -v $(pwd):/backup ubuntu tar czf /backup/opengrid-backup.tar.gz /data

# Restore from backup
docker run --rm -v opengrid_data:/data -v $(pwd):/backup ubuntu tar xzf /backup/opengrid-backup.tar.gz -C /
```

## Support

- Email: nikjois@llamasearch.ai
- Issues: [GitHub Issues](https://github.com/llamasearchai/OpenGrid/issues)
- Documentation: Available at `/docs` when server is running

---

**Ready to deploy? Choose your preferred method above and start analyzing power systems with OpenGrid!** 