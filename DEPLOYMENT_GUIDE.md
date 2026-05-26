# 🚀 Deployment Guide

## 📦 Preparing for Production

This guide helps you prepare the Research Platform for deployment to production environments or for thesis defense presentation.

---

## 🔧 Pre-Deployment Checklist

### Backend
- [ ] Update `config.py` with production settings
- [ ] Disable debug logging if needed
- [ ] Set appropriate `MAX_OUTPUT_LENGTH`
- [ ] Configure GPU/CPU settings
- [ ] Test all 6 models load correctly

### Frontend
- [ ] Build optimized production bundle
- [ ] Verify all environment variables
- [ ] Test responsive design on mobile
- [ ] Check for console errors
- [ ] Verify API endpoints point to correct server

### Documentation
- [ ] Verify all README files are accurate
- [ ] Check API documentation is complete
- [ ] Create presentation slides if needed
- [ ] Prepare sample documents for demo

---

## 📄 Build Frontend for Production

```bash
cd frontend

# Install dependencies (if not done)
npm install

# Build optimized production bundle
npm run build

# This creates a 'dist/' directory with optimized files
# Size: ~200-300 KB gzipped
```

### Verify Build
```bash
# Preview the production build locally
npm run preview

# Open http://localhost:4173 to test
```

---

## 🐳 Docker Deployment (Optional)

### Dockerfile for Backend

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port
EXPOSE 8000

# Run backend
CMD ["python", "-m", "backend.app.main"]
```

### Docker Compose (Backend + Frontend)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - LOG_LEVEL=info
    volumes:
      - ./models:/app/models
    restart: always

  frontend:
    image: node:18-alpine
    working_dir: /app
    volumes:
      - ./frontend:/app
    ports:
      - "5173:5173"
    command: sh -c "npm install && npm run dev"
    depends_on:
      - backend
    restart: always
```

### Deploy with Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## ☁️ Cloud Deployment Options

### Option 1: Heroku (Free Tier)

```bash
# Create Heroku app
heroku create your-app-name

# Add buildpacks
heroku buildpacks:add heroku/python
heroku buildpacks:add heroku/nodejs

# Deploy
git push heroku main

# Check logs
heroku logs --tail
```

### Option 2: AWS EC2

```bash
# SSH into instance
ssh -i your-key.pem ec2-user@your-instance-ip

# Clone repository
git clone <your-repo-url>
cd NLP-Text-Summarization-Transformer-System

# Install dependencies
pip install -r requirements.txt

# Run backend with systemd (permanent)
sudo nano /etc/systemd/system/nlp-backend.service
```

Add to service file:
```ini
[Unit]
Description=NLP Summarization Backend
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/app
ExecStart=/usr/bin/python3 -m backend.app.main
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl start nlp-backend
sudo systemctl enable nlp-backend
```

### Option 3: Google Cloud Run

```bash
# Setup gcloud CLI
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create app.yaml
cat > app.yaml << EOF
runtime: python310
env: standard
entrypoint: python -m backend.app.main

env_variables:
  API_HOST: "0.0.0.0"
  API_PORT: "8080"
EOF

# Deploy
gcloud app deploy

# View logs
gcloud app logs read
```

---

## 🌐 Nginx Configuration (Reverse Proxy)

Create `/etc/nginx/sites-available/nlp-platform`:

```nginx
upstream backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL certificates (use Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Frontend (React build)
    location / {
        alias /var/www/nlp-platform/dist/;
        try_files $uri $uri/ /index.html;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # API proxy
    location /api/ {
        proxy_pass http://backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # API docs
    location /docs {
        proxy_pass http://backend/docs;
        proxy_set_header Host $host;
    }
}
```

Enable and test:
```bash
sudo ln -s /etc/nginx/sites-available/nlp-platform /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 🔒 Security Checklist

- [ ] Enable HTTPS (Let's Encrypt)
- [ ] Set secure CORS headers
- [ ] Rate limit API endpoints
- [ ] Implement authentication if needed
- [ ] Hide sensitive environment variables
- [ ] Regular security updates
- [ ] Monitor logs for suspicious activity
- [ ] Backup model files regularly

### Environment Variables Setup

Create `.env` file:
```bash
# Backend
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
ENVIRONMENT=production

# Model settings
USE_GPU=true
FP16_INFERENCE=true
MAX_OUTPUT_LENGTH=150

# Security
CORS_ORIGINS=["https://your-domain.com"]
```

---

## 📊 Performance Optimization

### Backend Optimization

```python
# In config.py
# Increase model cache
PRELOAD_MODELS = True
GPU_MEMORY_FRACTION = 0.8

# Optimize tokenization
BATCH_SIZE = 32
NUM_WORKERS = 4

# Enable caching
CACHE_RESULTS = True
CACHE_TTL = 3600  # 1 hour
```

### Frontend Optimization

```javascript
// vite.config.js
export default {
  build: {
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true
      }
    },
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks: {
          recharts: ['recharts']
        }
      }
    }
  }
}
```

### Caching Strategy

- Frontend: Cache assets for 30 days
- API responses: Cache for 1 hour
- Model weights: Cache in VRAM

---

## 📈 Monitoring & Analytics

### Backend Health Check

```bash
# Setup health check endpoint
curl http://your-domain.com/health

# Expected response:
{
  "status": "healthy",
  "timestamp": "2024-05-26T10:30:00Z",
  "gpu_available": true,
  "models_loaded": 6,
  "vram_usage": "3.5GB"
}
```

### Error Tracking

```python
# Setup logging
import logging
logger = logging.getLogger(__name__)

# Log important events
logger.info(f"Comparison started for {len(text)} chars")
logger.error(f"Model load failed: {error}")
```

### Performance Metrics

Monitor these metrics:
- API response time (target: <10s for comparison)
- Model inference time (extractive: <100ms, abstractive: 6-8s)
- GPU memory usage (target: <8GB)
- Server uptime (target: 99.9%)
- Error rate (target: <0.1%)

---

## 📱 Mobile & Responsive Testing

### Test on Different Devices
```bash
# Chrome DevTools
1. Open DevTools (F12)
2. Click device toggle (Ctrl+Shift+M)
3. Test on iPhone 12, iPad, Android tablets
4. Check touch interactions
```

### Viewport Sizes to Test
- Mobile: 375x667 (iPhone 12)
- Tablet: 768x1024 (iPad)
- Desktop: 1920x1080
- Large Desktop: 2560x1440

---

## 🚨 Troubleshooting Deployment

### Backend Issues

**Port already in use**
```bash
# Find and kill process
lsof -i :8000
kill -9 <PID>
```

**GPU not available**
```python
# Fallback to CPU automatically
import torch
if not torch.cuda.is_available():
    print("Using CPU mode")
```

**Model download fails**
```bash
# Manually download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('vinai/vit5-base')"
```

### Frontend Issues

**Build fails**
```bash
# Clear cache and rebuild
rm -rf node_modules dist
npm install
npm run build
```

**API not accessible**
```javascript
// Check CORS headers
// Verify API URL in .env
// Check firewall rules
```

---

## 📚 Presentation Setup for Defense

### Setup Guide for Demo

```
1. Start Backend
   Terminal 1: python -m backend.app.main
   
2. Start Frontend
   Terminal 2: cd frontend && npm run dev
   
3. Open in Browser
   http://localhost:5173/comparison
   
4. Full-screen mode
   F11 or Fn+F11
   
5. Demo flow:
   - Input sample text
   - Click "▶ Chạy So Sánh"
   - Show comparison tab
   - Switch to charts
   - Show explanations
   - Export report
```

### Demo Script
```
"Let me demonstrate the Vietnamese NLP Text Summarization 
Research Platform. This system compares extractive versus 
abstractive summarization methods.

First, I'll input a Vietnamese document...
Then run the comparison which processes it through 6 algorithms...
Here we can see the extractive methods on the left extract 
important sentences from the original text...
And the abstractive methods on the right use AI to generate 
paraphrased summaries...
The metrics show extractive is 100x faster but abstractive 
achieves 15-20% higher quality..."
```

---

## ✅ Final Verification

Before submitting or presenting:

```bash
# 1. Backend running
curl http://localhost:8000/health

# 2. Frontend accessible
curl http://localhost:5173

# 3. Comparison endpoint works
curl -X POST http://localhost:8000/research/compare/detailed \
  -H "Content-Type: application/json" \
  -d '{"text":"Sample Vietnamese text..."}'

# 4. Charts load
# Open browser and verify all 5 chart types

# 5. Export works
# Generate HTML and Markdown reports

# 6. Performance acceptable
# Run comparison 5 times and check average time
```

---

## 🎓 Ready for Defense!

Your platform is now deployment-ready. Good luck with your thesis defense! 🎉

---

**Version**: 1.0.0 | **Last Updated**: May 26, 2024 | **Status**: Production Ready
