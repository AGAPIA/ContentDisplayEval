# Deployment Guide

1. **Build Docker Image**:
   ```bash
   docker build -t inference-server:latest -f docker/Dockerfile .
   ```
2. **Run Locally**:
   ```bash
   docker run -p 50051:50051 inference-server:latest
   ```
3. **Kubernetes**:
   ```bash
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl apply -f k8s/ingress.yaml
   ```
4. **Health Check**:
   ```bash
   ./scripts/health_check.sh
   ```
