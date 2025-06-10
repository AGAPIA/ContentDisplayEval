# Deployment Architecture

The inference service is composed of:

- **gRPC Server**: Receives image frames, runs models, streams back predictions.
- **Shared Storage**: Holds model checkpoints.
- **Kubernetes** or **Docker** for container orchestration.
