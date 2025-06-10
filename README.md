# Automated Visual Anomaly Detection Framework

This repository provides a complete pipeline for generating, training, and deploying automated visual anomaly detection in real-time 3D applications (e.g., video games). It covers both **texture** and **mesh** anomaly detection through two-stage deep learning architectures, as well as Unity-based data generation tools and deployment artifacts.

It is a continuations of the two papers:
- EASE2024: Automated evaluation of game content display using deep learning, https://dl.acm.org/doi/10.1145/3661167.3661184 
- EASE2025: https://conf.researchr.org/details/ease-2025/ease-2025-industry-papers/6/Hierarchical-deep-learning-framework-for-continuous-state-aware-visual-glitch-detect (to be presented)
## Project Overview

- **DataGenerator/**  
  Unity C# scripts to introduce controlled visual anomalies and capture screenshots + metadata.  
  - [Dataset Generator Documentation](DataGenerator/unity_dataset_generator_doc.md)

- **Pipelines/**  
  Python implementations of the two-stage detection pipelines and orchestration scripts:  
  - **Texture Pipeline**  
    - [Stage 1: Feature Extraction (ShuffleNetV2)](Pipelines/texture_stage1/)  
    - [Stage 2: Refinement (DenseNet121)](Pipelines/texture_stage2/)  
    - [Texture Pipeline User Guide](Pipelines/texture_pipeline_user_guide.md)  
    - [Run Full Texture Inference](Pipelines/run_full_texture_inference.py)  
  - **Mesh Pipeline**  
    - [Stage 1: Segmentation (U-Net)](Pipelines/mesh_stage1/)  
    - [Stage 2: Refinement (DeepLabV3+)](Pipelines/mesh_stage2/)  
    - [Mesh Pipeline User Guide](Pipelines/mesh_pipeline_user_guide.md)  
    - [Run Full Mesh Pipeline](Pipelines/run_full_mesh_pipeline.py)

- **Deployments/**  
  Artifacts for containerization and service deployment:  
  - [Deployments README](Deployments/README.md)  
  - Docker, Kubernetes, gRPC server code, scripts, configs, and docs in subfolders.

## Getting Started

1. **Data Generation**  
   - Open the Unity project in `DataGenerator/`.  
   - Configure and run the dataset generator.  
   - See [Dataset Generator Documentation](DataGenerator/unity_dataset_generator_doc.md).

2. **Training & Inference**  
   - Install Python dependencies:  
     ```bash
     pip install torch torchvision tqdm pyyaml
     ```  
   - Follow each pipeline’s user guide:  
     - [Texture Pipeline User Guide](Pipelines/texture_pipeline_user_guide.md)  
     - [Mesh Pipeline User Guide](Pipelines/mesh_pipeline_user_guide.md)

3. **Deployment**  
   - Follow the instructions in [Deployments README](Deployments/README.md) to build and run the service.

---

For detailed information on each component, click the links above.
