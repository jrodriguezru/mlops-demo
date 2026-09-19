# MLOps Demo (Self-Hosted)

This repository is a compact, infrastructure-first example of an end-to-end ML + MLOps project that can run on self-hosted/on-device environments.

The focus is on:

- reproducible model preparation/training flow
- experiment tracking with self-hosted MLflow
- API deployment with FastAPI + Docker
- Kubernetes/ArgoCD GitOps delivery patterns
- CI/CD automation with GitHub Actions
- optional HPC (Slurm) training path

> The goal is to showcase platform and delivery architecture, not model sophistication.

## Repository structure

```text
.
├── model/                 # data prep, training, MLflow logging helpers
├── deployment/            # FastAPI inference service + Docker assets
├── k8s/                   # Kubernetes/ArgoCD/MLflow manifests
├── .github/workflows/     # CI/CD pipelines and automated checks
└── hpc/                   # optional Slurm/HPC training scripts
```

Each main directory includes its own `README.md` with local details.

## Typical flow

1. Develop and test model pipeline logic in `model/`.
2. Track training runs and metadata in MLflow (self-hosted service in the cluster).
3. Build and publish the inference image from `deployment/`.
4. Update manifests in `k8s/` to trigger ArgoCD reconciliation.
5. Optionally launch larger training jobs through `hpc/` on Slurm resources.

## Quick start (local)

```bash
python -m unittest discover -s tests
```