# k8s/

This directory stores Kubernetes/GitOps manifests for cluster components:

- `argocd/`: ArgoCD `Application` resources
- `mlflow/`: self-hosted MLflow deployment/service
- `inference/`: FastAPI model serving deployment/service/HPA

ArgoCD should track this repository path and reconcile workloads in-cluster.
