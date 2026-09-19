# .github/workflows/

This directory contains CI/CD automation:

- Run unit tests for Python pipeline code
- Build and publish deployment container image
- Update Kubernetes manifest image tag to trigger ArgoCD sync

The workflow expects repository secrets for registry authentication when publishing images.
