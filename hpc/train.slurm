#!/bin/bash
#SBATCH --job-name=mlops-demo-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

module load python || true

python hpc/train_hpc.py
