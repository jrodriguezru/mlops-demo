# hpc/

Optional training path for environments with HPC resources.

- `train.slurm`: Slurm submission script for scheduled training
- `train_hpc.py`: entrypoint wrapper that can be extended with distributed/HPC-specific logic

This directory is designed to keep HPC-specific concerns separate from default local/cluster training.
