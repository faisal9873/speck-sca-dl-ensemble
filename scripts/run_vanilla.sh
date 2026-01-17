#!/bin/bash
#
# SLURM job script for running deep learning ensemble–based
# profiling side-channel analysis experiments on the SPECK cipher.
#
# Notes:
# - Submit this job from the repository root directory.
# - This script is intended for GPU-enabled HPC environments.
# - Users may need to modify SLURM directives and Conda paths
#   to match their local cluster configuration.
#

#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=30
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --constraint=jubail

# ==============================
# Environment setup
# ==============================

# Load Conda (cluster-specific)
source /share/apps/BYUAD5/miniconda/3-4.11.0/bin/activate

conda activate tf-ensemble

# ==============================
# Run experiment
# ==============================

# Ensure execution from repository root
PROJECT_ROOT=$(pwd)

python ${PROJECT_ROOT}/src/ensemble_vanilla.py

