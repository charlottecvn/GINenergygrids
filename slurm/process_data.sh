#!/bin/bash
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=4:59:00
#SBATCH --output=logs/out/slurm_processdata-%j.out
#SBATCH --error=logs/err/slurm_processdata-%j.err

source ../virtual_environment/bin/activate
python -u gnn-nmin1-alliander/process_data.py
