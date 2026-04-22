#!/bin/bash
#SBATCH --job-name=arag_plots
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=arag_plots_%j.out
#SBATCH --error=arag_plots_%j.err

module load anaconda/3
eval "$(conda shell.bash hook)"
conda activate adaptiverag

pip install matplotlib --quiet 2>/dev/null
python ~/adaptive-rag/analysis/generate_plots.py
