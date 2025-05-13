#!/bin/bash
#SBATCH --job-name=bert_classifier
#SBATCH --output=logs/bert_classifier_%j.out
#SBATCH --error=logs/bert_classifier_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asbarman1@sheffield.ac.uk

# Load necessary modules
module purge
module load Python/3.10.8-GCCcore-12.2.0

# Activate your environment
source ~/project/dissenv/bin/activate