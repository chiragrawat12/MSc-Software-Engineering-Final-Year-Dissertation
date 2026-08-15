#!/bin/bash

#SBATCH --job-name=regenerate_predictions
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=30:00:00
#SBATCH --output=/home/chirag.singh/project/logs/regenerate_predictions/regenerate_predictions_%j.out
#SBATCH --error=/home/chirag.singh/project/logs/regenerate_predictions/regenerate_predictions_%j.err

# Activate your environment
source ~/envs/myenv/bin/activate

# Move to your project directory
cd ~/project

# Run your script
python scripts/regenerate_predictions.py
