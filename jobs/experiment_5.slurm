#!/bin/bash

#SBATCH --job-name=experiment_5
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=/home/chirag.singh/project/logs/experiment_5/experiment_5_%j.out
#SBATCH --error=/home/chirag.singh/project/logs/experiment_5/experiment_5_%j.err

# Activate your environment
source ~/envs/myenv/bin/activate

# Move to your project directory
cd ~/project

# Run your script
python scripts/Experiment_5.py
#python scripts/Experiment_5.py --models clip