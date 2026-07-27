#!/bin/bash

#SBATCH --job-name=alexnet_train
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/chirag.singh/project/logs/alexnet/alexnet_%j.out
#SBATCH --error=/home/chirag.singh/project/logs/alexnet/alexnet_%j.err

# Activate your environment
source ~/envs/myenv/bin/activate

# Move to your project directory
cd ~/project

# Run your script
python scripts/AlexNet_Food101.py
