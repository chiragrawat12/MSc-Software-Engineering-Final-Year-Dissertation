#!/bin/bash

#SBATCH --job-name=yolov8_train
#SBATCH --partition=student
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=/home/chirag.singh/project/logs/yolo/yolov8_%j.out
#SBATCH --error=/home/chirag.singh/project/logs/yolo/yolov8_%j.err

# Activate your environment
source ~/envs/myenv/bin/activate

# Move to your project directory
cd ~/project

# Run your script
python scripts/YOLOv8_Food101.py
