#!/bin/bash
# Job name:
#SBATCH --job-name=otflow
#
# Account:
#SBATCH --account=fc_ntugame
#
# Partition:
#SBATCH --partition=savio2_gpu
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of “gpu:[1-4]“, or “gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit (8hrs):
#SBATCH --time=06:00:00
export PYTHONUNBUFFERED=1 # write to output with .err extension
export WANDB_API_KEY=0c79d5a0c295ca9d1bac8a08de98f9f0196e7b2e
source otEnv/bin/activate
wandb agent market-maker/otflow/$1 --count 1
