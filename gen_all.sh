#!/bin/bash
# Job name:
#SBATCH --job-name=flow_microgrid
#
# Account:
#SBATCH --account=fc_ntugame
#
# Partition:
#SBATCH --partition=savio2_1080ti
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
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit (8hrs):
#SBATCH --time=07:00:00

source otEnv/bin/activate

split_path=${2}/${1}

python gensynth.py \
--experiment_folder_path experiments/cnf/reward_evaluation_new \
--raw_data_folder_name reward_evaluation_new \
--use_num_days_data 365
