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
#SBATCH --time=00:45:00

source otEnv/bin/activate

split_path=${2}/${1}

python trainLargeOTflow.py \
--data prosumer \
--data_split_path split_path \
--m 256 \
--batch_size 64 \
--test_batch_size 64 \
--save experiments/cnf/$2 \
--lr 0.004 \
--alph_C=500 \
--alph_R=15 \
--weight_decay 0.08 \
--use_num_days_data 365 \
--prosumer_name $1

