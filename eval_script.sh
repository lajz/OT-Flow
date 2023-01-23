#!/bin/bash
# Job name:
#SBATCH --job-name=train
#
# Account:
#SBATCH -A m3691
#
# Partition:
#SBATCH -C gpu
#
# Number of nodes:
#SBATCH -N 1
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH -c 2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH -G 1
#
# Wall clock limit:
#SBATCH --time=1:30:00
#SBATCH -o out/slurm.%N.%j.out # STDOUT
#SBATCH -e out/slurm.%N.%j.err # STDERR
## Command(s) to run (example):
export PYTHONUNBUFFERED=1 # write to output with .err extension
source otEnv/bin/activate

python gensynth.py \
	--experiment_folder_path experiments/cnf/grouped_prices_o15 \
	--raw_data_folder_name grouped_prices_o15 \
	--use_num_days_data 365