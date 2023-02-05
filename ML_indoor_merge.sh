#!/bin/bash
#SBATCH --job-name=ML_merge
#SBATCH --mail-type=NONE
# SBATCH --partition=test
#SBATCH --partition=normal
#SBATCH --ntasks=1             # Run with single wall clock
#SBATCH --cpus-per-task=1      # Number of CPU cores per task
# SBATCH --nodes=1              # Run all processes on a single node
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --output=/lustre/scratch/laukkara/ML_indoor/slurm_logs/%j_stdout.log  # Standard output and error log
#SBATCH --error=/lustre/scratch/laukkara/ML_indoor/slurm_logs/%j_stderr.log

# One task is always allocated to one node by default
# %A and %a are filename patterns that sbatch allows, they are not available in general
# $SLURM_ARRAY_TASK_ID is the environment variable, that can be used in code below

# This causes the bash script to abort immediately if any of the lines throw and error
#set -e

var="Hyper Drive"
echo "Good day, sir, it is $(date) and we are ready to commence ${var}"

pwd; hostname; date


# With partition=test and module purge commented, error is given on conda --version 
# With partition=test and module purge uncommented, error is given on conda --version
# With partition=normal and module purge commented, error is given on conda --version
# With partition=normal and module purge uncommented, error is given on conda --version


module purge
module --ignore-cache load fgci-common
module --ignore-cache load miniconda3
module list

# conda activate ML_indoor # This didn't work
# conda activate /home/laukkara/.conda/envs/ML_indoor
source activate /home/laukkara/.conda/envs/ML_indoor # This didn't give error
# source activate ML_indoor # This did not give error -> It did give an error in another case -> Then it didn't give an error again


echo "conda version: $(conda --version)"

conda info --envs

python3 /home/laukkara/github/ML_indoor/myMerge.py

date

