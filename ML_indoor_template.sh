#!/bin/bash
#SBATCH --job-name=ML_indoor
#SBATCH --mail-type=NONE
# SBATCH --partition=test
#SBATCH --partition=normal
#SBATCH --ntasks=1             # Run with single wall clock
#SBATCH --cpus-per-task=1      # Number of CPU cores per task
# SBATCH --nodes=1              # Run all processes on a single node
#SBATCH --mem=16GB
#SBATCH --time=4-00:00:00
#SBATCH --array=5-47
#SBATCH --output=/lustre/scratch/laukkara/ML_indoor/%A_%a_stdout.log  # Standard output and error log
#SBATCH --error=/lustre/scratch/laukkara/ML_indoor/%A_%a_stderr.log

# One task is always allocated to one node by default
# %A and %a are filename patterns that sbatch allows, they are not available in general
# $SLURM_ARRAY_TASK_ID is the environment variable, that can be used in code below

# This causes the bash script to abort immediately if any of the lines throw and error
#set -e

var="Hyper Drive"
echo "Good day, sir, it is $(date) and we are ready to commence ${var}"
echo "conda version: $(conda --version)"

pwd; hostname; date

echo "module list:"

# With partition=test and module purge commented, error is given on conda --version 
# With partition=test and module purge uncommented, error is given on conda --version
# With partition=normal and module purge commented, error is given on conda --version
# With partition=normal and module purge uncommented, error is given on conda --version

module purge
module load fgci-common
module load miniconda3
module list


conda info --envs

# conda activate ML_indoor # This didn't work
# conda activate /home/laukkara/.conda/envs/ML_indoor
source activate /home/laukkara/.conda/envs/ML_indoor # This didn't give error
# source activate ML_indoor # This didn't give error -> It did give an error in another case -> Then it didn't give an error again

# idx_start=$SLURM_ARRAY_TASK_ID
# idx_end=$(( $SLURM_ARRAY_TASK_ID + 1 ))
# echo $idx_start $idx_end
# echo "First: $SLURM_ARRAY_TASK_ID, second: $(( $SLURM_ARRAY_TASK_ID + 1 ))"
echo "Begin!"


python3 /home/laukkara/github/ML_indoor/main.py 0 5 $SLURM_ARRAY_TASK_ID $(( $SLURM_ARRAY_TASK_ID + 1 )) 2 3 3 10 1 0 0


date

