#!/bin/bash
#SBATCH --job-name=vasp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=24:00:00
#SBATCH --mem=0
# SBATCH --partition=your_partition
# SBATCH --account=your_account

module purge
# module load vasp/6.x  # edit for your cluster
# module load intel-mpi  # or openmpi

srun vasp_std > vasp.out
