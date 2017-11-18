#!/bin/bash
#SBATCH --partition=slurm_shortgpu
#SBATCH --time=0-00:05:00 # run time in days-hh:mm:ss
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --error=/srv/home/rpohlman2/fall2017_ece759/759--rpohlman/Project/results/trial.err
#SBATCH --output=/srv/home/rpohlman2/fall2017_ece759/759--rpohlman/Project/results/trial.out
#SBATCH --gres=gpu:1

./project1 5 5 3 3 5 5