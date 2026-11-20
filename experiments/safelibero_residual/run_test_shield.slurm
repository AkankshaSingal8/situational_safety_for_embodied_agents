#!/bin/bash
#SBATCH -p pleiades
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /data/asingal2/slurm_logs/%x_%j.out
cd /home/asingal2/safety/situational_safety_for_embodied_agents/experiments/safelibero_residual
./pleiades_run.sh python test_shield.py
