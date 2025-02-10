#!/bin/bash
#SBATCH --job-name=llm
#SBATCH -o test_py.o
#SBATCH -A llm_workshop2024
#SBATCH -p aisc_short
#SBATCH --reservation=llm24
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=0:02:00    # Days-HH:MM:SS

nvidia-smi
#sleep 120    #  s: seconds (default) m: minutes h: hours d: days
