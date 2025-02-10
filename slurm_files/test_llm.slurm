#!/bin/bash
#SBATCH --job-name=llm_rxy216
#SBATCH -o llm_%j_gsm8k.o
#SBATCH -A llm_workshop2024
#SBATCH -p aisc_short
#SBATCH --reservation=llm24
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=0:03:00    # Days-HH:MM:SS

singularity exec --nv --bind /mnt/pan/courses/llm24/xxh584:/mnt/pan/courses/llm24/xxh584 --bind /home/rxy216/llm:/home/rxy216/llm /mnt/pan/courses/llm24/xxh584/llm_ws_0.2.sif python /home/rxy216/llm/llm/evaluate/gsm8k.py
