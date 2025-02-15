#!/bin/bash
#SBATCH --job-name=test_GRPO            # Name of the job
#SBATCH --output=log/output_%j.log           
#SBATCH --error=log/error_%j.log             # Error file
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks (typically 1 for GPU jobs)
#SBATCH --gpus-per-task=4               # Number of GPUs
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=80G
#SBATCH --time=00:30:00                  # Maximum runtime (HH:MM:SS)
#SBATCH --partition=pli 
#SBATCH --account=jasonleegroup                  # GPU partition (adjust based on your system)
#SBATCH --mail-type=end
#SBATCH --mail-user=rqzhang@berkeley.edu

export uid="$(date +%Y%m%d-%H%M%S)" 
export WANDB_DISABLED=true
export TRITON_CACHE_DIR="../../../../../scratch/gpfs/gt2974/openR1/cache"
export TRANSFORMERS_CACHE="../../../../../scratch/gpfs/gt2974/openR1/cache"
export HF_DATASETS_CACHE="../../../../../scratch/gpfs/gt2974/openR1/cache"
export TORCH_EXTENSIONS_DIR="../../../../../scratch/gpfs/gt2974/openR1/cache"

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --num_processes=3 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml > logs/grpo_Qwen1.5B_${uid}.txt 2>&1

