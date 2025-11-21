#!/bin/bash
#SBATCH --partition=dgx-b200
#SBATCH --ntasks=1                 # one task per node
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --array=11-11

cd ~/projects/tiny_dreamer_4
source .venv/bin/activate

EXPERIMENT="train_bc_rew_heads_flippedrew"
#export WANDB_API_KEY="24e6ba2cb3e7bced52962413c58277801d14bba0"
#export WANDB_RUN_GROUP=$EXPERIMENT;
SEED=$SLURM_ARRAY_TASK_ID
EXP_SUFFIX="${EXPERIMENT}_${SEED}"


# 1. Train dynamics model.
# python -u train_dynamics.py --suffix $EXP_SUFFIX >> ${EXP_SUFFIX}.out

# 2. Finetune dynamics model with BC and reward prediction.
python -u train_bc_rew_heads.py --suffix $EXP_SUFFIX >> ${EXP_SUFFIX}.out
wait
