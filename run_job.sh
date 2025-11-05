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

EXPERIMENT="train_dynamics"
#export WANDB_API_KEY="24e6ba2cb3e7bced52962413c58277801d14bba0"
#export WANDB_RUN_GROUP=$EXPERIMENT;
SEED=$SLURM_ARRAY_TASK_ID
EXP_SUFFIX="${EXPERIMENT}_${SEED}"


python -u train_dynamics.py --suffix $EXP_SUFFIX >> ${EXP_SUFFIX}.out


# python -u learning/train_jax_ppo.py --env_name=LeapCubeRotateObjectsZAxisTouchCont --wandb_entity edhu --use_wandb --num_timesteps 1000000000 --suffix $EXP_SUFFIX >> ${EXP_SUFFIX}.out

# python learning/train_jax_ppo.py --env_name=LeapCubeRotateZAxisTouchDropout50 --domain_randomization --wandb_entity edhu --use_wandb --num_timesteps 1000000000 --suffix $EXP_SUFFIX >> ${EXP_SUFFIX}.out

# python -u learning/train_jax_ppo.py --env_name=LeapCubeReorientTouch --use_wandb --domain_randomization --num_timesteps 10000000000 --suffix $EXP_SUFFIX >> ${EXP_SUFFIX}.out

# python -u learning/train_jax_ppo.py --env_name=LeapCubeReorient --use_wandb --domain_randomization --num_timesteps 10000000000 --suffix $EXP_SUFFIX >> ${EXP_SUFFIX}.out

wait
