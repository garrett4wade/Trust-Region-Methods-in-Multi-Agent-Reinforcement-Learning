#!/bin/sh
env="StarCraft2"
map=$1
actor_distill_coef=$2
critic_distill_coef=$3
algo="happo"
# exp="happo2_share_ar_a${actor_distill_coef}_c${critic_distill_coef}"
exp="happo_ar"
running_max=20
kl_threshold=0.06
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in `seq ${running_max}`;
do
    # pkill -9 Main
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
    --n_rollout_threads 8 --episode_length 400 --num_env_steps 10000000 --kl_threshold ${kl_threshold} --n_eval_rollout_threads 32 \
    --use_value_active_masks --use_eval --add_center_xy --use_state_agent --use_wandb --autoregressive #--share_policy --actor_distill_coef ${actor_distill_coef} --critic_distill_coef ${critic_distill_coef}

    rm -rf core.*
done
