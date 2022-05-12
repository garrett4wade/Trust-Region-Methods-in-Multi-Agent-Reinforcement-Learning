#!/bin/sh
env="bridge"
algo="happo"
exp="check"
running_max=1
kl_threshold=0.06
echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in $(seq ${running_max}); do
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_bridge.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
        --n_rollout_threads 10 --episode_length 50 --use_agent_id --share_reward \
        --ppo_epoch 5 --kl_threshold ${kl_threshold} --num_env_steps 1000000 \
        --share_reward --autoregressive --no_factor --random_order #--use_wandb
done
