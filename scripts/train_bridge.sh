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
        --n_rollout_threads 50 --episode_length 50 --share_reward --sample_reuse 10 \
        --ppo_epoch 1 --kl_threshold ${kl_threshold} --num_env_steps 100000 --use_eval --n_eval_rollout_threads 32 --eval_episodes 32 --eval_interval 10 \
        --autoregressive --no_factor --random_order --share_policy
done
