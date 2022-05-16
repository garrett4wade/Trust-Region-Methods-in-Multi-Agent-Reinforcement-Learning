#!/bin/sh
env="bridge"
algo="happo"
exp="render"
running_max=1
kl_threshold=0.06
echo "env is ${env}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in $(seq ${running_max}); do
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_bridge.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
        --episode_length 50 --share_reward --kl_threshold ${kl_threshold} --share_policy \
        --autoregressive --no_factor --random_order --use_render --render_episodes 5 --cuda --use_eval --n_eval_rollout_threads 1 \
        --n_render_rollout_threads 1 \
        --model_dir /home/fw/workspace/Trust-Region-Methods-in-Multi-Agent-Reinforcement-Learning/scripts/results/bridge/happo/check/6782/run1/models
done
