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
        --n_rollout_threads 50 --episode_length 50 --data_chunk_length 10 --use_agent_id --share_reward \
        --ppo_epoch 15 --kl_threshold ${kl_threshold} --n_eval_rollout_threads 100 --num_env_steps 1000000 \
        --use_eval --share_reward --use_recurrent_policy --autoregressive --no_factor --random_order #--use_wandb
done
