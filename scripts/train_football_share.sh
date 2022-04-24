#!/bin/sh
env="football"
map=$1
algo="happo"
distill_coef=$2
exp="happo_share_"${distill_coef}
running_max=20
kl_threshold=0.06
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in $(seq ${running_max}); do
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_football.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
        --n_training_threads 32 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 \
        --ppo_epoch 5 --kl_threshold ${kl_threshold} --n_eval_rollout_threads 8 \
        --use_eval --share_reward --share_policy --distill_coef ${distill_coef} --use_recurrent_policy --use_wandb
done
