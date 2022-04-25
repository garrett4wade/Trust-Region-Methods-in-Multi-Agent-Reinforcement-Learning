#!/bin/sh
env="football"
map=$1
algo="happo"
actor_distill_coef=$2
critic_distill_coef=$3
exp="happo_share_a"${actor_distill_coef}"_c"${critic_distill_coef}
running_max=20
kl_threshold=0.06
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in $(seq ${running_max}); do
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_football.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
        --n_rollout_threads 50 --num_mini_batch 2 --episode_length 200 --data_chunk_length 10 \
        --ppo_epoch 15 --kl_threshold ${kl_threshold} --n_eval_rollout_threads 100 \
        --use_eval --share_reward --share_policy --actor_distill_coef ${actor_distill_coef} --critic_distill_coef ${critic_distill_coef} --use_recurrent_policy --use_wandb
done
