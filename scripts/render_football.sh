#!/bin/sh
env="football"
map=$1
seed=$2
algo="happo"
exp="render"
running_max=1
kl_threshold=0.06
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}"
for number in $(seq ${running_max}); do
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_football.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
        --kl_threshold ${kl_threshold} --n_eval_rollout_threads 100 --eval_episodes 100 --eval_only \
        --use_eval --share_reward --use_recurrent_policy --autoregressive \
        --model_dir results/football/${map}/happo/happo/${seed}/wandb/latest-run/files # --autoregressive
done
