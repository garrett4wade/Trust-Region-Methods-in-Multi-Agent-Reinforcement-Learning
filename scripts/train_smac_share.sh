#!/bin/sh
env="StarCraft2"
map=$1
algo="happo"
actor_distill_coef=$2
critic_distill_coef=$3
exp="happo_share_a"${actor_distill_coef}"_c"${critic_distill_coef}
running_max=20
kl_threshold=0.06
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for number in `seq ${running_max}`;
do
    pkill -9 Main
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=0 python3 train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} \
    --n_rollout_threads 8 --episode_length 400 --num_env_steps 20000000 --kl_threshold ${kl_threshold} --n_eval_rollout_threads 32 \
    --share_policy --actor_distill_coef ${actor_distill_coef} --critic_distill_coef ${critic_distill_coef} \
    --use_value_active_masks --use_eval --add_center_xy --use_state_agent --use_wandb
done
