#!/bin/sh
exp="check"
running_max=1
for number in $(seq ${running_max}); do
    echo "the ${number}-th running:"
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=~/workspace/Trust-Region-Methods-in-Multi-Agent-Reinforcement-Learning \
        python3 -m scripts.train.train_ned90 --env_name ned90 --algorithm_name happo --experiment_name ${exp} \
        --n_rollout_threads 180 --num_mini_batch 1 --episode_length 100 --data_chunk_length 10 --use_value_active_masks \
        --hidden_size 256 --layer_N 3 --use_popart --lr 1e-5 --critic_lr 1e-5 --weight_decay 1e-6 --entropy_coef 1e-3 --max_grad_norm 10 \
        --ppo_epoch 10 --use_recurrent_policy --share_policy --no_factor #--use_wandb
done
