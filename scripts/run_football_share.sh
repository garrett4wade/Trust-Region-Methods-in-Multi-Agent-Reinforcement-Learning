#!/bin/bash
maps=(
    # "academy_3_vs_1_with_keeper"
    # "academy_run_pass_and_shoot_with_keeper" 
    "academy_corner"
    "academy_counterattack_easy"
    "academy_counterattack_hard"
    # "academy_pass_and_shoot_with_keeper"
)
shortcut=(
    # "3v1"
    # "rps"
    "corner"
    "ca_easy"
    "ca_hard"
    # "ps"
)
len=${#maps[@]}
for (( i=0; i<$len; i++ ));
do
    screen -d -m -S g1${shortcut[$i]}_share_ar bash train_football_share.sh ${maps[$i]} 0.0 0.0
    echo "${maps[$i]}"
done