#!/bin/bash
maps=(
    "1c3s5z"
    "2s3z"
    "3s5z"
    "3s5z_vs_3s6z"
    "5m_vs_6m"
    "6h_vs_8z"
    "10m_vs_11m"
    "corridor"
    "MMM2"
)
len=${#maps[@]}
for (( i=0; i<$len; i++ ));
do
    screen -d -m -S ${maps[$i]} bash train_smac.sh ${maps[$i]} 0.0 0.0
    echo "${maps[$i]}"
done