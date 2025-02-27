#!/bin/bash

# 检查是否在 "scripts" 目录中，如果是则切换到上一级目录
CURRENT_DIR="${PWD##*/}"
if [ "$CURRENT_DIR" == "scripts_ios" ]; then
    cd ..
fi

# 定义实验调用函数
#run_experiment() {
#    local group=$1
#    local config_count=$2
#    shift 2  # 跳过前两个参数
#    local configs=("$@")
#
#    # 调用实验配置
#    echo "Running experiment group $group with $config_count configurations..."
#    bash scripts_ios/main_multi.sh "$group" "$config_count" "${configs[@]}"
#}
#
## 运行实验配置
#run_experiment 1 3 "github-drl-2h-new" "github-rl-2h-q" "github-rl-2h-w"
##run_experiment 1 3 "eatingwell-drl-2h-new" "eatingwell-rl-2h-q" "eatingwell-rl-2h-w"
##run_experiment 2 3 "gamespot-drl-2h-new" "gamespot-rl-2h-q" "gamespot-rl-2h-w"
#run_experiment 2 3 "github-drl-2h-new" "github-rl-2h-q" "github-rl-2h-w"
##run_experiment 2 3 "eatingwell-drl-2h-new" "eatingwell-rl-2h-q" "eatingwell-rl-2h-w"


bash scripts_ios/main_multi.sh 2 3 github-drl-2h-new github-rl-2h-q github-rl-2h-w
#bash scripts_ios/main_multi.sh 1 3 eatingwell-drl-2h-new eatingwell-rl-2h-q eatingwell-rl-2h-w
#
#bash scripts_ios/main_multi.sh 2 3 gamespot-drl-2h-new gamespot-rl-2h-q gamespot-rl-2h-w
#bash scripts_ios/main_multi.sh 2 3 github-drl-2h-new github-rl-2h-q github-rl-2h-w
#bash scripts_ios/main_multi.sh 2 3 eatingwell-drl-2h-new eatingwell-rl-2h-q eatingwell-rl-2h-w


# 实验组名 实验配置数量 实验配置名依次排列