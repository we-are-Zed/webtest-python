#!/bin/bash

# 检查是否在 "scripts_ios" 目录中，如果是则切换到上一级目录
CURRENT_DIR="${PWD##*/}"
if [ "$CURRENT_DIR" == "scripts_ios" ]; then
    cd ..
fi

# 调用实验配置
# 实验组名和实验配置名
bash scripts_ios/main.sh 1 recipetineats-drl-2h-new
bash scripts_ios/main.sh 1 allrecipes-drl-2h-new
bash scripts_ios/main.sh 1 recipetineats-drl-2h-allrecipes-update
bash scripts_ios/main.sh 1 allrecipes-drl-2h-recipetineats-update
bash scripts_ios/main.sh 1 recipetineats-drl-2h-allrecipes-stop
bash scripts_ios/main.sh 1 allrecipes-drl-2h-recipetineats-stop
bash scripts_ios/main.sh 1 recipetineats-rl-2h-q
bash scripts_ios/main.sh 1 recipetineats-rl-2h-w
bash scripts_ios/main.sh 1 allrecipes-rl-2h-q
bash scripts_ios/main.sh 1 allrecipes-rl-2h-w
