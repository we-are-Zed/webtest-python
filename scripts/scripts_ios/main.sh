#!/bin/bash

# 检查参数是否提供
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "用法: ./script.sh <session> <profile_list>"
    exit 1
fi

SESSION=$1
PROFILES=$2
WEBTEST_RUNNING="webtest_running"

for PROFILE in $PROFILES; do
    echo "Running $PROFILE..."

    # 杀掉 Chrome 和 chromedriver 的进程
    pkill -f "chrome" 2>/dev/null
    pkill -f "chromedriver" 2>/dev/null

    # 如果 webtest_running 文件存在，删除它
    if [ -e "$WEBTEST_RUNNING" ]; then
        rm "$WEBTEST_RUNNING"
    fi

    # 创建 webtest_running 文件表示正在运行
    touch "$WEBTEST_RUNNING"

    # 激活虚拟环境并运行 Python 脚本
    conda activate webtest && python ./main.py --profile="$PROFILE" --session="$SESSION"

    # 删除 webtest_running 文件表示完成
    rm "$WEBTEST_RUNNING"

    # 等待5秒
    sleep 5

    # 再次杀掉 Chrome 和 chromedriver 的进程
    pkill -f "chrome" 2>/dev/null
    pkill -f "chromedriver" 2>/dev/null

    # 再次等待5秒
    sleep 5

    # 可选的文件复制部分（如果需要）
    # CURRENT_DATE=$(date "+%Y_%m_%d")
    # CURRENT_TIME=$(date "+%H_%M_%S")
    # if [ ! -d "ExpData" ]; then
    #    
