#!/bin/bash

echo "inini"
# 检查参数是否提供
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./script.sh <session> <total_loop> <profile1> <profile2> ..."
    exit 1
fi

# 设定变量
SESSION=$1
TOTAL_LOOP=$2
shift 2 # 移动位置参数，跳过前两个参数

# 关闭现有的 Chrome 和 Chromedriver 进程
pkill -f "chrome" 2>/dev/null
pkill -f "chromedriver" 2>/dev/null

# 清除旧的 agent 运行标记文件
for (( i=1; i<=TOTAL_LOOP; i++ )); do
    AGENT_FILE="agent${i}_running"
    if [ -e "$AGENT_FILE" ]; then
        rm "$AGENT_FILE"
    fi
done

echo "Running $TOTAL_LOOP settings..."

LOOP_ROUND=1

# 逐个运行配置文件
while [ "$#" -gt 0 ]; do
    PROFILE=$1
    echo "Running $PROFILE..."

    # 创建运行标记文件
    AGENT_FILE="agent${LOOP_ROUND}_running"
    touch "$AGENT_FILE"

    # 运行 Python 脚本并在完成后删除运行标记文件
    python ../main.py --profile="$PROFILE" --session="$SESSION"
    rm "$AGENT_FILE"

    shift # 继续下一个 profile
    ((LOOP_ROUND++))
done

# 调用其他脚本，等待所有任务完成
scripts_ios/CheckFinish_multi.sh "$TOTAL_LOOP"
sleep 5

# 再次关闭 Chrome 和 Chromedriver 进程
pkill -f "chrome" 2>/dev/null
pkill -f "chromedriver" 2>/dev/null
sleep 5

echo "Copying files..."

# 创建目录并移动文件
if [ ! -d "ExpData" ]; then
    mkdir "ExpData"
fi
# mv "webtest_output/result" "ExpData/" # 如果需要，将输出文件移动到 ExpData 目录

echo "All tasks completed."
