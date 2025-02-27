#!/bin/bash

echo "Waiting for agents to finish..."

# 检查用户是否提供了一个数字参数
if [ -z "$1" ]; then
    echo "请提供要检查的代理数量。"
    exit 1
fi

# 循环遍历每个代理文件
for (( i=1; i<=$1; i++ )); do
    while [ -e "agent${i}_running" ]; do
        sleep 1
    done
done

echo "All agents have finished."
