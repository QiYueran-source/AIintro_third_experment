#!/bin/bash

# 从 /mnt/e/齐悦然的文件/forwsl 剪切所有文件到 /home/frank/files/programs/AI引论第三次实验/tmp

SOURCE_DIR="/mnt/e/齐悦然的文件/forwsl"
TARGET_DIR="/home/frank/files/programs/AI引论第三次实验/tmp"

# 确保目标目录存在
mkdir -p "$TARGET_DIR"

# 剪切所有文件从源目录到目标目录
mv "$SOURCE_DIR"/* "$TARGET_DIR"/

echo "文件已成功从 $SOURCE_DIR 剪切到 $TARGET_DIR"
