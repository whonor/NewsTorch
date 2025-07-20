#!/bin/bash

# 检查 requirements.txt 是否存在
if [ ! -f "requirements.txt" ]; then
  echo "Error: requirements.txt not found in the current directory."
  exit 1
fi

# 使用 pip 安装 requirements.txt 中的所有包
echo "Installing packages from requirements.txt..."
pip install --no-cache-dir -r requirements.txt

# 检查安装是否成功
if [ $? -eq 0 ]; then
  echo "All packages installed successfully."
else
  echo "Error occurred during package installation."
  exit 1
fi