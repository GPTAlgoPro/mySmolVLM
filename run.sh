#!/bin/bash

# 检查环境
echo "检查环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3 命令"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
if [ ! -f "requirements.txt" ]; then
    echo "错误: 未找到 requirements.txt 文件"
    exit 1
fi

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 检查模型和数据目录
echo "检查模型和数据目录..."
if [ ! -d "model/SmolVLM2-256M-Video-Instruct" ] || [ ! -d "model/Qwen3-0.6B" ] || [ ! -d "data/the_cauldron" ]; then
    echo "警告: 模型或数据目录不完整，正在运行下载脚本..."
    
    # 检查下载脚本是否存在
    if [ ! -f "download_resource.sh" ]; then
        echo "错误: 未找到 download_resource.sh 脚本"
        exit 1
    fi
    
    # 运行下载脚本
    echo "运行下载脚本..."
    bash download_resource.sh
    
    # 再次检查目录
    if [ ! -d "model/SmolVLM2-256M-Video-Instruct" ] || [ ! -d "model/Qwen3-0.6B" ] || [ ! -d "data/the_cauldron" ]; then
        echo "错误: 下载失败，请手动检查 download_resource.sh 脚本"
        exit 1
    fi
fi

# 创建输出目录
echo "创建输出目录..."
mkdir -p model/qwen-smovlm

# 启动训练
echo "启动训练..."
if [ -f "cocoqa_train.yaml" ]; then
    echo "使用 cocoqa_train.yaml 配置文件进行训练"
    python3 train.py cocoqa_train.yaml
else
    echo "使用默认参数进行训练"
    python3 train.py --train_data cocoqa --output_dir ./model/qwen-smovlm
fi

echo "训练完成!"
echo "模型已保存到 ./model/qwen-smovlm 目录"
echo "可以使用以下命令进行推理测试:"
echo "python3 inference.py --image_path ./resource/dog.png --question \"图中有什么动物?\""