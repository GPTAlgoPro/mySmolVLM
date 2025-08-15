# mySmolVLM - 一种超小中文多模态模型的“拼接微调”方法

* 原始项目参见：https://github.com/KwaiGroup/Qwen3-SmVL-VLM
* 本项目尝试基于上述原始项目做一个扩展，最终希望基于拼接微调的miniVLM开发一个可以在iPhone14 pro以上设备本地运行的iOS应用

## 摘要
  
最近Huggingface团队发布了超小多模态模型SmolVLM2，可以做到端侧1GB显存推理。在怀着惊喜试用后发现，虽然模型有极其强大的视觉文本理解能力，但是模型却无法理解中文。因此萌生了把当前中文小模型扛把子Qwen3与SmolVLM2直接微调拼接的想法。本项目介绍一种模型拼接的思路，将SmolVLM2的视觉模块（0.09B）与Qwen3最小的模型（0.6B）进行对齐微调，最终使Qwen3模型具备一定的视觉理解能力。

## 特性

- 支持 Apple Silicon (M1/M2/M3) 和 CUDA GPU 设备
- 自动设备检测和优化
- 支持多种多模态数据集
- 参数冻结策略，实现高效微调
- SwanLab 集成，提供训练可视化
- 支持 YAML 配置文件

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- 训练至少 40GB+ 内存 (推荐 48GB+)
- 支持 MPS 的 Apple Silicon 设备或 CUDA 兼容的 GPU

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/mySmolVLM.git
cd mySmolVLM
```

2. 安装依赖：

```bash
pip install -r requirements.txt

# 安装 modelscope（用于下载模型和数据集）
pip install modelscope
```

3. 下载预训练模型和数据集：

```bash
# 运行下载脚本
bash download_resource.sh
```

下载脚本将自动执行以下操作：
- 下载 Qwen3-0.6B 模型到 ./model/Qwen3-0.6B 目录
- 下载 SmolVLM2-256M-Video-Instruct 模型到 ./model/SmolVLM2-256M-Video-Instruct 目录
- 下载 the_cauldron 数据集到 ./data/the_cauldron 目录

## 数据准备

数据集会通过 `download_resource.sh` 脚本自动下载到 `data/the_cauldron` 目录下。该数据集包含多种多模态数据，包括：

- cocoqa
- chartqa
- finqa
- aokvqa
- figureqa
- 等多种多模态数据集

如果您想手动下载数据集，可以使用以下命令：

```bash
modelscope download --dataset AI-ModelScope/the_cauldron --local_dir ./data/the_cauldron
```

## 使用方法

### 使用 YAML 配置文件训练

```bash
python train.py cocoqa_train.yaml
```

或

```bash
python train.py full_train.yaml
```

### 使用命令行参数训练

```bash
python train.py --train_data cocoqa --output_dir ./model/qwen-smolvlm-cocoqa --learning_rate 1e-4
```

## 配置参数

主要训练参数包括：

- `train_data`: 训练数据集名称，默认为 "cocoqa"
- `output_dir`: 模型输出目录
- `learning_rate`: 学习率，默认为 1e-4
- `per_device_train_batch_size`: 每个设备的训练批次大小，默认为 1
- `gradient_accumulation_steps`: 梯度累积步数，默认为 4
- `bf16`: 是否使用 bfloat16 精度，默认为 True（在 MPS 设备上会自动检测兼容性）

更多参数请参考 `train.py` 中的 `MyTrainArgs` 类。

## Apple Silicon 优化

对于 Apple Silicon 设备，框架会自动：

1. 检测 MPS 后端并启用相应优化
2. 检测 bfloat16 支持情况，不支持时自动回退到 float32
3. 调整梯度累积步数，优化内存使用

## 推理示例

训练完成后，模型会自动使用示例图片进行推理测试，并通过 SwanLab 记录结果。

## 自定义训练

如需自定义训练配置，可以：

1. 修改现有的 YAML 配置文件
2. 创建新的 YAML 配置文件
3. 直接通过命令行参数指定训练参数

## 许可证

MIT

## 贡献

欢迎提交 Issue 和 Pull Request！