import argparse
import os
from PIL import Image
import torch
import swanlab

from utils import load_model, load_processor

def parse_args():
    """
    解析命令行参数。
    
    Returns:
        argparse.Namespace: 解析后的参数。
    """
    parser = argparse.ArgumentParser(description="使用训练好的SmolVLM模型进行推理")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./model/qwen-smovlm",
        help="训练好的模型路径"
    )
    parser.add_argument(
        "--image_path", 
        type=str, 
        required=True,
        help="输入图像路径"
    )
    parser.add_argument(
        "--question", 
        type=str, 
        default="图中有什么？",
        help="关于图像的问题"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256,
        help="生成的最大token数"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="运行设备，如'cuda'、'mps'或'cpu'，默认自动检测"
    )
    return parser.parse_args()

def main():
    """
    主函数，加载模型并执行推理。
    """
    args = parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"图像文件不存在: {args.image_path}")
    
    # 加载处理器和模型
    processor = load_processor()
    model = load_model(args.device)
    
    # 如果指定了模型路径且存在，则加载训练好的模型
    if args.model_path and os.path.exists(args.model_path):
        print(f"加载训练好的模型: {args.model_path}")
        model.load_state_dict(torch.load(
            os.path.join(args.model_path, "pytorch_model.bin"),
            map_location=model.device
        ))
    
    # 准备输入
    image = Image.open(args.image_path)
    messages = [
        {
            "role": "system",
            "content": "使用中文回答所有问题。",
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.question},
            ],
        },
    ]
    texts = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True,
    )
    
    # 处理输入
    batch = processor(
        text=[texts],
        images=[[image]],
        max_length=1024,
        return_tensors="pt",
        padding_side="left",
        padding=True,
    )
    
    # 根据设备类型选择数据类型
    if model.device.type == "mps":
        try:
            batch = batch.to(model.device, dtype=torch.bfloat16)
        except RuntimeError:
            print("警告: MPS设备不支持bfloat16，回退到float32")
            batch = batch.to(model.device)
    else:
        batch = batch.to(model.device, dtype=torch.bfloat16)
    
    # 生成回答
    with torch.no_grad():
        generated_ids = model.generate(
            **batch, do_sample=False, max_new_tokens=args.max_new_tokens
        )
    
    # 解码生成的文本
    input_ids_len = batch["input_ids"].shape[1]
    generated_text = processor.batch_decode(
        generated_ids[:, input_ids_len:], skip_special_tokens=True
    )[0]
    
    # 打印结果
    print("\n" + "="*50)
    print(f"问题: {args.question}")
    print("-"*50)
    print(f"回答: {generated_text}")
    print("="*50)
    
    # 记录结果到SwanLab
    try:
        # 初始化SwanLab实验
        swanlab.init(
            experiment_name="inference_result",
            description="推理结果可视化",
            config=vars(args)
        )
        
        # 创建表格
        table = swanlab.echarts.Table()
        headers = ["问题", "回答"]
        rows = [[args.question, generated_text]]
        table.add(headers, rows)
        
        # 记录结果
        swanlab.log({
            "inference/输入图像": swanlab.Image(image),
            "inference/问题&回答": table,
        })
        
        print("\nSwanLab可视化界面: http://localhost:8888")
        print("可以在浏览器中查看推理结果")
    except Exception as e:
        print(f"\n记录SwanLab结果时出错: {e}")
        print("推理结果未记录到SwanLab，但不影响模型输出")

if __name__ == "__main__":
    main()