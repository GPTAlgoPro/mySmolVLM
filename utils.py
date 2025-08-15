import torch
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoProcessor

def load_processor():
    """
    加载模型处理器。
    
    该函数加载用于处理图像和文本输入的处理器。
    
    Returns:
        AutoProcessor: 加载的处理器实例。
        
    Example:
        >>> processor = load_processor()
        >>> inputs = processor(text=["这是什么图片？"], images=[image])
    """
    processor = AutoProcessor.from_pretrained("model/SmolVLM2-256M-Video-Instruct")
    return processor


def load_model(device=None):
    """
    加载模型并进行必要的配置。
    
    该函数加载SmolVLM2模型和Qwen3模型，并进行必要的配置，如词汇表大小匹配等。
    
    Args:
        device (str, optional): 模型加载的设备。如果为None，则自动检测可用设备。
        
    Returns:
        AutoModelForImageTextToText: 配置好的SmolVLM2模型实例。
        
    Example:
        >>> model = load_model("cuda")
        >>> outputs = model(**inputs)
    """
    if device is None:
        import platform
        if platform.system() == "Darwin" and platform.processor() == "arm":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 根据设备类型选择数据类型
    if device == "mps":
        try:
            # 尝试使用bfloat16
            dtype = torch.bfloat16
            smolvlm2_02B_model = AutoModelForImageTextToText.from_pretrained(
                "model/SmolVLM2-256M-Video-Instruct",
                torch_dtype=dtype,
                _attn_implementation="eager",
            ).to(device)
            qwen3_06b_model = AutoModelForCausalLM.from_pretrained(
                "model/Qwen3-0.6B", torch_dtype=dtype
            ).to(device)
        except RuntimeError:
            # 如果MPS不支持bfloat16，回退到float32
            print("警告: MPS设备不支持bfloat16，回退到float32")
            smolvlm2_02B_model = AutoModelForImageTextToText.from_pretrained(
                "model/SmolVLM2-256M-Video-Instruct",
                _attn_implementation="eager",
            ).to(device)
            qwen3_06b_model = AutoModelForCausalLM.from_pretrained(
                "model/Qwen3-0.6B"
            ).to(device)
    else:
        # CUDA或CPU设备使用bfloat16
        smolvlm2_02B_model = AutoModelForImageTextToText.from_pretrained(
            "model/SmolVLM2-256M-Video-Instruct",
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",
        ).to(device)
        qwen3_06b_model = AutoModelForCausalLM.from_pretrained(
            "model/Qwen3-0.6B", torch_dtype=torch.bfloat16
        ).to(device)

    # 确保词汇表大小匹配
    smolvlm2_02B_model.model.config.vocab_size = qwen3_06b_model.vocab_size
    
    # 打印模型信息
    print(f"SmolVLM2模型加载完成，设备: {device}")
    print(f"词汇表大小: {smolvlm2_02B_model.model.config.vocab_size}")
    
    return smolvlm2_02B_model


def freeze_model(model):
    """
    冻结模型的部分参数，只保留特定层可训练。
    
    该函数冻结模型的文本编码器和视觉编码器参数，只保留连接器层和输出层可训练。
    
    Args:
        model (torch.nn.Module): 需要冻结参数的模型。
        
    Returns:
        torch.nn.Module: 参数已冻结的模型，只有连接器层可训练。
        
    Example:
        >>> model = load_model()
        >>> model = freeze_model(model)
        >>> print_trainable_parameters(model)  # 查看可训练参数数量
    """
    # 冻结文本编码器和视觉编码器
    for _, param in model.model.text_model.named_parameters():
        param.requires_grad = False
    for _, param in model.model.vision_model.named_parameters():
        param.requires_grad = False
    
    # 连接器层和输出层保持可训练
    # 注意：这里假设连接器层是通过 model.model.connector 访问的
    # 如果实际结构不同，可能需要调整
    
    return model


def print_trainable_parameters(model):
    """
    打印模型中可训练参数的数量和比例。
    
    该函数计算模型中可训练参数的数量和总参数数量，并打印它们的比例。
    
    Args:
        model (torch.nn.Module): 需要分析的模型。
        
    Returns:
        None: 函数直接打印结果，不返回值。
        
    Example:
        >>> model = load_model()
        >>> print_trainable_parameters(model)  # 打印可训练参数比例
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"可训练参数: {trainable_params:,d} ({100 * trainable_params / all_param:.2f}%)"
    )
    print(f"所有参数: {all_param:,d}")