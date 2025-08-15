from dataclasses import dataclass, field

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMConnector


def load_processor():
    """
    加载并配置多模态处理器。
    
    该函数加载SmolVLM2处理器和Qwen3分词器，并进行自定义配置以支持多模态输入处理。
    
    Returns:
        AutoProcessor: 配置好的SmolVLM2处理器，已集成Qwen3分词器和自定义token设置。
    
    Example:
        >>> processor = load_processor()
        >>> processor.tokenizer  # 访问集成的Qwen3分词器
    """
    smolvlm2_processor = AutoProcessor.from_pretrained(
        "model/SmolVLM2-256M-Video-Instruct"
    )
    qwen3_tokenizer = AutoTokenizer.from_pretrained("model/Qwen3-0.6B")

    smolvlm2_processor.tokenizer = qwen3_tokenizer
    with open("chat_template.jinja", "r") as f:
        smolvlm2_processor.chat_template = f.read()
    smolvlm2_processor.fake_image_token = "<vision_start>"
    smolvlm2_processor.image_token = "<|image_pad|>"
    smolvlm2_processor.image_token_id = 151655
    smolvlm2_processor.end_of_utterance_token = "<im_end>"
    smolvlm2_processor.global_image_token = "<|vision_pad|>"
    smolvlm2_processor.video_token = "<|video_pad|>"

    return smolvlm2_processor


def load_model(device="cuda:0"):
    """
    加载并配置多模态模型。
    
    该函数加载SmolVLM2视觉语言模型和Qwen3语言模型，并将它们集成为一个统一的多模态模型。
    主要进行以下操作：
    1. 加载预训练模型
    2. 创建新的连接器配置
    3. 替换连接器层
    4. 集成Qwen3语言模型到SmolVLM2
    5. 统一配置参数
    
    Args:
        device (str, optional): 模型加载的设备。默认为"cuda:0"。
    
    Returns:
        AutoModelForImageTextToText: 配置好的多模态模型，集成了SmolVLM2视觉模型和Qwen3语言模型。
    
    Example:
        >>> model = load_model(device="cuda:1")
        >>> model.generate(...)  # 使用模型生成内容
    """
    smolvlm2_02B_model = AutoModelForImageTextToText.from_pretrained(
        "model/SmolVLM2-256M-Video-Instruct",
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    ).to(device)
    qwen3_06b_model = AutoModelForCausalLM.from_pretrained(
        "model/Qwen3-0.6B", torch_dtype=torch.bfloat16
    ).to(device)

    # 构建配置并且创建连接器
    @dataclass
    class VisionConfig:
        """视觉模型配置类，定义视觉模型的隐藏层大小。"""
        hidden_size: int = 768

    @dataclass
    class TextConfig:
        """文本模型配置类，定义文本模型的隐藏层大小。"""
        hidden_size: int = 1024

    @dataclass
    class ConnectConfig:
        """连接器配置类，定义视觉和文本模型的连接参数。"""
        scale_factor: int = 4
        vision_config: VisionConfig = field(default_factory=VisionConfig)
        text_config: TextConfig = field(default_factory=TextConfig)

    new_connector_config = ConnectConfig()

    # 替换 SigLit 到 LLM 的 connector 层
    new_connector = SmolVLMConnector(new_connector_config).to(device).to(torch.bfloat16)
    smolvlm2_02B_model.model.connector = new_connector

    # 替换 VL 模型的 LLM 部分
    # 替换text模型和head
    smolvlm2_02B_model.model.text_model = qwen3_06b_model.model
    smolvlm2_02B_model.lm_head = qwen3_06b_model.lm_head
    # 替换词表大小
    smolvlm2_02B_model.vocab_size = qwen3_06b_model.vocab_size
    smolvlm2_02B_model.model.vocab_size = qwen3_06b_model.vocab_size
    smolvlm2_02B_model.config.vocab_size = qwen3_06b_model.vocab_size
    smolvlm2_02B_model.config.text_config.vocab_size = qwen3_06b_model.vocab_size
    smolvlm2_02B_model.model.config.vocab_siz = qwen3_06b_model.vocab_size
    smolvlm2_02B_model.model.config.text_config.vocab_size = qwen3_06b_model.vocab_size
    # 替换图像token
    smolvlm2_02B_model.image_token_id = 151655
    smolvlm2_02B_model.model.image_token_id = 151655
    smolvlm2_02B_model.config.image_token_id = 151655
    smolvlm2_02B_model.model.config.image_token_id = 151655
    # 替换模型生成停止符
    smolvlm2_02B_model.generation_config.eos_token_id = 151645
    return smolvlm2_02B_model
