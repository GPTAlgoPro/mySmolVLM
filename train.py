import os
import sys
from dataclasses import dataclass
from functools import partial
from PIL import Image

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint
import datasets
import swanlab

from utils import load_model, load_processor

# 设置默认设备为CUDA
device = "cuda"

################
# 加载数据集
################
def load_mm_data(select_data):
    """
    加载多模态数据集。
    
    该函数根据指定的数据集名称加载相应的多模态数据集，支持单个数据集或所有数据集的加载。
    
    Args:
        select_data (str): 要加载的数据集名称，可以是具体数据集名称或"all"表示加载所有数据集。
    
    Returns:
        datasets.DatasetDict: 包含训练集和测试集的数据集字典。
    
    Raises:
        ValueError: 当指定的数据集名称不存在时抛出异常。
    
    Example:
        >>> data = load_mm_data("cocoqa")
        >>> print(f"训练集大小: {len(data['train'])}")
        >>> print(f"测试集大小: {len(data['test'])}")
    """
    all_data_names = [
        "chartqa",
        "finqa",
        "aokvqa",
        # "mimic_cgd",  # bad dataset
        "figureqa",
        "diagram_image_to_text",
        "geomverse",
        "ai2d",
        "iam",
        "infographic_vqa",
        # "localized_narratives",  # bad dataset
        "intergps",
        "hateful_memes",
        "clevr",
        "iconqa",
        "multihiertt",
        "mapqa",
        "datikz",
        # "okvqa", # bad dataset
        "hitab",
        "chart2text",
        # "ocrvqa",  # bad dataset
        # "clevr_math", # bad dataset
        # "nlvr2",  # bad dataset
        "cocoqa",
        "docvqa",
        "dvqa",
    ]
    # fix select_data
    if select_data == "all":
        tmp_data = all_data_names
    elif select_data in all_data_names:
        tmp_data = [select_data]
    else:
        raise ValueError(f"cannot find dataset: {select_data}")

    data_list = []
    for data_name in tmp_data:
        try:
            data_list.append(
                datasets.load_dataset("data/the_cauldron", data_name)["train"]
            )
        except:
            print(f"bad dataset:{data_name}")
    raw_data = datasets.concatenate_datasets(data_list)
    raw_data = raw_data.train_test_split(
        64, shuffle=True, seed=training_args.data_seed
    )  # 预留64条用于训练中测试，仅仅使用64条是因为减少测试时间长度
    if select_data == "all":
        raw_data["train"] = raw_data["train"].select(range(60 * 1024))  # 选取60K token
    return raw_data


################
# 冻结模型参数&打印模型可训练参数数
################
def freeze_model(qwen_smvl):
    """
    冻结模型的部分参数，只保留特定层可训练。
    
    该函数冻结模型的文本编码器和视觉编码器参数，只保留连接器层和输出层可训练。
    
    Args:
        qwen_smvl (AutoModelForImageTextToText): 需要冻结参数的模型实例。
    
    Returns:
        AutoModelForImageTextToText: 参数已冻结的模型实例。
    
    Example:
        >>> model = load_model()
        >>> model = freeze_model(model)
        >>> print_trainable_parameters(model)  # 查看可训练参数数量
    """
    for _, param in qwen_smvl.model.text_model.named_parameters():
        param.requires_grad = False
    for _, param in qwen_smvl.model.vision_model.named_parameters():
        param.requires_grad = False
    # for _, param in qwen_smvl.lm_head.named_parameters():
    #     param.requires_grad = False
    return qwen_smvl


def print_trainable_parameters(model):
    """
    打印模型的可训练参数数量和比例。
    
    Args:
        model (torch.nn.Module): 需要分析的模型实例。
    
    Returns:
        None: 直接打印参数统计信息，不返回值。
    
    Example:
        >>> model = load_model()
        >>> print_trainable_parameters(model)
        trainable params: 12.34M || all params: 123.45M || trainable%: 10.0
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params/(2**20):.2f}M || all params: {all_param/(2**20):.2f}M || trainable%: {100 * trainable_params / all_param}"
    )


################
# 数据处理
################
def data_collate_fix2k(examples, processor, device, max_length=2048):
    """
    数据批处理函数，将原始数据转换为模型输入格式。
    
    该函数处理包含图像和文本的多模态数据，将其转换为模型可接受的输入格式，
    并设置适当的标签用于训练。
    
    Args:
        examples (list): 包含多个样本的列表，每个样本包含图像和文本数据。
        processor (AutoProcessor): 用于处理文本和图像的处理器。
        device (str): 数据将被发送到的设备，如"cuda"或"cpu"。
        max_length (int, optional): 文本序列的最大长度。默认为2048。
    
    Returns:
        dict: 包含模型输入的字典，已转换为指定设备和数据类型。
    
    Example:
        >>> batch = data_collate_fix2k(examples, processor, "cuda")
        >>> model_outputs = model(**batch)
    """
    batch_text = []
    batch_image = []
    for example in examples:
        images = example["images"][:1]  # 只允许一张图，不然显存顶不住
        batch_image.append(images)
        image_num = len(images)
        chat_texts = example["texts"][0]
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}] * image_num
                + [{"type": "text", "text": chat_texts["user"]}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": chat_texts["assistant"]}],
            },
        ]
        text = processor.apply_chat_template(
            messages, enable_thinking=False, add_generation_prompt=False
        )

        batch_text.append(text)

    batch = processor(
        text=batch_text,
        images=batch_image,
        max_length=max_length,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  # 忽略padding token的损失
    labels[labels == processor.image_token_id] = -100  # 忽略图像token的损失
    batch["labels"] = labels
    return batch.to(device, dtype=torch.bfloat16)


################
# 初始化训练参数
################
@dataclass
class MyTrainArgs(TrainingArguments):
    """
    自定义训练参数类，继承自transformers.TrainingArguments。
    
    该类扩展了TrainingArguments，添加了特定于本项目的训练参数配置。
    
    Attributes:
        train_data (str): 训练数据集名称，默认为"cocoqa"。
        seed (int): 随机种子，默认为42。
        data_seed (int): 数据随机种子，默认为42。
        per_device_train_batch_size (int): 每个设备的训练批次大小，默认为1。
        per_device_eval_batch_size (int): 每个设备的评估批次大小，默认为1。
        gradient_accumulation_steps (int): 梯度累积步数，默认为4。
        dataloader_pin_memory (bool): 是否使用数据加载器的pin_memory，默认为False。
        warmup_ratio (float): 学习率预热比例，默认为0.1。
        learning_rate (float): 学习率，默认为1e-4。
        lr_scheduler_type (str): 学习率调度器类型，默认为"cosine"。
        weight_decay (float): 权重衰减，默认为0.01。
        logging_steps (int): 日志记录步数，默认为5。
        evaluation_strategy (str): 评估策略，默认为"steps"。
        eval_steps (int): 评估步数，默认为10。
        save_strategy (str): 保存策略，默认为"steps"。
        save_steps (int): 保存步数，默认为10。
        save_total_limit (int): 保存的检查点总数限制，默认为8。
        optim (str): 优化器类型，默认为"adamw_torch"。
        bf16 (bool): 是否使用bfloat16精度，默认为True。
        output_dir (str): 输出目录，默认为"./model/qwen-smovlm"。
        overwrite_output_dir (bool): 是否覆盖输出目录，默认为True。
        report_to (str): 报告工具，默认为"swanlab"。
        run_name (str): 运行名称，默认为"freeze_except_connector_fulldata"。
        remove_unused_columns (bool): 是否移除未使用的列，默认为False。
        gradient_checkpointing (bool): 是否使用梯度检查点，默认为False。
    """
    # 更新TrainingArguments的参数形式，原本的形式会报错参数不存在
    train_data: str = "cocoqa"
    seed: int = 42
    data_seed: int = 42
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1  # 设定为1防止默认的8eval导致显存占用过大
    gradient_accumulation_steps: int = 4
    dataloader_pin_memory: bool = False
    warmup_ratio: float = 0.1
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    logging_steps: int = 5
    evaluation_strategy: str = "steps"         # 注意这里 eval_strategy 要改为 evaluation_strategy
    eval_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 10
    save_total_limit: int = 8
    optim: str = "adamw_torch"
    bf16: bool = True
    output_dir: str = "./model/qwen-smovlm"
    overwrite_output_dir: bool = True
    report_to: str = "swanlab"
    run_name: str = "freeze_except_connector_fulldata"
    remove_unused_columns: bool = False
    gradient_checkpointing: bool = False


def main(training_args):
    """
    主函数，执行模型训练和推理的完整流程。
    
    该函数包含以下主要步骤：
    1. 初始化模型和处理器
    2. 准备训练数据集
    3. 训练模型
    4. 保存模型
    5. 使用训练好的模型进行推理示例
    
    Args:
        training_args (MyTrainArgs): 训练参数配置。
    
    Returns:
        None
    
    Example:
        >>> args = MyTrainArgs(output_dir="./output", train_data="cocoqa")
        >>> main(args)
    """
    ################
    # 初始化模型&Tokenizer
    ################
    qwen_smvl_processor = load_processor()
    qwen_smvl = load_model(device)
    # 冻结参数
    qwen_smvl = freeze_model(qwen_smvl)
    # 打印可训练参数量
    print_trainable_parameters(qwen_smvl)

    ################
    # 准备训练数据集
    ################
    raw_data = load_mm_data(select_data=training_args.train_data)
    print(f"总数据条数：{raw_data}")

    # data formatting
    collate_fn = partial(
        data_collate_fix2k, processor=qwen_smvl_processor, device=device
    )

    ################
    # 开启训练
    ################
    last_checkpoint = None  # load last checkpoint if available
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        print(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )
    # Init Trainer
    trainer = Trainer(
        model=qwen_smvl,
        args=training_args,
        train_dataset=raw_data["train"],
        eval_dataset=raw_data["test"],
        data_collator=collate_fn,
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)
    qwen_smvl.save_pretrained(training_args.output_dir)

    ################
    # 对单条数据进行推理
    ################
    with torch.no_grad():
        if trainer.state.is_world_process_zero:
            question = "图中有什么动物？"
            messages = [
                {
                    "role": "system",
                    "content": "使用中文回答所有问题。",
                    # "content": "使用中文回答所有问题，在<think>和</think>中写出思考过程，如果没有思考则为<think> </think>",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            texts = qwen_smvl_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=True,
            )
            print("################# 输入文本 #################")
            print(texts)
            # update img path
            images = [[Image.open("./resource/dog.png")]]
            batch = qwen_smvl_processor(
                text=[texts],
                images=images,
                max_length=1024,
                return_tensors="pt",
                padding_side="left",
                padding=True,
            ).to(qwen_smvl.device, dtype=torch.bfloat16)
            generated_ids = qwen_smvl.generate(
                **batch, do_sample=False, max_new_tokens=256
            )
            model_context = qwen_smvl_processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )
            input_ids_len = batch["input_ids"].shape[1]
            generated_texts = qwen_smvl_processor.batch_decode(
                generated_ids[:, input_ids_len:], skip_special_tokens=True
            )
            print("################# 生成文本 #################")
            print(generated_texts[0])

            # 记录推理结果到SwanLab
            table = swanlab.echarts.Table()
            headers = ["输入问题", "模型输出"]
            rows = [[question, generated_texts[0]]]
            table.add(headers, rows)
            swanlab.log(
                {
                    "sample/输入图像": swanlab.Image(images[0][0]),
                    "sample/问题&回复": table,
                    "sample/上下文": swanlab.Text(model_context[0]),
                }
            )


if __name__ == "__main__":
    """
    脚本入口点，解析命令行参数并启动训练流程。
    
    支持两种参数传递方式：
    1. 通过命令行参数直接传递
    2. 通过YAML配置文件传递
    """
    parser = HfArgumentParser(MyTrainArgs)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a yaml file,
        # let's parse it to get our arguments.
        (training_args,) = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1])
        )
    else:
        (training_args,) = parser.parse_args_into_dataclasses()
    # (training_args,) = parser.parse_yaml_file(yaml_file='full_train.yaml')
    main(training_args)
