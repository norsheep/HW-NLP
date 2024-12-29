import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from transformers import TrainingArguments, HfArgumentParser, Trainer, AutoTokenizer, AutoModelForCausalLM
from transformers.integrations import TensorBoardCallback
import sys
import torch
import datasets

SYSTEM_PROMPT = "You are an intelligent assistant capable of answering various types of questions based on context. Carefully read each instruction and use logical reasoning to determine the best answer. Provide your answer after considering relevant input."

# Define the arguments required for the main program.
# NOTE: You can customize any arguments you need to pass in.
@dataclass
class ModelArguments:
    """Arguments for model
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the LLM to fine-tune or its name on the Hugging Face Hub."
        }
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype."
            ),
            "choices": ["bfloat16", "float16", "float32"],
        },
    )
    # TODO: add your model arguments here
    model_name_or_path = "Qwen2.5-0.5B"  
    torch_dtype = "float32"

@dataclass
class DataArguments:
    """Arguments for data
    """
    dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the fine-tuning dataset or its name on the Hugging Face Hub."
        }
    )
    # TODO: add your data arguments here
    dataset_path = "alpaca-cleaned/alpaca_data_cleaned.json"
    # 该数据集格式为 list[dict],每个dict包含instruction,input,output

# The main function
# NOTE You can customize some logs to monitor your program.
def finetune():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # 解析器
    # 返回模型参数，数据参数，训练参数
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    dtype = torch.float16 if model_args.torch_dtype == "float16" else(
        torch.bfloat16 if model_args.torch_dtype == "bfloat16" else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)


    dataset = datasets.load_dataset('json', data_files=data_args.dataset_path)
    # 新增，同时修改了
    #train_test_split = dataset['train'].train_test_split(test_size=0.2)
    

    def data_collator(batch: List[Dict]):
        """
        batch: list of dict, each dict of the list is a sample in the dataset.
        一个列表，每个元素是一个字典，字典是数据集中的一个样本
        """
        inputs = []
        labels = []
        max_length = 0

        for sample in batch:
            instruction = sample.get("instruction","")
            input_text = sample.get("input","")
            output_text = sample.get("output","")

            # 检查input和output是否为空
            if not input_text.strip():
                input_text = "<empty>"

            if not output_text.strip():
                output_text = "<empty>"
            
            # 构建输入序列
            input_ids = tokenizer.encode_plus(
                f"{SYSTEM_PROMPT}instruction:{instruction}input:{input_text}", # 将instruction和input_text进行拼接，生成文本输入
                return_tensors = "pt", # 输出转换为pytorch的张量格式
                max_length = tokenizer.model_max_length, # 如果输入序列超过最大长度则截断
                truncation = True, 
                padding = False, # 即使输入序列没有达到最大长度，也不进行填充
            ).input_ids # 用于获取tokenizer返回字典中的‘input_ids’字段

            # 构建输出序列
            output_ids = tokenizer.encode_plus(
                f"output:{output_text}",
                return_tensors = "pt",
                max_length = tokenizer.model_max_length,
                truncation = True,
                padding = False,
            ).input_ids

            # 拼接输入与输出序列，获得模型所需的input_ids
            full_input = torch.cat([input_ids, output_ids], dim=1)
            inputs.append(full_input)

            
            # 创建标签张量 loss1
            labels_tensor = torch.full_like(full_input, -100) # 用-100填充表示这些位置在损失计算中被忽略
            labels_tensor[:, input_ids.shape[1]:] = full_input[:, input_ids.shape[1]:] # 将output_ids对应的位置替换为output_ids原来的值，input_ids对应的位置仍为-100，表示学习时只学习output部分
            labels.append(labels_tensor)
            

            # 创建标签张量 loss2
            # labels_tensor = full_input.clone() # 复制一个full_input张量
            # labels.append(labels_tensor)

            if full_input.shape[1] > max_length:
                max_length = full_input.shape[1]


        # 处理batch的padding，将同一个batch的label和input都填充到同一个长度
        # inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
        # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        inputs = [torch.nn.functional.pad(input, (0,max_length - input.size(1)),value=tokenizer.pad_token_id) for input in inputs]
        labels = [torch.nn.functional.pad(label, (0, max_length - label.size(1)), value=-100) for label in labels]

        # inputs = torch.stack(inputs)
        # labels = torch.stack(labels)

        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)

        return {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": (inputs != tokenizer.pad_token_id).to(dtype=torch.int), 
            # pad_token_id是tokenizer定义的填充令牌ID，也就是对padding的部分填充一个特殊的令牌
            # 该行代码将生成一个与inputs张量形状相同的布尔张量，其中值为True:表示对应的输入ID不是填充ID（即该令牌是有效的；值为False: 表示对应的输入ID是填充ID（即该令牌是无效的，需要被忽略）。
        }

  
    trainer = Trainer(
        args = training_args,
        model=model,  # Pretrained model
        tokenizer=tokenizer,
        data_collator=data_collator,
        # train_dataset=train_test_split['train'],
        # eval_dataset = train_test_split['test'],
        train_dataset=dataset['train'],
        callbacks=[TensorBoardCallback()]
    )

    trainer.train()


sys.argv = [
    "notebook", 
    "--output_dir", "./our-model/1224promptloss1",
    "--learning_rate","1e-5",
    "--num_train_epochs", "3",  # 通常3-5个epoch即可收敛，长时间训练可能会过拟合
    "--per_device_train_batch_size", "8",  # 每个GPU上的训练批量大小
    # "--per_device_eval_batch_size", "16",  # 每个GPU上的评估批量大小
    "--overwrite_output_dir","True",  #开发过程中覆盖旧的文件
    "--save_strategy", "steps",  # 每间隔step保存模型
    "--save_steps", "500",   # 每1000步保存一次
    "--save_total_limit", "2",  # 只保留最好的三个模型
    # "--evaluation_strategy", "steps",  # 每个step结束时进行验证
    # "--eval_steps", "500",  # 每1000步进行一次验证
    # "--load_best_model_at_end", "True",  # 在训练结束时加载最佳模型
    # "--metric_for_best_model", "accuracy",  # 以accuracy作为最佳模型的评估指标
    # "--greater_is_better", "True",  # 指定评估指标越大越好（对于accuracy）
    "--logging_steps","50",
    "--logging_dir", "./logs/1224promptloss1",
    "--remove_unused_columns","False",
    "--dataloader_drop_last", "True",
    '--seed','42',
    "--fp16","True",  # 开启混合精度加速
     "--lr_scheduler_type", "linear",  # 使用线性衰减调度器
    # "--local_rank","-1",
]
if __name__ == "__main__":
    import torch.distributed as dist
     # 初始化进程组
    dist.init_process_group(backend='nccl')

    try:
        finetune()
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        torch.cuda.empty_cache()
        # 在程序结束前销毁进程组
        dist.destroy_process_group()
# python -m torch.distributed.launch --nproc_per_node=6 finetune_eval.py

'''
python opencompass/run.py --datasets mmlu_ppl hellaswag_clean_ppl winogrande_ll ARC_e_ppl ARC_c_clean_ppl SuperGLUE_BoolQ_few_shot_ppl --summarizer example --hf-type base --hf-path "./our-model/1224promptloss1/checkpoint-3234" --tokenizer-kwargs padding_side="left" truncation="left"  --max-seq-len 2048 --batch-size 8 --hf-num-gpus 6 --work-dir "outputs/evals/sft" --debug
'''