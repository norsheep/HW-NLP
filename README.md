### 模型和知识库下载链接（包括两次大作业）
jbox：https://jbox.sjtu.edu.cn/l/81DMsf

### 项目内容
CS3602课程学习大作业一，对Qwen0.5B进行有监督微调，使用opencompass评测。

### 部署步骤
1. 创建conda环境

2. pip install -r requirements.txt

3. 从git上下载opencompass源文件，pip install -e .

### 目录描述
- alpaca-cleaned：存放数据集
- logs：训练时tensorboard所用，注意更改子目录名
- our-model：存放训练得到的模型
- outputs：存放评测结果
- finetune_eval.py：最终训练所使用的版本
- finetune_eval.ipynb：脚本的notebook版本（未经对齐）

### 训练
在项目目录下，执行命令。
```bash
python finetune_eval.py
```
若显存不足，可使用以下命令开启DDP并行训练方式。
```bash
python -m torch.distributed.launch --nproc_per_node=<your cpu numbers> finetune_eval.py
```

### 评测
```bash
python opencompass/run.py --datasets mmlu_ppl hellaswag_clean_ppl winogrande_ll ARC_e_ppl ARC_c_clean_ppl SuperGLUE_BoolQ_few_shot_ppl --summarizer example --hf-type base --hf-path "<your model path>" --tokenizer-kwargs padding_side="left" truncation="left"  --max-seq-len 2048 --batch-size 8 --hf-num-gpus 6 --work-dir "result save-path" --debug
```

### 实验
1. 对于训练中使用不同batchsize的探究：更改finetune_eval.py中sys.argv的--per_device_train_batch_size
2. 对于不同loss计算方式对模型SFT结果的影响：更改finetune_eval.py的data_collator函数

    1. 全Loss
    ```python
    # 创建标签张量 
    labels_tensor = full_input.clone() # 复制一个full_input张量
    labels.append(labels_tensor)
    ```

    2. 仅output loss
    ```python
    # 创建标签张量 
    labels_tensor = torch.full_like(full_input, -100) 
    labels_tensor[:, input_ids.shape[1]:] = full_input[:, input_ids.shape[1]:] 
    labels.append(labels_tensor)
    ```
3. 探究SYSTEM_PROMPT的作用：若不加，仅需去掉finetune_eval.py中,data_calltor中,input_ids的SYSTEM_PROMPT



### 项目内容
CS3602课程学习大作业二

### 完成情况
bonus1、2、3均完成，包括搭建聊天机器人实现多轮对话；利用lora对Qwen2.5-3b和Qwen2.5-1.5b微调；搭建外部知识库实现检索；实现虚拟人并拥有长程记忆；以及对上述全部任务的详尽测评。

### 实验平台
1. lora微调在www.autodl.com算力平台上实现，租用的服务器配置为六张L20GPU，每个GPU显存为48G
2. 聊天机器人相关的全部代码可在kaggle平台上运行并测试，包括多轮对话、RAG、虚拟人等实现。

### 目录新增文件
- chat_robot.py：lora微调模型代码（bonus1）
- ChatRobot_LoRA.ipynb：lora微调模型代码(notebook格式)
- ChatRobot_LoRA.ipynb：聊天机器人多轮对话实现代码（base）
- ChatRobot_LoRA.ipynb：基于外部知识库检索的聊天机器人实现（bonus2）
- ChatRobot_LoRA.ipynb：虚拟人实现（bonus3）
- ChatRobot_LongMemory.ipynb：长程记忆实现（bonus3）

### 各文件运行方式

### lora模型微调
在autodl-tmp目录下创建models文件夹，存放未经过指令微调的原模型(qwen2.5-transformers-1.5b-v1/qwen2.5-transformers-3b-v1)；创建output文件夹，存放经过lora微调后的模型，即用于后续聊天机器人部署的基底模型；在autodl-tmp目录下上传train.csv文件，即alpaca-cleaned数据集；上传chat_robot.py文件，即实际训练代码。
在项目目录下，执行命令：
```
python -m torch.distributed.launch --nproc_per_node=6(GPU数量) chat_robot.py
```

### 聊天机器人启动
将ChatRobot.ipynb上传至kaggle平台，将微调好的模型(lora-1.5b-transformers-default-v1/lora-3b-transformers-default-v1)以及未经过微调的原模型(qwen2.5-transformers-1.5b-v1/qwen2.5-transformers-3b-v1)上传至kaggle/input/MODELS，依次运行单元格，即可与聊天机器人进行多轮对话。

### 外部知识库检索
将ChatRobot_RAG.ipynb上传至kaggle平台，将微调好的模型以及未经过微调的原模型上传至kaggle/input/MODELS，将文档库(documents)上传至kaggle/input/DATASETS，依次运行单元格，即可与聊天机器人进行基于外部知识库检索的多轮对话。

### 虚拟人实现
将ChatRobot_VirtualHuman.ipynb上传至kaggle平台，将微调好的模型以及未经过微调的原模型上传至kaggle/input/MODELS，将文档库上传至kaggle/input/DATASETS，依次运行单元格，即可使得聊天机器人以虚拟身份进行多轮对话。

### 长程记忆启动
将ChatRobot_LongMemory.ipynb上传至kaggle平台，将微调好的模型以及未经过微调的原模型上传至kaggle/input/MODELS，将文档库上传至kaggle/input/DATASETS，依次运行单元格，即可使得聊天机器人以虚拟身份进行多轮对话，同时拥有长程记忆，kaggle/working/chat_history.txt即为超出最大序列长度的对话历史知识库。


