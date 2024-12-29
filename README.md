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


