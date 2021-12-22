import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding
from transformers import get_scheduler
import torch
from accelerate import Accelerator
from transformers import Trainer


def tokenize_function(example):
    '''
    example可以是一个样本，也是一个一批样本

    注意根据指定的最大长度填充是非常低效的，最好的方法是按照该批次内最大长度进行填充。
    '''
    return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding=True, max_length=512)




if __name__ == '__main__':
    device = torch.device(1)
    print(torch.cuda.current_device())

    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    raw_datasets = load_dataset('csv', data_files={'train': ['./data/train.csv'], 'dev': './data/dev.csv'})
    # datasets.map 可以分批加载数据，而不是一次加载整个数据集
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    print("start")
    # 此时不再需要这些原始字符串序列了
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["sentence1", "sentence2"]
    )

    # 模型期望的命名是labels
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # 设置datasets返回PyTorch Tensor而不是列表
    tokenized_datasets.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=128, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["dev"], shuffle=True, batch_size=128, collate_fn=data_collator
    )

    accelerator = Accelerator()

    model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=2)
    # 指定优化器
    # optimizer = AdamW(model.parameters(), lr=5e-5)

    model.to(device)

    # 多GPU
    # train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    #    train_dataloader, eval_dataloader, model, optimizer)

    num_epochs = 3
    # num_training_steps = num_epochs * len(train_dataloader)
    # 学习率scheduler
    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps
    # )
    # print(f"num_training_steps:{num_training_steps}")

    training_args = TrainingArguments("trainer", evaluation_strategy="epoch", per_device_train_batch_size=16,
                                      save_steps=1000)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["eval"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # 显示进度条
    # progress_bar = tqdm(range(num_training_steps))
    #
    # # 运行之前，在命令行敲入 accelerate config，回答一些问题
    #
    # for epoch in range(num_epochs):
    #     model.train()
    #     for batch in train_dataloader:
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         # accelerator.backward(loss)
    #         loss.backward()
    #
    #         optimizer.step()
    #         lr_scheduler.step()
    #         optimizer.zero_grad()
    #         progress_bar.update(1)
    #
    #     model.eval()
    #     for batch in eval_dataloader:
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         with torch.no_grad():
    #             outputs = model(**batch)
    #         predictions = outputs.logits.argmax(dim=-1)
    #         metric.add_batch(predictions=predictions, references=batch["labels"])
    #     eval_metric = metric.compute()
    #     # Use accelerator.print to print only on the main process.
    #     accelerator.print(f"epoch {epoch}:", eval_metric)
