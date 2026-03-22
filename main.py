import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset

ds = load_dataset("PyThaiNLP/prachathai67k", trust_remote_code=True)
df_train = ds["train"].to_pandas()

# label columns
LABEL_COLS = ["politics","human_rights","quality_of_life","international",
              "social","environment","economics","culture",
              "labor","national_security","ict","education"]

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
NUM_LABELS = len(LABEL_COLS)

f1 = evaluate.load("f1")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification")

def read_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def preprocess(text_dataset: pd.DataFrame) -> dict:
    tokenized = tokenizer(
        text_dataset["body_text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128) 
    ### มันจะเอาส่วนที่เป็นข้อความยาวๆ หรือเนื้อหาจากข่าวมาแปลงเป็น Tokens ([3, 412, 887, ..., 1, 1, 1]) ผลลัพธ์คือ 
    # input: ["รัฐบาลประกาศ...", "นักเรียนออกมา..."]
    # output tokenized:
    # {
    #   "input_ids":      [[3, 412, 887, ..., 1, 1, 1],   # 128 ตัวเลข
    #                      [3, 991, 23,  ..., 1, 1, 1]],
    #   "attention_mask": [[1, 1,   1,   ..., 0, 0, 0],   # 1=real, 0=padding
    #                      [1, 1,   1,   ..., 0, 0, 0]]
    # }
    tokenized["labels"] = [
        [float(text_dataset[col][i]) for col in LABEL_COLS]
        for i in range(len(text_dataset["body_text"]))
    ] #เพิ่ม Label ให้กับข้อมูลที่ถูกแปลงเป็น Tokens ไว้ก่อนหน้านี้ ตัวอย่าง
    # input: 
    # {
    #   "input_ids":      [[3, 412, 887, ..., 1, 1, 1],   # 128 ตัวเลข
    #                      [3, 991, 23,  ..., 1, 1, 1]],
    #   "attention_mask": [[1, 1,   1,   ..., 0, 0, 0],   # 1=real, 0=padding
    #                      [1, 1,   1,   ..., 0, 0, 0]]
    # }
    # Label:
    # {
    #   "input_ids":      [[3, 412, 887, ..., 1, 1, 1],   # 128 ตัวเลข
    #                      [3, 991, 23,  ..., 1, 1, 1]],
    #   "attention_mask": [[1, 1,   1,   ..., 0, 0, 0],   # 1=real, 0=padding
    #                      [1, 1,   1,   ..., 0, 0, 0]],
    #  "labels":        [[0.0, 1.0, 0.0, ..., 0.0],     # 12 ตัวเลข (0 หรือ 1)
    #                    [1.0, 0.0, 0.0, ..., 0.0]]
    # }
    return tokenized

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits > 0).astype(int)
    return f1.compute(
        predictions=preds.flatten(), 
        references=labels.flatten(), 
        average="micro")

if __name__ == "__main__":
    ds = ds.map(preprocess, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    # tensor([3, 412, 887, 23, 2, 1, 1, ...])  ← torch.Tensor
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=2e-5,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("./results")