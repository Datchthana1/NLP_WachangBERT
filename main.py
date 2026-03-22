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

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification")

def read_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def preprocess(text_dataset: pd.DataFrame) -> dict:
    tokenized = tokenizer(text_dataset["body_text"], truncation=True, padding="max_length", max_length=128)
    tokenized["labels"] = [
        [float(text_dataset[col][i]) for col in LABEL_COLS]
        for i in range(len(text_dataset["body_text"]))
    ]
    return tokenized

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits > 0).astype(int)
    return f1.compute(predictions=preds.flatten(), references=labels.flatten(), average="micro")

if __name__ == "__main__":
    ds = ds.map(preprocess, batched=True)
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

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