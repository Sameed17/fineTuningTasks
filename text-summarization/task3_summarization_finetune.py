import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import evaluate

# CONFIG
MODEL_NAME = "t5-base"
OUTPUT_DIR = "models/t5_summarization"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 64
TRAIN_BATCH = 4
EVAL_BATCH = 4
NUM_EPOCHS = 3
LR = 5e-5
DATASET_SAMPLE_SIZE = 0.0001

def preprocess_examples(examples, tokenizer):
    inputs = ["summarize: " + article.strip() for article in examples["article"]]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        text_target=examples["highlights"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def sample_dataset(dataset, sample_size):
    if sample_size >= 1.0:
        return dataset
    
    sampled = {}
    for split_name, split_data in dataset.items():
        num_samples = int(len(split_data) * sample_size)
        sampled[split_name] = split_data.shuffle().select(range(num_samples))
    return DatasetDict(sampled)

def compute_metrics(eval_pred):
    rouge = evaluate.load("rouge")
    preds, labels = eval_pred
    
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 with pad_token_id for labels
    labels_tensor = torch.tensor(labels)
    labels_tensor = torch.where(
        labels_tensor != -100,
        labels_tensor,
        tokenizer.pad_token_id
    )
    decoded_labels = tokenizer.batch_decode(labels_tensor, skip_special_tokens=True)
    
    # Compute ROUGE scores
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )
    
    # Round and format results
    result = {k: round(v, 4) for k, v in result.items()}
    
    # Compute average generation length
    preds_tensor = torch.tensor(preds)
    prediction_lens = torch.sum(preds_tensor != tokenizer.pad_token_id, dim=1)
    result["gen_len"] = int(torch.mean(prediction_lens.float()).item())
    return result

if __name__ == "__main__":
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    dataset = load_dataset(
        "csv",
        data_files={
            "train": "data/train.csv",
            "validation": "data/validation.csv",
            "test": "data/test.csv"
        }
    )
    dataset = sample_dataset(dataset, DATASET_SAMPLE_SIZE)
    tokenized = {}
    for split in dataset:
        tokenized[split] = dataset[split].map(
            lambda ex: preprocess_examples(ex, tokenizer),
            batched=True,
            batch_size=1000,
            remove_columns=dataset[split].column_names,
        )
        tokenized[split] = tokenized[split].map(
            lambda x: {
                "input_ids": torch.tensor(x["input_ids"]),
                "attention_mask": torch.tensor(x["attention_mask"]),
                "labels": torch.tensor(x["labels"])
            },
            batched=False
        )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        predict_with_generate=True,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        logging_steps=100,
        save_total_limit=1,
        gradient_accumulation_steps=4,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    metrics = trainer.evaluate(tokenized["test"])
    print("Test ROUGE:", metrics)

    sample_articles = dataset["test"]["article"][:4]
    inputs = tokenizer(
        sample_articles,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    
    generated = model.generate(
        **inputs,
        max_length=MAX_TARGET_LENGTH,
        num_beams=2,
        early_stopping=True,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        do_sample=False,
    )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    
    for i, (article, summary) in enumerate(zip(sample_articles, decoded), 1):
        print(f"\n=== ARTICLE {i} ===\n{article[:400]}...")
        print(f"--- GENERATED SUMMARY ---\n{summary}\n")
