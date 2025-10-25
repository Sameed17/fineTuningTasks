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
MODEL_NAME = "t5-base"  # or "facebook/bart-base"
OUTPUT_DIR = "models/t5_summarization"
MAX_INPUT_LENGTH = 128 
MAX_TARGET_LENGTH = 64
# MAX_INPUT_LENGTH = 512
# MAX_TARGET_LENGTH = 128
TRAIN_BATCH = 4
EVAL_BATCH = 4
NUM_EPOCHS = 3
LR = 5e-5
SEED = 42
# Dataset sampling - reduce dataset size to save memory
DATASET_SAMPLE_SIZE = 0.0001  # Use 50% of the dataset (0.5 = 50%, 0.25 = 25%, etc.)
# For even more memory savings, try: 0.25 (25%), 0.1 (10%), or 0.05 (5%)

def preprocess_examples(examples, tokenizer):
    # examples: dict with 'article' and 'highlights' (summary)
    inputs = ["summarize: " + a.strip() for a in examples["article"]] if "t5" in MODEL_NAME else [a.strip() for a in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")

    # tokenize targets using the new recommended approach
    labels = tokenizer(text_target=examples["highlights"], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_local_kaggle_csv(path):
    # expects CSV or JSON with 'article' and 'summary' columns
    ds = load_dataset("csv", data_files={"train": path})
    return ds

def sample_dataset(dataset, sample_size):
    """
    Sample a subset of the dataset to reduce memory usage.
    
    Args:
        dataset: DatasetDict with train/validation/test splits
        sample_size: Float between 0 and 1, fraction of data to keep
    
    Returns:
        Sampled DatasetDict
    """
    if sample_size >= 1.0:
        return dataset
    
    sampled = {}
    for split_name, split_data in dataset.items():
        # Calculate number of samples to keep
        num_samples = int(len(split_data) * sample_size)
        
        # Sample the data
        sampled[split_name] = split_data.shuffle(seed=SEED).select(range(num_samples))
        print(f"Sampled {len(sampled[split_name])} examples from {split_name} (originally {len(split_data)})")
    
    return DatasetDict(sampled)

def compute_metrics(eval_pred):
    rouge = evaluate.load("rouge")
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # decode labels: replace -100 with pad_token_id then decode
    labels = torch.where(torch.tensor(labels) != -100, torch.tensor(labels), tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE expects newline-separated sentences sometimes; basic call works
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # trim/format
    result = {k: round(v, 4) for k, v in result.items()}
    # optionally compute length using torch operations
    preds_tensor = torch.tensor(preds)
    prediction_lens = torch.sum(preds_tensor != tokenizer.pad_token_id, dim=1)
    result["gen_len"] = int(torch.mean(prediction_lens.float()).item())
    return result

if __name__ == "__main__":
    
    model_name = MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Load dataset example: replace with path to your CSV/JSON files (train/val/test)
    # Option A: load from local csv files where columns are article and summary:
    # dataset = load_dataset("csv", data_files={"train":"data/train.csv","validation":"data/val.csv","test":"data/test.csv"})
    # Option B: if you have the Kaggle CNN/DailyMail prepared as JSON/CSV, load accordingly.
    dataset = load_dataset("csv", data_files={"train":"data/train.csv", "validation":"data/validation.csv", "test":"data/test.csv"})
    
    # Sample the dataset to reduce memory usage
    print(f"Original dataset sizes:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")
    
    dataset = sample_dataset(dataset, DATASET_SAMPLE_SIZE)
    
    print(f"Sampled dataset sizes:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")

    # Make sure the dataset has 'article' and 'highlights' names; otherwise rename columns accordingly.
    # Tokenize with memory-efficient settings
    tokenized = {}
    for split in dataset:
        tokenized[split] = dataset[split].map(
            lambda ex: preprocess_examples(ex, tokenizer),
            batched=True,
            batch_size=1000,  # Process in smaller batches
            remove_columns=dataset[split].column_names,
            desc=f"Tokenizing {split}",
        )
        # Alternative 1: Convert to numpy arrays first, then to torch
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
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Enable FP16 only if GPU is available
        dataloader_pin_memory=torch.cuda.is_available(),  # Pin memory only for GPU
        dataloader_num_workers=0,     # Use single worker to save memory
        gradient_accumulation_steps=4, # Accumulate gradients to simulate larger batch
        seed=SEED,
        report_to=None,
        # Additional memory optimizations
        remove_unused_columns=True,
        dataloader_drop_last=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    # Evaluate on test set with generation (ROUGE)
    metrics = trainer.evaluate(tokenized["test"])
    print("Test ROUGE:", metrics)

    # Example generation (qualitative comparison) - optimized for memory
    sample_articles = dataset["test"]["article"][:4]  # Reduced from 8 to 4
    inputs = tokenizer(sample_articles, max_length=MAX_INPUT_LENGTH, truncation=True, return_tensors="pt", padding=True).to(model.device)
    generated = model.generate(
        **inputs,
        max_length=MAX_TARGET_LENGTH,
        num_beams=2,  # Reduced from 4 to save memory
        early_stopping=True,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        do_sample=False,  # Disable sampling to save memory
    )
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    
    # Clear GPU memory
    del inputs, generated
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    for i, (art, gen) in enumerate(zip(sample_articles, decoded), 1):
        print(f"\n=== ARTICLE {i} ===\n{art[:400]}...\n--- GENERATED SUMMARY ---\n{gen}\n")
