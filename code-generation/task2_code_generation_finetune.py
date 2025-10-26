"""
train_spoc_codegen.py
Fine-tune a model to translate SPoC pseudo-code → C++ code
Supports: GPT-2, StarCoder Tiny, DialoGPT, or other causal language models
Compatible with low-VRAM GPUs using LoRA fine-tuning

Example usage:
  python task2_code_generation_finetune.py --model_name gpt2  # Default, fast
  python task2_code_generation_finetune.py --model_name bigcode/tiny_starcoder_py  # Code-focused, 164M params
  python task2_code_generation_finetune.py --model_name gpt2-medium  # More capacity
"""

import os, json, glob, torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from evaluate import load
import gradio as gr

# ----------------------------
# 1. PREPROCESS SPOC DATA
# ----------------------------
def preprocess_spoc(spoc_root="spoc/data", sample_ratio=0.005):
    """Load SPoC TSV files (pseudocode <TAB> code format).
    
    Args:
        spoc_root: Path to SPoC data directory
        sample_ratio: Fraction of data to use (0.005 = 0.5%, 1.0 = 100%)
    """
    import csv
    import random
    
    def load_split(split):
        examples = []
        
        # Try multiple possible locations
        possible_paths = [
            os.path.join(spoc_root, f"{split}.tsv"),
            os.path.join(spoc_root, f"{split}/*.tsv"),  # matches train/*.tsv or test/*.tsv
            os.path.join(spoc_root, f"{split}/split/*.tsv"),
            os.path.join(spoc_root, f"train/split/spoc-train-{split}.tsv"),
            os.path.join(spoc_root, f"split/spoc-train-{split}.tsv"),
        ]
        
        # Specific paths for SPoC dataset structure
        if split == "train":
            possible_paths.append(os.path.join(spoc_root, "train/spoc-train.tsv"))
        elif split == "test":
            possible_paths.append(os.path.join(spoc_root, "test/spoc-test*.tsv"))
        
        # If split is "val", also try "eval"
        if split == "val":
            possible_paths.extend([
                os.path.join(spoc_root, "eval.tsv"),
                os.path.join(spoc_root, "train/split/spoc-train-eval.tsv"),
            ])
        
        for pattern in possible_paths:
            files = glob.glob(pattern)
            for f in files:
                try:
                    with open(f, 'r', encoding='utf-8') as tsvfile:
                        reader = csv.reader(tsvfile, delimiter='\t')
                        for row in reader:
                            if len(row) >= 2:
                                pseudo = row[0].strip()
                                code = row[1].strip()
                                # Skip empty or dummy lines
                                if pseudo and code and pseudo != "DUMMY" and code != "DUMMY":
                                    examples.append({"input": pseudo, "output": code})
                except Exception as e:
                    print(f"Warning: Could not read {f}: {e}")
                    continue
        
        # Remove duplicates based on input
        seen = set()
        unique_examples = []
        for ex in examples:
            if ex['input'] not in seen:
                seen.add(ex['input'])
                unique_examples.append(ex)
        
        # Sample the data
        if sample_ratio < 1.0:
            random.seed(42)  # For reproducibility
            sample_size = min(len(unique_examples), max(1, int(len(unique_examples) * sample_ratio)))
            unique_examples = random.sample(unique_examples, sample_size)
        
        return unique_examples
    
    data = {}
    for split in ["train", "val", "test"]:
        exs = load_split(split)
        if exs:
            data[split] = exs
            print(f"{split}: {len(exs)} examples loaded.")
        else:
            print(f"Warning: No {split} data found. Check paths in {spoc_root}/")
    
    if not data:
        raise ValueError(
            f"No data found in {spoc_root}. Expected TSV files with format: pseudocode<TAB>code"
        )
    return data


# ----------------------------
# 2. TOKENIZATION / MODEL INIT
# ----------------------------
def prepare_model(model_name="distilgpt2"):
    import os
    
    # Check if we're loading a saved model
    is_local_saved = os.path.exists(model_name) and os.path.isdir(model_name)
    adapter_config_data = None  # Initialize
    
    if is_local_saved:
        print(f"Loading saved model from {model_name}...")
        # Load tokenizer from saved directory
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load base model from config
        import json
        adapter_config_path = os.path.join(model_name, "adapter_config.json")
        with open(adapter_config_path, "r") as f:
            adapter_config_data = json.load(f)
        base_model_name = adapter_config_data["base_model_name_or_path"]
        
        print(f"Loading base model {base_model_name}...")
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        # Resize embeddings to match the tokenizer (which has the special tokens)
        model.resize_token_embeddings(len(tokenizer))
    else:
        adapter_config_data = None  # Will be created below
        print(f"Initializing new model with base {model_name}...")
        # Load base tokenizer  
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|PSEUDO|>", "<|CODE|>"]})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model (no quantization)
        print("Loading model without quantization...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Resize embeddings for new model
        model.resize_token_embeddings(len(tokenizer))

    # Determine target modules based on model architecture
    # Check the actual model that was loaded
    model_check_name = model_name if not is_local_saved else (adapter_config_data.get("base_model_name_or_path") if adapter_config_data else model_name)
    
    if "gpt2" in model_check_name.lower() or "distilgpt2" in model_check_name.lower() or "dialogpt" in model_check_name.lower():
        target_modules = ["c_attn"]  # GPT-2 family
    elif "codegen" in model_check_name.lower() or "starcoder" in model_check_name.lower():
        target_modules = ["c_attn", "qkv_proj"]  # StarCoder/CodeGen - try both patterns
    else:
        # Default: try common attention patterns
        print(f"Warning: Unknown model architecture for {model_check_name}, using default target modules")
        target_modules = ["c_attn"]
    
    # LoRA config (small for low VRAM, but better for code)
    if is_local_saved:
        # Use adapter config data already loaded
        lora_config = LoraConfig(
            r=adapter_config_data.get("r", 16),  # Increased for better code understanding
            lora_alpha=adapter_config_data.get("lora_alpha", 32),
            target_modules=adapter_config_data.get("target_modules", target_modules),
            lora_dropout=adapter_config_data.get("lora_dropout", 0.05),
            bias=adapter_config_data.get("bias", "none"),
            task_type=TaskType[adapter_config_data.get("task_type", "CAUSAL_LM")]
        )
        model = get_peft_model(model, lora_config)
        
        # Load the adapter weights
        print(f"Loading adapter weights from {model_name}...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_name)
    else:
        lora_config = LoraConfig(
            r=16,  # Increased for better code understanding
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()
    return model, tokenizer


# ----------------------------
# 3. BUILD DATASET
# ----------------------------
def build_dataset(data, tokenizer, max_len=512):
    def tokenize_fn(example):
        # Simplified prompt - model should learn the pattern better
        prompt = f"Pseudo-code:\n{example['input']}\n\nC++ Code:\n"
        full = prompt + example["output"]
        enc = tokenizer(full, truncation=True, padding="max_length", max_length=max_len)
        
        # Create labels where only code part is trained (prompt has -100)
        prompt_tokens = tokenizer(prompt, truncation=True, max_length=max_len)
        p_len = len(prompt_tokens["input_ids"])
        labels = [-100] * p_len + enc["input_ids"][p_len:]
        enc["labels"] = labels
        return enc

    train = Dataset.from_list(data["train"]).map(tokenize_fn, remove_columns=["input", "output"])
    val = Dataset.from_list(data.get("val", data["train"][:500])).map(tokenize_fn, remove_columns=["input", "output"])
    test = Dataset.from_list(data.get("test", data["train"][:500])).map(tokenize_fn, remove_columns=["input", "output"])
    return DatasetDict({"train": train, "validation": val, "test": test})


# ----------------------------
# 4. TRAINING
# ----------------------------
def train_model(model, tokenizer, dataset):
    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    args = TrainingArguments(
        output_dir="out_spoc_codegen",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=5,  # More epochs for better learning
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=False,  # Keep False for stability
        learning_rate=5e-5,  # Lower LR for better convergence
        warmup_steps=100,
        weight_decay=0.01,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator
    )
    trainer.train()
    trainer.save_model("final_spoc_distilgpt2_lora")
    tokenizer.save_pretrained("final_spoc_distilgpt2_lora")
    return trainer


# ----------------------------
# 5. EVALUATION (BLEU + CodeBLEU)
# ----------------------------
def evaluate_model(model, tokenizer, original_data):
    print("Evaluating...")
    model.eval()
    bleu = load("bleu")
    preds, refs = [], []

    # Get test examples from original data
    test_data = original_data.get("test", original_data.get("train", []))[:100]
    
    if not test_data:
        print("No test data available. Skipping evaluation.")
        return
    
    print(f"Evaluating on {len(test_data)} examples...")
    
    for i, ex in enumerate(test_data):  # subset for speed
        if i % 10 == 0:
            print(f"Processing example {i+1}/{len(test_data)}")
        
        prompt = f"Pseudo-code:\n{ex['input']}\n\nC++ Code:\n"
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=256, do_sample=False)
        gen = tokenizer.decode(out[0], skip_special_tokens=False)
        
        # Extract generated code (remove prompt)
        if prompt in gen:
            gen_code = gen.split(prompt)[-1].strip()
        elif "CODE" in gen:
            # Try to extract after CODE token
            parts = gen.split("<|CODE|>")
            gen_code = parts[-1].strip() if len(parts) > 1 else gen.strip()
        else:
            gen_code = gen.strip()
        
        # Ensure we have non-empty code
        if not gen_code:
            gen_code = "// No code generated"
        
        preds.append(gen_code)
        refs.append([ex["output"]])  # BLEU expects references as list of lists
    
    # Check if we have valid predictions and references
    if not preds or not refs:
        print("Error: No predictions or references generated. Skipping BLEU calculation.")
        return
    
    # Filter out empty references and predictions
    valid_indices = [i for i in range(len(refs)) 
                     if refs[i] and refs[i][0].strip() and preds[i].strip()]
    if not valid_indices:
        print("Error: No valid references and predictions. Skipping BLEU calculation.")
        print(f"Sample pred: {preds[0][:50] if preds else 'None'}")
        print(f"Sample ref: {refs[0] if refs else 'None'}")
        return
    
    preds = [preds[i] for i in valid_indices]
    refs = [refs[i] for i in valid_indices]

    print(f"Computing BLEU on {len(preds)} examples...")
    try:
        score = bleu.compute(predictions=preds, references=refs)
        print("BLEU:", score)
    except ZeroDivisionError:
        print("Error: Division by zero in BLEU calculation. This usually means all predictions or references are empty.")
        print(f"Sample predictions (first 3):")
        for i, p in enumerate(preds[:3]):
            print(f"  {i+1}: {p[:100]}")
        print(f"Sample references (first 3):")
        for i, r in enumerate(refs[:3]):
            print(f"  {i+1}: {r[0][:100] if r else 'None'}")
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        print(f"Number of preds: {len(preds)}, refs: {len(refs)}")

    try:
        from CodeBLEU import calc_code_bleu
        cb = calc_code_bleu.get_code_bleu(refs, preds, lang="cpp")
        print("CodeBLEU:", cb)
    except Exception as e:
        print("CodeBLEU skipped:", e)


# ----------------------------
# 6. GRADIO APP
# ----------------------------
def launch_app(model, tokenizer):
    def generate_code(pseudo):
        prompt = f"Pseudo-code:\n{pseudo}\n\nC++ Code:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.3,  # Lower temperature for more focused output
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        gen = tokenizer.decode(out[0], skip_special_tokens=True)
        # Extract just the generated code
        if prompt in gen:
            return gen.split(prompt)[-1].strip()
        return gen.split("C++ Code:")[-1].strip() if "C++ Code:" in gen else gen

    iface = gr.Interface(
        fn=generate_code,
        inputs="text",
        outputs="code",
        title="SPoC Pseudo→C++ Code Generator",
        description="Fine-tuned distilgpt2 on SPoC dataset"
    )
    iface.launch(share=False)


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune DistilGPT-2 on SPoC dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="spoc_data",
        help="Path to SPoC dataset directory (default: spoc_data)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bigcode/tiny_starcoder_py",
        help="Base model to fine-tune. Options: gpt2 (fast, default), bigcode/tiny_starcoder_py (code-focused), gpt2-medium"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only evaluate/launch app (requires trained model)"
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after training"
    )
    parser.add_argument(
        "--skip_app",
        action="store_true",
        help="Skip launching Gradio app"
    )
    args = parser.parse_args()
    
    if not args.skip_training:
        print("Loading and preprocessing SPoC...")
        data = preprocess_spoc(spoc_root=args.data_path)
        print("Loading model/tokenizer...")
        model, tokenizer = prepare_model(model_name=args.model_name)
        print("Building dataset...")
        dataset = build_dataset(data, tokenizer)
        print("Training...")
        trainer = train_model(model, tokenizer, dataset)
        
        if not args.skip_eval:
            print("Evaluating...")
            evaluate_model(model, tokenizer, data)
    else:
        print("Loading trained model...")
        model, tokenizer = prepare_model(model_name="final_spoc_distilgpt2_lora")
        
        # Load data for evaluation/app if needed
        data = None
        if not args.skip_eval or not args.skip_app:
            print("Loading and preprocessing SPoC...")
            data = preprocess_spoc(spoc_root=args.data_path)
        
        # Evaluate if needed
        if not args.skip_eval and data:
            print("Evaluating...")
            evaluate_model(model, tokenizer, data)
    
    if not args.skip_app:
        print("Launching Gradio app...")
        launch_app(model, tokenizer)
    else:
        print("Done!")
