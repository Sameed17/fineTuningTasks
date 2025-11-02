import os
import json
import random
import csv
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from evaluate import load
import gradio as gr


# ==================== CONFIGURATION ====================
MODEL_NAME = "bigcode/tiny_starcoder_py"
SAMPLE_RATIO = 0.005
MAX_LENGTH = 256

ENABLE_TRAINING = True
ENABLE_EVALUATION = True
ENABLE_GRADIO_APP = False

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 5
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
MODEL_SAVE_DIR = "final_spoc_distilgpt2_lora"
# ========================================================

def make_prompt(pseudo):
    return f"Pseudo-code:\n{pseudo}\n\nC++ Code:\n"

def extract_code(generated_text, prompt):
    if prompt in generated_text:
        return generated_text.split(prompt)[-1].strip()
    if "CODE" in generated_text:
        parts = generated_text.split("<|CODE|>")
        return parts[-1].strip() if len(parts) > 1 else generated_text.strip()
    return generated_text.split("C++ Code:")[-1].strip() if "C++ Code:" in generated_text else generated_text.strip()

def generate_code(model, tokenizer, pseudo, max_tokens=256, temperature=0.3, do_sample=True):
    prompt = make_prompt(pseudo)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=0.9 if do_sample else None,
        pad_token_id=tokenizer.eos_token_id
    )
    gen = tokenizer.decode(out[0], skip_special_tokens=do_sample)
    return extract_code(gen, prompt) or "// No code generated"

def get_target_modules(model_name):
    name_lower = model_name.lower()
    if any(x in name_lower for x in ["gpt2", "distilgpt2", "dialogpt"]):
        return ["c_attn"]
    if any(x in name_lower for x in ["codegen", "starcoder"]):
        return ["c_attn", "qkv_proj"]
    return ["c_attn"]

def load_tsv_file(filepath):
    examples = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for row in csv.reader(f, delimiter='\t'):
                if len(row) >= 2:
                    pseudo, code = row[0].strip(), row[1].strip()
                    if pseudo and code and pseudo != "DUMMY" and code != "DUMMY":
                        examples.append({"input": pseudo, "output": code})
    except Exception:
        pass
    return examples

def preprocess_spoc(sample_ratio=SAMPLE_RATIO):
    split_files = {
        "train": "spoc_data/train.tsv",
        "val": "spoc_data/eval.tsv",
        "test": "spoc_data/test.tsv"
    }
    
    data = {}
    for split, filepath in split_files.items():
        if not os.path.exists(filepath):
            continue
        examples = load_tsv_file(filepath)
        unique_examples = list({ex['input']: ex for ex in examples}.values())
        
        if sample_ratio < 1.0 and unique_examples:
            random.seed(42)
            sample_size = max(1, int(len(unique_examples) * sample_ratio))
            unique_examples = random.sample(unique_examples, sample_size)
        
        if unique_examples:
            data[split] = unique_examples
            print(f"{split}: {len(unique_examples)} examples loaded from {os.path.basename(filepath)}")
    
    if not data:
        raise ValueError("No data found in spoc_data")
    return data

def prepare_model(model_name=MODEL_NAME):
    is_local_saved = os.path.exists(model_name) and os.path.isdir(model_name)
    
    if is_local_saved:
        print(f"Loading saved model from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        with open(os.path.join(model_name, "adapter_config.json")) as f:
            adapter_config_data = json.load(f)
        base_model_name = adapter_config_data["base_model_name_or_path"]
        model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model.resize_token_embeddings(len(tokenizer))
    else:
        print(f"Initializing new model with base {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|PSEUDO|>", "<|CODE|>"]})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
        adapter_config_data = None
    
    model_check_name = (adapter_config_data.get("base_model_name_or_path") if is_local_saved and adapter_config_data else model_name)
    target_modules = get_target_modules(model_check_name)
    
    lora_config = LoraConfig(
        r=adapter_config_data.get("r", LORA_R) if is_local_saved else LORA_R,
        lora_alpha=adapter_config_data.get("lora_alpha", LORA_ALPHA) if is_local_saved else LORA_ALPHA,
        target_modules=adapter_config_data.get("target_modules", target_modules) if is_local_saved else target_modules,
        lora_dropout=adapter_config_data.get("lora_dropout", LORA_DROPOUT) if is_local_saved else LORA_DROPOUT,
        bias=adapter_config_data.get("bias", "none") if is_local_saved else "none",
        task_type=TaskType[adapter_config_data.get("task_type", "CAUSAL_LM")] if is_local_saved else TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    if is_local_saved:
        print(f"Loading adapter weights from {model_name}...")
        model = PeftModel.from_pretrained(model, model_name)
    
    model.print_trainable_parameters()
    return model, tokenizer

def build_dataset(data, tokenizer, max_len=MAX_LENGTH):
    def tokenize_fn(example):
        prompt = make_prompt(example['input'])
        full = prompt + example["output"]
        enc = tokenizer(full, truncation=True, padding="max_length", max_length=max_len)
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"])
        enc["labels"] = [-100] * prompt_len + enc["input_ids"][prompt_len:]
        return enc

    splits = {
        "train": data["train"],
        "validation": data.get("val", data["train"][:500]),
        "test": data.get("test", data["train"][:500])
    }
    return DatasetDict({k: Dataset.from_list(v).map(tokenize_fn, remove_columns=["input", "output"]) for k, v in splits.items()})

def train_model(model, tokenizer, dataset):
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=".",
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=NUM_EPOCHS,
            logging_steps=20,
            save_strategy="no",
            eval_strategy="no",
            logging_dir=None,
            fp16=False,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            weight_decay=WEIGHT_DECAY,
            remove_unused_columns=False,
            save_total_limit=0,
            report_to=[],
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")
    )
    trainer.train()
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    return trainer

def evaluate_model(model, tokenizer, original_data):
    print("Evaluating...")
    model.eval()
    test_data = original_data.get("test", original_data.get("train", []))[:100]
    
    if not test_data:
        print("No test data available.")
        return
    
    print(f"Evaluating on {len(test_data)} examples...")
    preds, refs = [], []
    
    for i, ex in enumerate(test_data):
        if i % 10 == 0:
            print(f"Processing example {i+1}/{len(test_data)}")
        preds.append(generate_code(model, tokenizer, ex['input'], do_sample=False))
        refs.append([ex["output"]])
    
    valid = [(p, r) for p, r in zip(preds, refs) if p.strip() and r[0].strip()]
    if not valid:
        print("Error: No valid predictions or references.")
        return
    
    preds, refs = zip(*valid)
    
    try:
        score = load("bleu").compute(predictions=list(preds), references=[list(r) for r in refs])
        print("BLEU:", score)
    except Exception as e:
        print(f"Error computing BLEU: {e}")

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
        fn=lambda pseudo: generate_code(model, tokenizer, pseudo),
        inputs="text",
        outputs="code",
        title="SPoC Pseudo→C++ Code Generator",
        description="Fine-tuned model on SPoC dataset"
    )
    iface.launch(share=False)

if __name__ == "__main__":
    if ENABLE_TRAINING:
        print("Loading and preprocessing SPoC...")
        data = preprocess_spoc()
        print("Loading model/tokenizer...")
        model, tokenizer = prepare_model(model_name=MODEL_NAME)
        print("Building dataset...")
        dataset = build_dataset(data, tokenizer)
        print("Training...")
        train_model(model, tokenizer, dataset)
        
        if ENABLE_EVALUATION:
            evaluate_model(model, tokenizer, data)
    else:
        print("Loading trained model...")
        model, tokenizer = prepare_model(model_name=MODEL_SAVE_DIR)
        
        if ENABLE_EVALUATION:
            print("Loading data for evaluation...")
            data = preprocess_spoc()
            evaluate_model(model, tokenizer, data)
    
    if ENABLE_GRADIO_APP:
        print("Launching Gradio app...")
        launch_app(model, tokenizer)
    else:
        print("Done!")
