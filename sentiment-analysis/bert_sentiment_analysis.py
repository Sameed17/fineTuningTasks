import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, List
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)
class SentimentPreprocessor:    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df['text'] = df['text'].astype(str).str.replace('"', '').str.strip()
        df['sentiment'] = df['sentiment'].astype(str).str.replace('"', '').str.strip()
        df['label'] = df['sentiment'].map(self.label_map)
        
        df = df.dropna(subset=['text', 'label'])
        df['label'] = df['label'].astype(int)
        df = df[df['text'].str.len() > 0].drop_duplicates(subset=['text'])
        
        return df[['text', 'sentiment', 'label']]
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=self.max_length
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': examples['label']
        }
    
    def create_datasets(self, df: pd.DataFrame, train_ratio: float = 0.8, 
                       val_ratio: float = 0.1) -> DatasetDict:
        df = df.sample(frac=1).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': df[:train_end],
            'validation': df[train_end:val_end],
            'test': df[val_end:]
        }
        
        datasets = {}
        for name, split_df in splits.items():
            dataset = Dataset.from_pandas(split_df)
            tokenized = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            tokenized.set_format(type='torch')
            datasets[name] = tokenized
        
        return DatasetDict(datasets)


class SentimentTrainer:
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def load_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def load_trained_model(self, model_path: str = "models/sentiment_bert"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        training_args = TrainingArguments(
            output_dir=model_path,
            per_device_eval_batch_size=16,
            report_to=None
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
            'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0)
        }
    
    def train(self, train_dataset, val_dataset, output_dir: str = "models/sentiment_bert",
              num_epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        if self.model is None:
            self.load_model()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            save_total_limit=1
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        train_result = self.trainer.train()
        self.trainer.save_model()
        self.tokenizer.save_pretrained(str(output_path))        
        return train_result
    
    def evaluate(self, test_dataset) -> Dict:
        if self.trainer is None:
            raise ValueError("Model must be trained first")
        
        predictions = self.trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'f1_macro': f1_score(true_labels, pred_labels, average='macro', zero_division=0),
            'f1_weighted': f1_score(true_labels, pred_labels, average='weighted', zero_division=0),
            'classification_report': classification_report(
                true_labels, pred_labels, 
                target_names=['Negative', 'Neutral', 'Positive'],
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(true_labels, pred_labels).tolist()
        }
        
        return metrics

def main():
    train_new_model = True
    
    preprocessor = SentimentPreprocessor()
    df = preprocessor.load_and_preprocess_data("sentiment-analysis-cleaned.csv")
    datasets = preprocessor.create_datasets(df)
    
    trainer = SentimentTrainer()
    
    if train_new_model:
        print("Training new model...")
        trainer.train(
            train_dataset=datasets['train'],
            val_dataset=datasets['validation'],
            num_epochs=5,
            batch_size=4,
            learning_rate=2e-5,
        )
        print("Training completed!")
    else:
        print("Loading existing trained model...")
        trainer.load_trained_model("models/sentiment_bert")

    # Model Evaluation
    test_metrics = trainer.evaluate(datasets['test'])
    
    print("EVALUATION RESULTS:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1-Score (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {test_metrics['f1_weighted']:.4f}")

if __name__ == "__main__":
    main()