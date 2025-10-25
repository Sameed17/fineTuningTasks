import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
import re
import os
from typing import Dict, List

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)



class SentimentPreprocessor:
    """
    Handles preprocessing and tokenization of customer feedback data for sentiment analysis.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df['text'] = df['text'].astype(str).str.replace('"', '').str.strip()
        df['sentiment'] = df['sentiment'].astype(str).str.replace('"', '').str.strip()
        df['label'] = df['sentiment'].map(self.label_map)
        
        df = df.dropna(subset=['text', 'label'])
        df['label'] = df['label'].astype(int)
        df = df[df['text'].str.len() > 0]
        df = df.drop_duplicates(subset=['text'])
        
        return df[['text', 'sentiment', 'label']]
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': tokenized['input_ids'].tolist(),
            'attention_mask': tokenized['attention_mask'].tolist(),
            'labels': examples['label']
        }
    
    def create_datasets(self, df: pd.DataFrame, train_ratio: float = 0.8, 
                       val_ratio: float = 0.1, test_ratio: float = 0.1) -> DatasetDict:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n_samples = len(df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        train_tokenized = train_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=16,
            remove_columns=train_dataset.column_names
        )
        
        val_tokenized = val_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=16,
            remove_columns=val_dataset.column_names
        )
        
        test_tokenized = test_dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=16,
            remove_columns=test_dataset.column_names
        )
        
        train_tokenized.set_format(type='torch')
        val_tokenized.set_format(type='torch')
        test_tokenized.set_format(type='torch')
        
        return DatasetDict({
            'train': train_tokenized,
            'validation': val_tokenized,
            'test': test_tokenized
        })

class SentimentTrainer:
    """
    Handles training and validation of BERT-based sentiment analysis models.
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def load_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def load_trained_model(self, model_path: str = "models/sentiment_bert"):
        """Load a previously trained model from the specified path."""
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        # Create a minimal trainer for evaluation purposes
        training_args = TrainingArguments(
            output_dir=model_path,
            per_device_eval_batch_size=16,
            seed=42,
            report_to=None,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        print(f"Loaded trained model from {model_path}")
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
        f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
    
    def train(self, train_dataset, val_dataset, output_dir: str = "models/sentiment_bert",
              num_epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5,
              warmup_steps: int = 100, early_stopping_patience: int = 2):
        if self.model is None:
            self.load_model()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir=str(output_path / "logs"),
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=2,
            seed=42,
            report_to=None,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=0.0
        )
        self.trainer.add_callback(early_stopping)
        
        train_result = self.trainer.train()
        
        self.trainer.save_model()
        self.tokenizer.save_pretrained(str(output_path))
        
        metrics = train_result.metrics
        with open(output_path / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        return train_result
    
    def evaluate(self, test_dataset) -> Dict:
        if self.trainer is None:
            raise ValueError("Model must be trained first")
        
        predictions = self.trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        accuracy = accuracy_score(true_labels, pred_labels)
        f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
        
        class_names = ['Negative', 'Neutral', 'Positive']
        report = classification_report(
            true_labels, 
            pred_labels, 
            target_names=class_names,
            output_dict=True
        )
        
        cm = confusion_matrix(true_labels, pred_labels)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': pred_labels.tolist(),
            'true_labels': true_labels.tolist()
        }
        
        return metrics

        
class SentimentPredictor:
    """
    Handles predictions using trained sentiment analysis models.
    """
    
    def __init__(self, model_path: str = "models/sentiment_bert"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.class_names = ['Negative', 'Neutral', 'Positive']
        self.label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    
    def load_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.eval()
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        return inputs
    
    def predict_single(self, text: str) -> Dict:
        if self.model is None:
            self.load_model()
        
        inputs = self.preprocess_text(text)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
        
        confidence = probabilities[0][predicted_class].item()
        
        class_probabilities = {
            self.class_names[i]: probabilities[0][i].item() 
            for i in range(len(self.class_names))
        }
        
        result = {
            'text': text,
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'class_probabilities': class_probabilities
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        if self.model is None:
            self.load_model()
        
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        
        return results

def main():
    train_new_model = False
    # Data Preprocessing
    preprocessor = SentimentPreprocessor(model_name="bert-base-uncased", max_length=128)
    csv_path = os.path.join(os.path.dirname(__file__), "sentiment-analysis-cleaned.csv")
    df = preprocessor.load_and_preprocess_data(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Sentiment distribution:")
    print(df['sentiment'].value_counts())
    
    # Create datasets
    datasets = preprocessor.create_datasets(df)
    print(f"Dataset splits: Train={len(datasets['train'])}, Val={len(datasets['validation'])}, Test={len(datasets['test'])}")
    
    # Model Training
    trainer = SentimentTrainer()
    
    if train_new_model:
        print("Training new model...")
        train_result = trainer.train(
            train_dataset=datasets['train'],
            val_dataset=datasets['validation'],
            num_epochs=5,
            batch_size=4,
            learning_rate=2e-5,
            early_stopping_patience=3
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
    # Example Predictions
    predictor = SentimentPredictor()
    if not train_new_model:
        predictor.load_model()
    
    example_texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is terrible. I'm very disappointed with the quality.",
        "The product is okay, nothing special but it works.",
        "Excellent customer service! They were very helpful and responsive.",
        "Poor quality materials, not worth the money at all."
    ]
    
    predictions = predictor.predict_batch(example_texts)
    
    print("\nPREDICTION RESULTS:")
    for i, pred in enumerate(predictions, 1):
        print(f"{i}. {pred['text']}")
        print(f"   Sentiment: {pred['predicted_class'].upper()} (Confidence: {pred['confidence']:.2%})")
        print()

if __name__ == "__main__":
    main()
