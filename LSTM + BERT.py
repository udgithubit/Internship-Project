import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             confusion_matrix)
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import gc
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_and_preprocess_data(filepath="hindi_dataset.csv"):
    """Load and preprocess the Hindi toxic text dataset"""
    if not os.path.exists(filepath):
        available_files = [f for f in os.listdir() if f.endswith('.csv')]
        raise FileNotFoundError(
            f"Dataset file '{filepath}' not found. Available CSV files: {available_files}"
        )

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

    # Validate required columns
    required_columns = ['text', 'label']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Existing columns: {df.columns.tolist()}")

    # Clean Hindi text with robust error handling
    def clean_text(text):
        try:
            if not isinstance(text, str):
                text = str(text)
            # Keep only Hindi characters and basic punctuation
            text = re.sub(r'[^\u0900-\u097F\s।?!,.]', '', text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text if text else "[EMPTY]"
        except Exception as e:
            print(f"Error cleaning text: {str(e)}")
            return "[ERROR]"

    print("Cleaning text data...")
    df['text'] = df['text'].apply(clean_text)

    # Filter and validate labels
    print("\nOriginal label distribution:")
    print(df['label'].value_counts(dropna=False))

    df = df.dropna(subset=['label'])
    valid_labels = df['label'].isin([0, 1])

    print(f"\nRemoving {len(df) - valid_labels.sum()} samples with invalid labels")
    df = df[valid_labels].copy()
    df['label'] = df['label'].astype(int)

    print("\nFinal dataset stats:")
    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    print(df['label'].value_counts(normalize=True))

    return df


class HindiToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class LSTMBertToxic(nn.Module):
    def __init__(self, bert_model_name='bert-base-multilingual-cased',
                 lstm_hidden_dim=256, lstm_layers=1, dropout=0.3):
        super().__init__()

        # BERT model (frozen)
        self.bert = BertModel.from_pretrained(bert_model_name)
        for param in self.bert.parameters():
            param.requires_grad = False

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        pooled_output = bert_output.pooler_output
        lstm_input = pooled_output.unsqueeze(1)
        lstm_output, _ = self.lstm(lstm_input)
        last_hidden = lstm_output[:, -1, :]
        output = self.dropout(last_hidden)
        return torch.sigmoid(self.classifier(output)).squeeze()


def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.round(outputs).cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions,
                                   target_names=['Non-Toxic', 'Toxic'],
                                   zero_division=0)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'report': report
    }


def train_model(model, train_loader, val_loader, optimizer, criterion, scaler, device, epochs=4):
    best_f1 = 0
    history = {'train_loss': [], 'val_accuracy': [], 'val_f1': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float()

            optimizer.zero_grad()

            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'best_lstm_bert_model.pth')
            print(f"\nSaved new best model with F1: {best_f1:.4f}")

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation F1: {val_metrics['f1']:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    return history


def main():
    try:
        # Load data
        print("Loading dataset...")
        df = load_and_preprocess_data("hindi_datasheet.csv")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            df['text'].values,
            df['label'].values,
            test_size=0.2,
            random_state=42,
            stratify=df['label'].values
        )

        # Initialize tokenizer
        print("\nInitializing tokenizer...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        # Create datasets
        print("Creating data loaders...")
        train_dataset = HindiToxicDataset(X_train, y_train, tokenizer)
        val_dataset = HindiToxicDataset(X_val, y_val, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        # Initialize model
        print("\nInitializing LSTM + BERT model...")
        model = LSTMBertToxic().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-5)
        criterion = nn.BCELoss()
        scaler = GradScaler()

        # Train
        print("\nStarting training...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epochs=4
        )

        # Final evaluation
        print("\nLoading best model for final evaluation...")
        model.load_state_dict(torch.load('best_lstm_bert_model.pth'))
        print("\nFinal Evaluation Results:")
        final_metrics = evaluate_model(model, val_loader, device)

        # Sample predictions
        print("\nSample Predictions:")
        sample_texts = X_val[:5]
        sample_labels = y_val[:5]
        sample_dataset = HindiToxicDataset(sample_texts, sample_labels, tokenizer)
        sample_loader = DataLoader(sample_dataset, batch_size=5)

        model.eval()
        with torch.no_grad():
            for batch in sample_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                preds = torch.round(outputs).cpu().numpy()

                for i, (text, true, pred) in enumerate(zip(sample_texts, labels.cpu().numpy(), preds)):
                    print(f"\nSample {i + 1}:")
                    print(f"Text: {text[:50]}...")
                    print(f"True: {'Toxic' if true > 0.5 else 'Non-Toxic'}")
                    print(f"Pred: {'Toxic' if pred > 0.5 else 'Non-Toxic'} (score: {outputs[i].item():.4f})")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()