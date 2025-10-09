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


def load_and_validate_data(filepath):
    """Load and validate the dataset"""
    df = pd.read_csv(filepath)

    if not all(col in df.columns for col in ['text', 'label']):
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    print(f"\nDataset contains {len(df)} samples")
    print("Class distribution:")
    print(df['label'].value_counts())

    duplicate_texts = df['text'].duplicated().sum()
    print(f"\nFound {duplicate_texts} duplicate texts")
    if duplicate_texts > 0:
        df = df.drop_duplicates(subset=['text'])
        print(f"Removed duplicates, {len(df)} samples remaining")

    return df


def clean_hindi_text(text):
    """Clean Hindi text by removing special characters"""
    text = str(text)
    text = re.sub(r'[^\u0900-\u097F\s।?!,]', '', text)  # Keep only Hindi chars and basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text


def prepare_data(df):
    """Clean and prepare the data"""
    df['text'] = df['text'].apply(clean_hindi_text)
    df = df.dropna(subset=['label'])
    valid_labels = df['label'].isin([0, 1])
    print(f"\nRemoving {len(df) - valid_labels.sum()} samples with invalid labels")
    df = df[valid_labels]
    df['label'] = df['label'].astype(int)
    return df


class HindiToxicDataset(Dataset):
    """Custom PyTorch Dataset for Hindi toxic text"""

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


class BertBiLSTMToxic(nn.Module):
    """BERT + BiLSTM model for toxic text classification"""

    def __init__(self, n_classes=1, dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')

        # Freeze first 6 BERT layers
        for param in list(self.bert.parameters())[:6]:
            param.requires_grad = False

        self.bilstm = nn.LSTM(
            input_size=768,  # BERT hidden size
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if 2 > 1 else 0
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(512, n_classes)  # 256*2 for bidirectional

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = bert_output.last_hidden_state
        lstm_output, _ = self.bilstm(sequence_output)

        # Concatenate last hidden states from forward and backward LSTM
        last_hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=1)

        output = self.dropout(last_hidden)
        return torch.sigmoid(self.classifier(output)).squeeze()


def evaluate_model(model, data_loader, device):
    """Evaluate model and return all metrics"""
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

    # Calculate all metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    cm = confusion_matrix(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=['Non-Toxic', 'Toxic'], zero_division=0)

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


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load and prepare data
    df = load_and_validate_data("hindi_datasheet.csv")
    df = prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label'].values
    )

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertBiLSTMToxic().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCELoss()
    scaler = GradScaler()

    # Create data loaders
    train_dataset = HindiToxicDataset(X_train, y_train, tokenizer)
    test_dataset = HindiToxicDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Training loop
    for epoch in range(4):  # 4 epochs
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
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

        # Evaluation
        print(f"\nEpoch {epoch + 1} Training Loss: {train_loss / len(train_loader):.4f}")
        metrics = evaluate_model(model, test_loader, device)

        # Memory management
        torch.cuda.empty_cache()
        gc.collect()

    # Final evaluation
    print("\nFinal Evaluation Results:")
    final_metrics = evaluate_model(model, test_loader, device)

    # Sample predictions
    print("\nSample Predictions:")
    test_texts = X_test[:5]  # Show first 5 test samples
    test_dataset = HindiToxicDataset(test_texts, [0] * 5, tokenizer)  # Dummy labels
    test_loader = DataLoader(test_dataset, batch_size=5)

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.round(outputs).cpu().numpy()

            for i, (text, pred) in enumerate(zip(test_texts, preds)):
                print(f"\nText {i + 1}: {text[:50]}...")
                print(f"Predicted: {'Toxic' if pred > 0.5 else 'Non-Toxic'} (score: {pred:.4f})")


if __name__ == "__main__":
    main()