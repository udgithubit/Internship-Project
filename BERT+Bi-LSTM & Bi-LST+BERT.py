import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Configuration
DATASET_PATH = "final_Hindi_Toxic_Dataset.csv"  # <-- Place your dataset here
EPOCHS = 4
BATCH_SIZE = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Data Preparation
def load_data():
    df = pd.read_csv(DATASET_PATH)
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\u0900-\u097F\s।?!,]', '', str(x)))
    df = df[df['label'].isin([0, 1])]
    return df


class HindiDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts  # already a list
        self.labels = labels  # already a list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }


# Model Definitions
class BertBiLSTM(nn.Module):
    """BERT processes → BiLSTM"""

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.bilstm = nn.LSTM(768, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.bilstm(bert_out)
        output = lstm_out[:, -1]
        return torch.sigmoid(self.classifier(output)).squeeze()


class BiLSTMBert(nn.Module):
    """BERT pooled → BiLSTM"""

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bilstm = nn.LSTM(768, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        lstm_in = pooled.unsqueeze(1)  # Add sequence dimension
        lstm_out, _ = self.bilstm(lstm_in)
        return torch.sigmoid(self.classifier(lstm_out[:, -1])).squeeze()


# Training and Evaluation
def train_model(model, train_loader, epochs):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in loop:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())


def evaluate_model(model, loader):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label']
            outputs = model(**inputs)
            preds.extend(torch.round(outputs).cpu().numpy())
            truths.extend(labels.numpy())
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds)
    prec = precision_score(truths, preds)
    recall = recall_score(truths, preds)
    return acc, prec, recall, f1


def sample_predictions(model, df, tokenizer):
    print("\nSample Predictions:")
    sample_texts = df['text'].tolist()[:5]
    dummy_labels = [0] * 5
    sample_dataset = HindiDataset(sample_texts, dummy_labels, tokenizer)
    sample_loader = DataLoader(sample_dataset, batch_size=5)
    model.eval()
    with torch.no_grad():
        for batch in sample_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.round(outputs).cpu().numpy()
            for i, (text, pred) in enumerate(zip(sample_texts, preds)):
                label = "Toxic" if pred == 1.0 else "Non-Toxic"
                print(f"\nText {i + 1}: {text[:50]}...")
                print(f"Predicted: {label} (Score: {outputs[i].item():.4f})")


# Main Flow
def main():
    df = load_data()
    X_train, X_val, y_train, y_val = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    train_loader = DataLoader(HindiDataset(X_train.tolist(), y_train.tolist(), tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(HindiDataset(X_val.tolist(), y_val.tolist(), tokenizer), batch_size=BATCH_SIZE)

    results = {}

    for model_name, model_class in [("1. BERT → Bi-LSTM", BertBiLSTM), ("2. Bi-LSTM → BERT", BiLSTMBert)]:
        print(f"\n{'=' * 50}\nTraining {model_name}...\n{'=' * 50}")
        model = model_class()
        train_model(model, train_loader, EPOCHS)
        acc, prec, recall, f1 = evaluate_model(model, val_loader)
        results[model_name] = {'accuracy': acc, 'precision': prec, 'recall': recall, 'f1': f1}
        sample_predictions(model, df, tokenizer)

    # Final comparison
    print("\n" + "=" * 80)
    print(f"FINAL COMPARISON ON {DATASET_PATH} (Epochs={EPOCHS})")
    print("=" * 80)
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    for name, metric in results.items():
        print(f"{name:<25} {metric['accuracy']:.4f}     {metric['precision']:.4f}     {metric['recall']:.4f}     {metric['f1']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
