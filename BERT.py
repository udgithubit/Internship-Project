import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Import Hugging Face Transformers for BERT
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf

# Step 1: Load the CSV file
df = pd.read_csv(r"C:\Users\HP\PycharmProjects\pythonProject1\hindi_datasheet.csv")


# Step 2: Clean the text column
def clean_hindi_text(text):
    text = re.sub(r'[^\u0900-\u097F\s।?!,]', '', str(text))  # Keep Hindi chars and basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['text'] = df['text'].astype(str).apply(clean_hindi_text)

# Step 3: Clean and validate labels
df = df.dropna(subset=['label'])
df = df[df['label'].isin([0, 1])]
df['label'] = df['label'].astype(int)

# Step 4: Prepare text and labels
texts = df['text'].tolist()
labels = df['label'].tolist()

# Step 5: Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


# Step 6: Tokenize and prepare input for BERT
def convert_to_bert_input(texts, labels, max_length=128):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    return {
        'input_ids': np.array(input_ids),
        'attention_mask': np.array(attention_masks)
    }, np.array(labels)


X, y = convert_to_bert_input(texts, labels)

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X['input_ids'], y, test_size=0.2, random_state=42, stratify=y
)
train_mask, test_mask = train_test_split(
    X['attention_mask'], test_size=0.2, random_state=42
)


# Step 8: Build BERT model
def build_bert_model():
    model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-cased',
        num_labels=2
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[metric]
    )

    return model


model = build_bert_model()
model.summary()

# Step 9: Train the model
history = model.fit(
    [X_train, train_mask], y_train,
    validation_split=0.1,
    epochs=3,  # BERT typically needs fewer epochs
    batch_size=16  # Smaller batch size due to BERT's memory requirements
)

# Step 10: Evaluate on test set
logits = model.predict([X_test, test_mask])[0]
y_pred = np.argmax(logits, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\n📊 Accuracy: {accuracy:.4f}")
print(f"📊 Precision: {precision:.4f}")
print(f"📊 Recall: {recall:.4f}")
print(f"📊 F1 Score: {f1:.4f}")
print("\n🧾 Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# Save the model
model.save_pretrained("hindi_bert_toxicity")
tokenizer.save_pretrained("hindi_bert_toxicity")