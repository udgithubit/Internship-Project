import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ------------------ Step 1: Load and Clean Dataset ------------------

def clean_hindi_text(text):
    text = re.sub(r'[^\u0900-\u097F0-9\s।?!,;:()-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Load CSV file (must be in same folder)
df = pd.read_csv("hindi_datasheet.csv")

# Drop missing values and clean text
df.dropna(subset=["text", "label"], inplace=True)
df["text"] = df["text"].astype(str).apply(clean_hindi_text)
df["label"] = df["label"].astype(int)

texts = df["text"].tolist()
labels = df["label"].tolist()

print(f"✅ Loaded {len(df)} samples from dataset")


# ------------------ Step 2: Tokenize and Pad ------------------

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

maxlen = int(np.percentile([len(s) for s in sequences], 95))  # or set manually
vocab_size = len(tokenizer.word_index) + 1

X = pad_sequences(sequences, maxlen=maxlen, padding='post')
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"🧠 Vocabulary size: {vocab_size}")
print(f"📏 Sequence length (maxlen): {maxlen}")


# ------------------ Step 3: Build and Train LSTM Model ------------------

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=16, input_length=maxlen),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

# ✅ Avoid unbuilt model error
model.build(input_shape=(None, maxlen))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))


# ------------------ Step 4: Evaluate with All Metrics ------------------

# Predict on test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Final Evaluation Metrics:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# ------------------ Step 5: Show Sample Predictions ------------------

# Recover test texts (since you tokenized earlier)
test_indices = X_test[:10]  # first 10 samples (you can randomize if you want)
original_texts = df.iloc[y_test[:10].index]['text'].values if hasattr(y_test, 'index') else df.iloc[:10]['text'].values

print("\n🔍 Sample Predictions:\n")
for i in range(10):
    text = original_texts[i]
    prob = y_pred_prob[i][0]
    pred_label = int(prob > 0.5)
    true_label = y_test[i]

    print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"✅ True Label   : {'Toxic' if true_label else 'Non-Toxic'}")
    print(f"🤖 Predicted    : {'Toxic' if pred_label else 'Non-Toxic'} (Confidence: {prob:.4f})")
    print(f"🔢 Length       : {len(text)} chars\n{'-'*60}")

