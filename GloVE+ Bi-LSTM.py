import pandas as pd
import numpy as np
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Step 1: Load Dataset
df = pd.read_csv("final_Hindi_Toxic_Dataset.csv")  # must have 'text' and 'label' columns


# Step 2: Clean Hindi Text
def clean_text(text):
    text = re.sub(r'[^\u0900-\u097F\s]', '', str(text))  # Keep only Hindi characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text


df['text'] = df['text'].apply(clean_text)

# Step 3: Filter Valid Labels
df = df.dropna(subset=['label'])
df = df[df['label'].isin([0, 1])]  # Binary classification
df['label'] = df['label'].astype(int)

# Step 4: Prepare Data
texts = df['text'].tolist()
labels = df['label'].tolist()

# Step 5: Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Step 6: Pad Sequences
maxlen = 100  # Optimal sequence length for most cases
X = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
y = np.array(labels)

# Step 7: Train-Test Split - THIS WAS THE PROBLEM AREA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Load GloVe Embeddings
embedding_dim = 100
embedding_index = {}
glove_file = "glove.6B.100d.txt"  # Replace with Hindi GloVe if available

print("Loading GloVe embeddings...")
try:
    with open(glove_file, encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vector
    print(f"Loaded {len(embedding_index)} word vectors from GloVe.")
except FileNotFoundError:
    print(f"Error: GloVe file '{glove_file}' not found.")
    exit()

# Step 9: Create Embedding Matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if word in embedding_index:
        embedding_matrix[i] = embedding_index[word]
    else:
        # Initialize unknown words with small random numbers
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))


# Step 10: Build and Compile Bi-LSTM Model
def build_model():
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=maxlen,
                  trainable=False),  # Freeze embeddings

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(),
                 tf.keras.metrics.Recall()]
    )
    return model


model = build_model()
model.summary()

# Step 11: Train with Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

print("\nTraining Bi-LSTM model with GloVe embeddings...")
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Step 12: Evaluate
print("\nEvaluating model...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Toxic', 'Toxic'], zero_division=0))

# Save the final model
model.save('glove_bilstm_toxicity_model.h5')
print("Model saved successfully.")