import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

import gensim

# Step 1: Load Dataset
df = pd.read_csv("final_Hindi_Toxic_Dataset.csv")


# Step 2: Clean Hindi Text
def clean_text(text):
    text = re.sub(r'[^\u0900-\u097F\s]', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['text'] = df['text'].apply(clean_text)

# Step 3: Filter Labels
df = df.dropna(subset=['label'])
df = df[df['label'].isin([0, 1])]
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
maxlen = 100
X = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
y = np.array(labels)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Load FastText Embeddings
print("Loading FastText embeddings...")
try:
    fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(
        "cc.hi.300.vec",
        encoding='utf-8'
    )
    print("FastText model loaded successfully!")
except Exception as e:
    print(f"Error loading FastText model: {e}")
    exit()

embedding_dim = 300
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if word in fasttext_model:
        embedding_matrix[i] = fasttext_model[word]
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))


# Step 9: Build and Initialize Model
def build_model():
    model = Sequential([
        Embedding(vocab_size, embedding_dim,
                  weights=[embedding_matrix],
                  input_length=maxlen,
                  trainable=False),

        Bidirectional(LSTM(128, return_sequences=True,
                           kernel_regularizer=l2(0.01))),
        Dropout(0.5),
        Bidirectional(LSTM(64, kernel_regularizer=l2(0.01))),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    return model


model = build_model()

# Explicitly build the model by passing a sample input
sample_input = np.zeros((1, maxlen), dtype='int32')
_ = model(sample_input)  # This builds all layers

model.summary()

# Step 10: Train with Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
]

print("\nTraining Bi-LSTM model with FastText embeddings...")
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Step 11: Evaluate
print("\nEvaluating model...")
y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Toxic', 'Toxic'], zero_division=0))

model.save('final_fasttext_bilstm_model.h5')
print("Model saved successfully.")