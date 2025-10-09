import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from gensim.models import Word2Vec

# Load dataset
df = pd.read_csv("final_Hindi_Toxic_Dataset.csv")  # Replace with your dataset path

# Clean Hindi text
def clean_text(text):
    text = re.sub(r'[^\u0900-\u097F\s]', '', str(text))  # Keep Hindi chars + spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

# Filter valid labels (0 or 1)
df = df.dropna(subset=['label'])
df = df[df['label'].isin([0, 1])]
df['label'] = df['label'].astype(int)

# Tokenize and pad sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
maxlen = 100  # Or calculate dynamically
X = pad_sequences(sequences, maxlen=maxlen)
y = np.array(df['label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Word2Vec
tokenized_texts = [text.split() for text in df['text']]
w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=10, min_count=1, workers=4)

# Build embedding matrix
embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# ====== KEY FIX: Build Model Properly ======
model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=maxlen,
        trainable=False
    ),
    LSTM(128),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Explicitly build the model by specifying input shape
model.build(input_shape=(None, maxlen))  # This ensures all layers are built

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()  # Now shows all parameters correctly

# Train
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# Evaluate
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, zero_division=0))