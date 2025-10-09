import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from gensim.models import Word2Vec

# Step 1: Load Dataset
df = pd.read_csv("final_Hindi_Toxic_Dataset.csv")  # must have 'text' and 'label' columns


# Step 2: Clean Hindi Text
def clean_text(text):
    text = re.sub(r'[^\u0900-\u097F\s]', '', str(text))  # Keep only Hindi characters and whitespace
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text


df['text'] = df['text'].apply(clean_text)

# Step 3: Filter Valid Labels
df = df.dropna(subset=['label'])
df = df[df['label'].isin([0, 1])]  # Ensure only binary labels 0 and 1
df['label'] = df['label'].astype(int)

# Step 4: Tokenize + Prepare Sentences for Word2Vec
texts = df['text'].tolist()
labels = df['label'].tolist()
sentences = [text.split() for text in texts]  # Tokenize into words for Word2Vec

# Step 5: Train Word2Vec on your data
print("Training Word2Vec model...")
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
embedding_dim = 100

# Step 6: Tokenizer for Keras
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1

# Step 7: Build Embedding Matrix from Word2Vec
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
    else:
        # Random initialization for unknown words
        embedding_matrix[i] = np.random.normal(0, 1, embedding_dim)

# Step 8: Pad Sequences
maxlen = 100  # Maximum sequence length
X = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
y = np.array(labels)

# Step 9: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 10: Build and Compile Bi-LSTM Model
def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],
                        input_length=maxlen,
                        trainable=False))

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# Build and initialize model
model = build_model()

# Explicitly build the model by calling it with sample input
sample_input = np.zeros((1, maxlen), dtype='int32')
_ = model(sample_input)  # This builds the model
model.summary()  # Now all layers should show parameters

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Step 11: Train
print("\nTraining Bi-LSTM model...")
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=64,
                    validation_split=0.1,
                    callbacks=[early_stop])

# Step 12: Evaluate
print("\nEvaluating model...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

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

# Save model if needed
# model.save('word2vec_bilstm_toxicity_model.h5')