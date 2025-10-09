import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, classification_report,
                             confusion_matrix)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, LSTM, Dense,
                                     Dropout, Bidirectional)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def load_data(filepath):
    df = pd.read_csv(filepath)

    if not all(col in df.columns for col in ['text', 'label']):
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    print(f"\nDataset contains {len(df)} samples")
    print("Class distribution:")
    print(df['label'].value_counts())

    return df


def clean_hindi_text(text):
    text = str(text)
    text = re.sub(r'[^\u0900-\u097F\s।?!,०-९]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_bilstm_model(vocab_size, embedding_dim, maxlen):
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  input_length=maxlen,
                  embeddings_regularizer=l2(1e-4)),
        Bidirectional(LSTM(128, return_sequences=True,
                           kernel_regularizer=l2(0.01))),
        Dropout(0.5),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.build(input_shape=(None, maxlen))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    # Load and clean data
    df = load_data("final_Hindi_Toxic_Dataset.csv")
    df['text'] = df['text'].apply(clean_hindi_text)

    # Filter labels
    df = df.dropna(subset=['label'])
    df = df[df['label'].isin([0, 1])]
    df['label'] = df['label'].astype(int)

    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])

    # Dynamic sequence length
    seq_lengths = [len(seq) for seq in sequences]
    maxlen = int(np.percentile(seq_lengths, 95))
    print(f"\nUsing sequence length: {maxlen}")

    X = pad_sequences(sequences, maxlen=maxlen)
    y = np.array(df['label'])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Build model
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 128
    model = build_bilstm_model(vocab_size, embedding_dim, maxlen)
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('best_bilstm_model.h5', save_best_only=True)
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    main()