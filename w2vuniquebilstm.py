import json
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from gensim.models import KeyedVectors

# Load your data (assuming df is your dataframe)
df = pd.read_csv('csvs/sorted_resume_new.csv')

# Load word_index
with open('csvs/dict.json') as json_file:
    word_index = json.load(json_file)

# Load Word2Vec embeddings
word2vec_model = KeyedVectors.load_word2vec_format('word2vec-google-news-300.txt', binary=False)

# Prepare sequences and labels
num_words = len(word_index)
embedding_dim = 300
embedding_matrix = np.zeros((num_words, embedding_dim))

sequences = []
for text in df['Updated_resume']:
    seq = []
    for word in text.split():
        if word in word_index:
            seq.append(word_index[word])
    sequences.append(seq)

# Pad sequences to a fixed length
maxlen = 150  # maximum sequence length
X = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

# Convert labels to categorical
labels = df['Category'].unique()
labels_to_idx = {label: i for i, label in enumerate(labels)}
idx_to_labels = {i: label for label, i in labels_to_idx.items()}
df['CategoryIdx'] = df['Category'].apply(lambda x: labels_to_idx[x])
labels_cat = to_categorical(df['CategoryIdx'])

num_words = min(len(word_index) + 1, len(word2vec_model.key_to_index))
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= num_words or word not in word2vec_model.key_to_index:
        continue
    embedding_vector = word2vec_model[word]
    embedding_matrix[i] = embedding_vector

num_labels = len(labels)

# Split data into train and test (70:30 ratio)
X_train, X_test, y_train, y_test = train_test_split(X, labels_cat, test_size=0.3, random_state=42, stratify=df['CategoryIdx'])

# Initialize StratifiedKFold
num_folds = 3
stratified_kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Lists to store accuracy scores for each fold
accuracy_scores = []
macro_f1_scores = []
weighted_f1_scores = []

# Iterate over the splits
for fold, (train_idx, val_idx) in enumerate(stratified_kfold.split(X_train, np.argmax(y_train, axis=1))):
    # Create a new instance of the model for each fold
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
    model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(units=num_labels, activation='softmax'))

    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    # Compile and train model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_fold_train, y_fold_train, validation_data=(X_fold_val, y_fold_val), epochs=10, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_fold_val, y_fold_val)
    print(f'Fold {fold + 1} - Validation accuracy: {accuracy}')
    accuracy_scores.append(accuracy)

    # Predict labels for validation set
    y_pred = model.predict(X_fold_val)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_val_labels = np.argmax(y_fold_val, axis=1)

    # Calculate macro F1 and weighted F1
    macro_f1_fold = f1_score(y_val_labels, y_pred_labels, average='macro')
    weighted_f1_fold = f1_score(y_val_labels, y_pred_labels, average='weighted')

    macro_f1_scores.append(macro_f1_fold)
    weighted_f1_scores.append(weighted_f1_fold)

    print(f"Fold {fold + 1} - Macro F1: {macro_f1_fold}, Weighted F1: {weighted_f1_fold}")

# Calculate and print the average accuracy across all folds
average_accuracy = np.mean(accuracy_scores)
average_macro_f1 = np.mean(macro_f1_scores)
average_weighted_f1 = np.mean(weighted_f1_scores)
print(f"\nAverage Validation Accuracy: {average_accuracy}")
print(f"Average Macro F1: {average_macro_f1}")
print(f"Average Weighted F1: {average_weighted_f1}")

# Now, train on the entire training set for the final model
final_model = Sequential()
final_model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=True))
final_model.add(Bidirectional(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2)))
final_model.add(Dense(units=num_labels, activation='softmax'))

final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
final_model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate on the test set
test_loss, test_accuracy = final_model.evaluate(X_test, y_test)
print(f"\nFinal Test Accuracy: {test_accuracy}")

# Predict labels for test set
y_test_pred = final_model.predict(X_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Calculate macro F1 and weighted F1 for the test set
final_macro_f1 = f1_score(y_test_labels, y_test_pred_labels, average='macro')
final_weighted_f1 = f1_score(y_test_labels, y_test_pred_labels, average='weighted')

print(f"Final Macro F1 on Test Set: {final_macro_f1}")
print(f"Final Weighted F1 on Test Set: {final_weighted_f1}")

# Save results to a text file
with open('w2vuniquebilstm.txt', 'w') as file:
    file.write(f"Average Validation Accuracy: {average_accuracy}\n")
    file.write(f"Average Macro F1: {average_macro_f1}\n")
    file.write(f"Average Weighted F1: {average_weighted_f1}\n")
    file.write(f"Final Test Accuracy: {test_accuracy}\n")
    file.write(f"Final Macro F1 on Test Set: {final_macro_f1}\n")
    file.write(f"Final Weighted F1 on Test Set: {final_weighted_f1}\n")
