# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================
import numpy
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd


def solution_B4():
    bbc = pd.read_csv(
        'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    sentences = bbc['text']
    labels = bbc['category']

    train_size = len(sentences)
    train_sentences = sentences[0:int(training_portion * train_size)]
    val_sentences = sentences[int(training_portion * train_size):train_size]

    tokenizer =Tokenizer(num_words=vocab_size, oov_token=oov_tok) # YOUR CODE HERE
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded_seq = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=padding_type)
    val_sequences = tokenizer.texts_to_sequences(val_sentences)
    val_padded_seq = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=padding_type)

    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    labels = label_tokenizer.texts_to_sequences(labels)
    labels = np.array(labels) - 1
    train_labels, val_labels = labels[0:int(training_portion * train_size)], labels[int(training_portion * train_size):train_size]

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(train_padded_seq, train_labels, epochs=10, validation_data=(val_padded_seq, val_labels))

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
