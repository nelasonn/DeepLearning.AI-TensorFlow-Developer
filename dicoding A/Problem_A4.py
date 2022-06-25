# ==========================================================================================================
# PROBLEM A4
#
# Build and train a binary classifier for the IMDB review dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in http://ai.stanford.edu/~amaas/data/sentiment/
#
# Desired accuracy and validation_accuracy > 83%
# ===========================================================================================================

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_A4():
    imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    # YOUR CODE HERE
    train_data, test_data = imdb['train'], imdb['test']
    training_sentences = []
    training_labels = []
    testing_sentences = []
    testing_labels = []

    for s, l in train_data:
        training_sentences.append(s.numpy().decode('utf8'))
        training_labels.append(l.numpy())

    for s, l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())
#    YOUR CODE HERE
    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    oov_tok = "<OOV>"


    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok) # YOUR CODE HERE
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(training_sentences)
    test_sequences = tokenizer.texts_to_sequences(testing_sentences)
    train_padded_seq = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating=trunc_type)
    test_padded_seq = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating=trunc_type)

    labels = np.concatenate((training_labels, testing_labels), axis=None)
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    train_seq = label_tokenizer.texts_to_sequences(training_labels)
    test_seq = label_tokenizer.texts_to_sequences(testing_labels)
    train_label_seq = np.array(train_seq)-1
    test_label_seq = np.array(test_seq)-1

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    model = tf.keras.Sequential([
        # YOUR CODE HERE. Do not change the last layer.
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GRU(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(train_padded_seq, training_labels_final, epochs=3, validation_data=(test_padded_seq, testing_labels_final))

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_A4()
    model.save("model_A4.h5")

