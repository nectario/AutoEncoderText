import numpy as np
import tensorflow as tf

import autokeras as ak
from GenrePredictionModel import *

def imdb_raw():
    max_features = 20000
    index_offset = 3  # word index offset

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=max_features,
        index_from=index_offset)
    x_train = x_train
    y_train = y_train.reshape(-1, 1)
    x_test = x_test
    y_test = y_test.reshape(-1, 1)

    word_to_id = tf.keras.datasets.imdb.get_word_index()
    word_to_id = {k: (v + index_offset) for k, v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2

    id_to_word = {value: key for key, value in word_to_id.items()}
    x_train = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_train))
    x_test = list(map(lambda sentence: ' '.join(
        id_to_word[i] for i in sentence), x_test))
    x_train = np.array(x_train, dtype=np.str)
    x_test = np.array(x_test, dtype=np.str)
    return (x_train, y_train), (x_test, y_test)


def genre_prediction():
    vectorizer = MultiVectorizer()
    genre_prediction = GenrePredictionModel(vectorizer=vectorizer)

    training_data_df, validation_data_df = genre_prediction.load_data("data/film_data_lots.xlsx", no_sentences=True)

    clf = ak.TextClassifier(max_trials=4, multi_label=True)

    X_train = np.array(training_data_df["Subtitles"].tolist())
    y_train = genre_prediction.training_labels

    X_validation = np.array(validation_data_df["Subtitles"].tolist())
    y_validation = genre_prediction.validation_labels

    clf.fit(X_train, y_train, validation_data=(X_validation, y_validation))



genre_prediction()

# # Prepare the data.
# (x_train, y_train), (x_test, y_test) = imdb_raw()
# print(x_train.shape)  # (25000,)
# print(y_train.shape)  # (25000, 1)
# print(x_train[0][:50])  # <START> this film was just brilliant casting <UNK>
#
# # Initialize the TextClassifier
# clf = ak.TextClassifier(max_trials=3)
# # Search for the best model.
# clf.fit(x_train, y_train, epochs=2)
# # Evaluate on the testing data.
# print('Accuracy: {accuracy}')