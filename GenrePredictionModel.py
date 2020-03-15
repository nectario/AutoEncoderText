import os
from collections import Counter
from operator import itemgetter

import tensorflow as tf
from common.MultiVectorizer import *
import pandas as pd
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM, TimeDistributed, SpatialDropout1D, Conv1D, MaxPooling1D, Dropout, AdditiveAttention, Attention, \
    GlobalAveragePooling1D, Concatenate, Bidirectional, GlobalMaxPool1D, Reshape, RepeatVector
from tensorflow.keras.models import Model
from common.data_utils import *
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm._tqdm_notebook import tqdm_notebook
from tensorflow.keras.metrics import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from tensorflow.keras.preprocessing.text import one_hot

tqdm_notebook.pandas()

class GenrePredictionModel():
    def __init__(self, vectorizer=None, load_weights=False):
        self.vectorizer = vectorizer
        self.load_weights = load_weights
        self.METRICS = [
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]

        self.checkpoint_path = "data/weights/checkpoints/cp-epoch_{epoch:02d}-accuracy_{accuracy:.3f}_val_precision_{val_precision:.3f}-val_recall_{val_recall:.3f}-val_auc_{val_auc:.3f}.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.latest_weights = tf.train.latest_checkpoint(self.checkpoint_dir)

    def load_data(self, file_path, rows=None, validation_split=None):
        data_df = pd.read_excel(file_path, nrows=rows)
        data_df = data_df[data_df.Exclude == False]
        data_df.replace("vislted", "visited", inplace=True)
        data_df = data_df.query(r"(Overview != 0 and Overview != '' and Overview.notnull()) or (Plot != 0 and Plot != '' and Plot.notnull())")
        data_df.replace(r'[^\x00-\x7F]+',' ', inplace=True, regex=True)
        data_df.replace(0,"", inplace=True)


        if validation_split is not None:
            data_df = data_df[data_df.Training == True].reset_index()
            training_df = data_df[:int(1 - data_df.shape[0]*validation_split)]
            validation_df = data_df[int(1 - data_df.shape[0]*validation_split)+1: - 1]
        else:
            training_df = data_df[data_df.Training == True]
            validation_df = data_df[data_df.Validation == True]

        training_df.loc[:, "Overview"] = training_df["Overview"].apply(lambda x: x+" ") + training_df["Plot"]
        validation_df.loc[:, "Overview"] = validation_df["Overview"] + validation_df["Plot"]

        training_df.loc[:,"Subtitles"] = training_df["Subtitles 1"] + training_df["Subtitles 2"]
        validation_df.loc[:, "Subtitles"] = validation_df["Subtitles 1"] + validation_df["Subtitles 2"]
        training_df.drop(["Subtitles 1", "Subtitles 2"], inplace=True, axis=1)
        validation_df.drop(["Subtitles 1", "Subtitles 2"], inplace=True, axis=1)

        training_df.loc[:,"Labels"] = training_df["Genres"].apply(self.parse_str_labels)
        validation_df.loc[:,"Labels"] = validation_df["Genres"].apply(self.parse_str_labels)

        self.genres = self.load_current_genres(file_path="data/current_genre_reduced.xlsx")

        mlb = MultiLabelBinarizer(classes=self.genres)

        training_binary_labels = mlb.fit_transform(training_df["Labels"])
        validation_binary_labels = mlb.fit_transform(validation_df["Labels"])

        self.training_labels =  training_binary_labels
        self.validation_labels =  validation_binary_labels

        return training_df, validation_df

    def load_current_genres(self, file_path, load_original=False):
        if load_original:
            with open("data/genres.pickle", "rb") as f:
                genres = pickle.load(f)
                return genres
        current_genres_df = pd.read_excel(file_path)
        genres = current_genres_df["Genres"].tolist()
        return genres

    def get_sentences(self, text):
        text = str(text if type(text) == str else "")
        sentences = sent_tokenize(text)
        return sentences

    def parse_str_labels(self, str_labels):
        if type(str_labels) == list:
            return str_labels
        if type(str_labels) != str:
            return []
        labels = list(map(str.strip, str_labels.split(",")))
        return labels

    def to_str(self, label_list):
        return str(label_list).replace("'","").replace("[","").replace("]","")

    def split_string_list(self, genres):
        if type(genres) == str:
            if genres == "":
                return []
            genres_list = genres.split(",")
            genres_list = [i.strip() for i in genres_list]
        else:
            return ""
        return genres_list

    def pad_list_of_lists(self, array, fill_value=0.0, shape=(), debug=False):

        sent_lens = []
        word_lens =  []

        for i, sents in enumerate(array):
            sent_lens.append(len(sents))
            for word in sents:
                word_lens.append(len(word))

        batch_size = len(array)
        max_sents = max(sent_lens)
        max_words = max(word_lens)
        avg_sents = np.mean(sent_lens)
        avg_words = np.mean(word_lens)
        most_common_sents = Counter(sent_lens).most_common(20)
        most_common_words = Counter(word_lens).most_common(20)
        most_common_sents = max(list(zip(*most_common_sents))[0]) + 20
        most_common_words = max(list(zip(*most_common_words))[0]) + 20

        if debug:
            print("Max sentences:", max_sents)
            print("Max words:", max_words)
            print("Avg sentences:", avg_sents)
            print("Avg words:", avg_words)
            print("Most common sentences:", most_common_sents)
            print("Most common words:", most_common_words)

        shape = (batch_size, most_common_sents, most_common_words)
        result = np.full(shape, fill_value)
        for index, value in enumerate(array):
            if index == shape[0]:
                break
            for idx, row in enumerate(value):
                if idx == shape[1]:
                    break
                # result[index: len(value)] = value
                result[index, idx, :len(row) if len(row) < shape[2] else shape[2]] = row[:len(row) if len(row) < shape[2] else shape[2]]
        return result

    def preprocess(self, data, save_encoded_data=False):

        overview_data = self.vectorizer.fit(data["Overview"].apply(self.get_sentences).values)
        subtitles_data = self.vectorizer.fit(data["Subtitles"].apply(self.get_sentences).values)

        #overview_one_hot = one_hot(data["Overview"], vectorizer.get_vocabulary_size(), filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
        #subtitles_one_hot = one_hot(data["Subtitles"], vectorizer.get_vocabulary_size(), filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

        if save_encoded_data:
            self.save_pickle_data(overview_data, file_path="data/overview_encoded_data.dat")
            self.save_pickle_data(overview_data, file_path="data/subtitles_encoded_data.dat")

        return self.pad_list_of_lists(overview_data), self.pad_list_of_lists(subtitles_data)

    def save_pickle_data(self, overview_data, file_path=None):
        if file_path is None:
            raise ValueError("Please specify file path in save_pickle_data()!")

        with open(file_path, "wb") as handle:
            pickle.dump(overview_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle_data(self, file_path):
        with open(file_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            return data

    def get_two_input_model(self, embedding_size=300, lstm_output_size=500, number_of_labels=130):

        if self.load_weights:
            self.vectorizer = self.vectorizer.load("data/weights/vectorizer.dat")

        print("Vocabulary Size:", self.vectorizer.get_vocabulary_size())

        overview_input = Input(shape=(None, None), dtype='int64', name="PlotInput")
        subtitles_input = Input(shape=(None, None), dtype='int64', name="SubtitlesInput")
        sentence_input = Input(shape=(None,), dtype='int64', name="SentenceInput")

        embedded_sentence = Embedding(self.vectorizer.get_vocabulary_size(), embedding_size, trainable=False, name="Embedding")(sentence_input)
        spatial_dropout_sentence = SpatialDropout1D(0.20, name="SpatialDropoutSentence")(embedded_sentence)
        cnn_sentence = Conv1D(220, 4, padding="same", activation="relu", strides=1, name="Conv1DSentence")(spatial_dropout_sentence)
        max_pool_sentence = MaxPooling1D(pool_size=3, name="MaxPooling1DSentence")(cnn_sentence)
        sentence_encoding = Bidirectional(LSTM(lstm_output_size, recurrent_dropout=0.10))(max_pool_sentence)

        #overview_sentence_output, sentence_encoding, subtitles_sentence_output = self.get_autoencoder_layers(lstm_1)

        sentence_model = Model(sentence_input, sentence_encoding)

        segment_time_distributed = TimeDistributed(sentence_model, name="TimeDistributedSegment")
        segment_cnn = Conv1D(number_of_labels, 2, padding="same", activation="relu", name="SegmentConv1D")
        segment_max_pool = MaxPooling1D(pool_size=3, name="SegementMaxPool1D")

        segment_cnn_2 = Conv1D(number_of_labels, 5, padding="same", activation="relu", name="Segment2Conv1D")
        segment_max_pool_2 = MaxPooling1D(pool_size=3, name="Segment2MaxPool1D")

        overview_time_distributed = segment_time_distributed(overview_input)
        overview_cnn = segment_cnn(overview_time_distributed)
        overview_maxpool = segment_max_pool(overview_cnn)

        subtitles_timedistributed = segment_time_distributed(subtitles_input)
        subtitles_cnn = segment_cnn_2(subtitles_timedistributed)
        subtitles_maxpool = segment_max_pool_2(subtitles_cnn)

        overview_dropout = SpatialDropout1D(0.20)(overview_maxpool)
        overview_pre_attention_output = Dense(number_of_labels, name="PlotPreAttnOutput")(overview_dropout)

        subtitles_dropout = SpatialDropout1D(0.20, name="SubtitlesDropout")(subtitles_maxpool)
        subtitles_pre_attention_output = Dense(number_of_labels, name="SubtitlesPreAttnOutput")(subtitles_dropout)

        attention_overview = AdditiveAttention(name="PlotAttention")([overview_pre_attention_output, overview_maxpool])
        attention_subtitles = AdditiveAttention(name="SubtitlesAttention")([subtitles_pre_attention_output, subtitles_maxpool])

        overview_output = GlobalAveragePooling1D(name="GlobalAvgPoolPlot")(attention_overview)
        subtitles_output = GlobalAveragePooling1D(name="GlobalAvgPoolSubitles")(attention_subtitles)

        concat_output = Concatenate(axis=-1, name="OutputConcatenate")([overview_output, subtitles_output])
        dropput = Dropout(0.40)(concat_output)
        output = Dense(number_of_labels, activation="sigmoid", name="Output")(dropput)

        model = Model([overview_input, subtitles_input], output)

        model.compile(loss='binary_crossentropy',
                      optimizer='adamax',
                      metrics=self.METRICS)

        print(sentence_model.summary())
        print(model.summary())
        self.sentence_model = sentence_model
        self.model = model
        if self.load_weights:
            self.sentence_model.load_weights("data/weights/sentence_model.h5")
            self.model.load_weights(self.latest_weights)  #("data/weights/main_model.h5")
        return sentence_model, model

    def get_autoencoder_layers(self, lstm_1):
        sentence_encoding = Dense(1000, activation="relu")(lstm_1)
        dense = Dense(1100, activation="relu")(sentence_encoding)
        dense = Dense(1200, activation="relu")(dense)
        dense = Dense(1300, activation="relu")(dense)
        dense = Dense(1400, activation="relu")(dense)
        dense = RepeatVector(100)(dense)
        overview_sentence_output = TimeDistributed(Dense(100000, activation="softmax"))(dense)
        subtitles_sentence_output = TimeDistributed(Dense(100000, activation="softmax"))(dense)
        return overview_sentence_output, sentence_encoding, subtitles_sentence_output

    def fit(self, data, labels, validation_data=None, validation_labels=None, batch_size=5, epochs=10):
        print("Pre-processing training data...")
        overview_input, subtitles_input = self.preprocess(data)
        print("Pre-processing validation data...")
        overview_validation_input, subtitles_validation_input = self.preprocess(validation_data)
        self.sentence_model, self.model = self.get_two_input_model(number_of_labels=len(self.genres))

        callback_actions = self.CallbackActions(main_model=self.model, sentence_model=self.sentence_model, vectorizer=self.vectorizer)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        self.model.fit([overview_input, subtitles_input], labels, validation_data=([overview_validation_input, subtitles_validation_input], validation_labels), epochs=epochs, callbacks=[callback_actions, cp_callback], batch_size=batch_size)

    def evaluate(self, data=None, binary_labels=None, file_path=None, genre_column="Genre Labels", batch_size=2, output_vectors=False, data_type="", load_encoded_data=False,
                 save_encoded_data=False):
        binary_predictions = None

        self.sentence_model, self.model = self.get_two_input_model(number_of_labels=len(self.genres))

        if file_path is not None:
            predictions_df = pd.read_excel(file_path)
        else:
            binary_predictions, predictions_df = self.predict(data, batch_size=batch_size, output_vectors=output_vectors, data_type=data_type, load_encoded_data=load_encoded_data,
                                       save_encoded_data=save_encoded_data)
        exact_match = []
        num_labels_matching = []
        single_miss = []
        double_miss = []
        num_labels = []
        predicted_num_labels = []
        accuracy = []

        for i, row in predictions_df.iterrows():
            genre_labels = set(self.split_string_list(row[genre_column]))
            genre_predictions = set(self.split_string_list(row["Genre Predictions"]))

            labels_len = len(genre_labels)
            pred_labels_len = len(genre_predictions)

            num_labels.append(labels_len)
            predicted_num_labels.append(pred_labels_len)

            labels_matching = len(genre_labels.intersection(genre_predictions))
            num_labels_matching.append(labels_matching)

            if genre_labels == genre_predictions and len(genre_predictions) > 0:
                exact_match.append(True)
            else:
                exact_match.append(False)

            set_union = genre_labels.union(genre_predictions)
            set_intersection = genre_labels.intersection(genre_predictions)

            accuracy.append(round(len(set_intersection) / (len(set_union) +np.e), 4) * 100 )

            diff = 0
            if len(genre_predictions) > 0 and len(set_intersection) > 0 and (len(genre_labels) != len(genre_predictions)):
                diff = len(genre_labels.difference(set_intersection))
            elif len(genre_predictions) > 0 and len(genre_labels) == len(genre_predictions):
                diff = np.absolute(len(genre_labels) - len(set_intersection))

            if diff == 1:
                single_miss.append(True)
            else:
                single_miss.append(False)

            if diff == 2:
                double_miss.append(True)
            else:
                double_miss.append(False)

        print("Exact Match:", sum(exact_match)/len(exact_match))

        precision = dict()
        recall = dict()
        average_precision = dict()

        try:
            binary_labels = np.array(binary_labels)
            binary_predictions = np.array(binary_predictions)
            for i in range(predictions_df.shape[1]):
                precision[i], recall[i], _ = precision_recall_curve(binary_labels[:, i],
                                                                    binary_predictions[:, i])
                average_precision[i] = average_precision_score(binary_labels[:, i], binary_predictions[:, i])

            # A "micro-average": quantifying score on all classes jointly
            precision["micro"], recall["micro"], _ = precision_recall_curve(binary_labels.ravel(),
                                                                            binary_predictions.ravel())
            average_precision["micro"] = average_precision_score(binary_labels, binary_predictions,
                                                                 average="micro")

            print('Average precision score, micro-averaged over all classes: {0:0.2f}'
                  .format(average_precision["micro"]))

        except Exception:
            print("Exception calucating precision/recall.")

        predictions_df["Exact Match"] = exact_match
        predictions_df["Single Miss"] = single_miss
        predictions_df["Double Miss"] = double_miss
        predictions_df["Accuracy"] = accuracy
        predictions_df["Num Labels"] = num_labels
        predictions_df["Predicted Num Labels"] = predicted_num_labels
        predictions_df["Num Labels Matching"] = num_labels_matching
        predictions_df["Additional Labels"] = np.array(predicted_num_labels) - np.array(num_labels_matching)

        print()
        print("Precision", precision)
        print("Recall", recall)

        print("-------")
        print("Micro Precision",precision["micro"])
        print("Micro Recall",recall["micro"])
        print("Average Precision", average_precision)

        predictions_df.sort_values(by=["Exact Match", "Single Miss", "Double Miss"], ascending=False, inplace=True)

        return predictions_df

    def predict(self, data, batch_size=1, output_vectors=False, output_subtitle_vectors=False, data_type="", load_encoded_data=False, save_encoded_data=False):

        if data is not None:
            if load_encoded_data:
                print("Loading encoded data for efficiency...")
                print("data/overview_encoded_data.dat")
                print("data/plot_encoded_data.dat")
                print("data/subtitle_encoded_data.dat")
                overview_encoded_data = self.load_pickle_data("data/weights/" + data_type + "_" if data_type != "" else "" + "overview_encoded_data.dat")
                subtitle_encoded_data = self.load_pickle_data("data/weights/" + data_type + "_" if data_type != "" else "" + "subtitle_encoded_data.dat")
            else:
                overview_encoded_data, subtitle_encoded_data = self.preprocess(data, save_encoded_data=save_encoded_data)

            labels = data["Labels"].apply(
                self.parse_str_labels)

            str_labels = data["Labels"].apply(
                self.to_str)

            ids = data["Id"]
            titles = data["Title"]

            print("Starting Predictions")

            X = [overview_encoded_data, subtitle_encoded_data]

            raw_predictions = self.model.predict(X, batch_size=batch_size, verbose=1)

            print("Finished with raw predictions...")

            predictions = []
            prediction_probs = []
            binary_predictions = []

            for i, prediction in enumerate(raw_predictions):
                indexes = [i for i, x in enumerate(prediction) if x >= 0.45]
                binary_prediction = [1 if x >= 0.45 else 0 for i, x in enumerate(prediction)]

                if len(indexes) > 0:
                    pred_text = itemgetter(*indexes)(self.genres)
                    pred_probs = itemgetter(*indexes)(prediction)
                else:
                    pred_text = ""
                    pred_probs = 0.0

                if type(pred_text) == str:
                    pred_text = [pred_text]
                    pred_probs = [pred_probs]

                binary_predictions.append(binary_prediction)
                pred_probs, pred_text = (list(t) for t in zip(*sorted(zip(pred_probs, pred_text), reverse=True)))
                prediction_probs.append(str([round(prob, 5) for prob in pred_probs]).replace("[", "").replace("]", "").replace("'", ""))
                predictions.append(str(pred_text).replace("[", "").replace("]", "").replace("'", ""))

            prediction_data_df = pd.DataFrame()
            prediction_data_df["Id"] = ids
            prediction_data_df["Title"] = titles
            prediction_data_df["Overview"] = data["Overview"]
            prediction_data_df["Plot"] = data["Plot"]
            prediction_data_df["Genre Labels"] = str_labels
            prediction_data_df["Genre Predictions"] = predictions
            prediction_data_df["Prediction Probabilities"] = prediction_probs
            prediction_data_df = prediction_data_df.replace(to_replace="[", value="").replace(to_replace="]", value="").replace(to_replace="'", value="")

            if output_vectors:
                if output_subtitle_vectors:
                    subtitles_output = self.model.get_layer("GlobalAvgPoolSubitles").output
                    encoded_data_model = Model(self.model.input, subtitles_output)
                    subtitles_vector = encoded_data_model.predict(X, use_multiprocessing=True, verbose=1, batch_size=batch_size)
                    print("Finished with subtitle vectors...")

                    sub_vectors = {}
                    for index, id in enumerate(ids):
                        sub_vectors[id] = list(subtitles_vector[index])

                    sub_df = pd.DataFrame.from_dict(sub_vectors, orient="index")
                    sub_df.to_csv("data/" + data_type + "_" if data_type != "" else "" + "subtitle_vectors.csv", header=False)

                pred_vectors = {}
                for index, id in enumerate(ids):
                    pred_vectors[id] = list(raw_predictions[index])

                pred_df = pd.DataFrame.from_dict(pred_vectors, orient="index")

                pred_df.to_csv("data/" + data_type + "_" if data_type != "" else "" + "output_vectors.csv", header=False)

            return binary_predictions, prediction_data_df

    class CallbackActions(Callback):
        def __init__(self, main_model=None, sentence_model=None, vectorizer=None):
            self.main_model = main_model
            self.sentence_model = sentence_model
            self.vectorizer = vectorizer
            return

        def on_train_begin(self, logs={}):
            return

        def on_train_end(self, logs={}):
            return

        def on_epoch_begin(self, epoch, logs={}):
            return

        def on_epoch_end(self, epoch, logs={}):
            self.main_model.save_weights("data/weights/main_model.h5")
            self.sentence_model.save_weights("data/weights/sentence_model.h5")
            self.vectorizer.save("data/weights/vectorizer.dat")
            return

if __name__ == "__main__":

    evaluate = False
    train = True
    load_weights = False

    vectorizer = MultiVectorizer(glove_path="D:/Development/Embeddings/Glove/glove.840B.300d.txt")
    genre_prediction = GenrePredictionModel(vectorizer=vectorizer, load_weights=load_weights)
    training_data_df, validation_data_df = genre_prediction.load_data("data/film_data_lots.xlsx")

    if evaluate:
        evaluation_df = genre_prediction.evaluate(validation_data_df, binary_labels=genre_prediction.validation_labels, batch_size=3)
        evaluation_df.to_excel("data/new_evaluation.xlsx", index=False)
    if train:
        genre_prediction.fit(training_data_df, genre_prediction.training_labels, validation_data=validation_data_df, validation_labels = genre_prediction.validation_labels, epochs=1200, batch_size=3)

    print("Done")
