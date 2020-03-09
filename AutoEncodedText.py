from collections import Counter

import tensorflow as tf
from common.MultiVectorizer import *
import pandas as pd
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM, TimeDistributed, SpatialDropout1D, Conv1D, MaxPooling1D, Dropout, AdditiveAttention, Attention, \
    GlobalAveragePooling1D, Concatenate, Bidirectional
from tensorflow.keras.models import Model
from common.data_utils import *
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm._tqdm_notebook import tqdm_notebook
from tensorflow.keras.metrics import *
tqdm_notebook.pandas()

class AutoEncoderTextModel():

    def __init__(self, vectorizer=None, load_weights=False):
        self.vectorizer = vectorizer
        self.load_weights = load_weights
        self.METRICS = [
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            Mean(name='mean')
        ]


    def load_data(self, file_path, rows=None, validation_split=None):
        data_df = pd.read_excel(file_path, nrows=rows)

        if validation_split is not None:
            data_df = data_df[data_df.Training == True].reset_index()
            training_df = data_df[:int(1 - data_df.shape[0]*validation_split)]
            validation_df = data_df[int(1 - data_df.shape[0]*validation_split)+1: - 1]
            pass
        else:
            training_df = data_df[data_df.Training == True]
            validation_df = data_df[data_df.Validation == True]

        training_df.loc[:,"Subtitles"] = training_df["Subtitles 1"] + training_df["Subtitles 2"]
        validation_df.loc[:, "Subtitles"] = validation_df["Subtitles 1"] + validation_df["Subtitles 2"]
        training_df.drop(["Subtitles 1", "Subtitles 2"], inplace=True, axis=1)
        validation_df.drop(["Subtitles 1", "Subtitles 2"], inplace=True, axis=1)

        training_df.loc[:,"Labels"] = training_df["Genres"].apply(self.parse_str_labels)
        validation_df.loc[:,"Labels"] = validation_df["Genres"].apply(self.parse_str_labels)

        with open("data/genres.pickle", "rb") as f:
            self.genres = pickle.load(f)

        mlb = MultiLabelBinarizer(classes=self.genres)

        training_binary_labels = mlb.fit_transform(training_df["Labels"])
        validation_binary_labels = mlb.fit_transform(validation_df["Labels"])

        self.training_labels =  training_binary_labels
        self.validation_labels =  validation_binary_labels

        return training_df, validation_df

    def get_sentences(self, text):
        text = str(text if type(text) == str else "")
        sentences = sent_tokenize(text)
        return sentences

    def parse_str_labels(self, str_labels):
        labels = list(map(str.strip, str_labels.split(",")))
        return labels

    def pad_list_of_lists(self, array, fill_value=0.0, shape=()):
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

        print("Max sentences:", max_sents)
        print("Max words:", max_words)

        print("Avg sentences:", avg_sents)
        print("Avg words:", avg_words)

        most_common_sents = max(list(zip(*most_common_sents))[0]) + 20
        most_common_words = max(list(zip(*most_common_words))[0]) + 20

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

    def preprocess(self, data):
        overview_data = self.vectorizer.fit(data["Overview"].apply(self.get_sentences).values)
        plot_data = self.vectorizer.fit(data["Plot"].apply(self.get_sentences).values)
        subtitles_data = self.vectorizer.fit(data["Subtitles"].apply(self.get_sentences).values)
        self.sentence_model, self.model = self.get_model()
        return self.pad_list_of_lists(overview_data), self.pad_list_of_lists(plot_data), self.pad_list_of_lists(subtitles_data)


    def get_model(self):

        print("Vocabulary Size:",vectorizer.get_vocabulary_size())

        overview_input = Input(shape=(None, None), dtype='int64', name="OverviewInput")
        plot_input = Input(shape=(None, None), dtype='int64', name="PlotInput")
        subtitles_input = Input(shape=(None, None), dtype='int64', name="SubtitlesInput")
        sentence_input = Input(shape=(None,), dtype='int64', name="SentenceInput")

        embedded_sentence = Embedding(vectorizer.get_vocabulary_size(), 300, trainable=True, name="Embedding")(sentence_input)
        spatial_dropout_sentence = SpatialDropout1D(0.20, name="SpatialDropoutSentence")(embedded_sentence)
        cnn_sentence = Conv1D(64, 4, padding="same", activation="relu", strides=1, name="Conv1DSentence")(spatial_dropout_sentence)
        max_pool_sentence = MaxPooling1D(pool_size=3, name="MaxPooling1DSentence")(cnn_sentence)
        sentence_encoding = Bidirectional(LSTM(500))(max_pool_sentence)
        sentence_model = Model(sentence_input, sentence_encoding)

        segment_time_distributed = TimeDistributed(sentence_model, name="TimeDistributedSegment")
        segment_cnn = Conv1D(172, 2, padding="same", activation="relu", name="SegmentConv1D")
        segment_max_pool = MaxPooling1D(pool_size=3, name="SegementMaxPool1D")

        segment_cnn_2 = Conv1D(172, 5, padding="same", activation="relu", name="Segment2Conv1D")
        segment_max_pool_2 = MaxPooling1D(pool_size=3, name = "Segment2MaxPool1D")

        overview_time_distributed = segment_time_distributed(overview_input)
        overview_cnn = segment_cnn(overview_time_distributed)
        overview_maxpool = segment_max_pool(overview_cnn)

        plot_time_distributed = segment_time_distributed(plot_input)
        plot_cnn = segment_cnn(plot_time_distributed)
        plot_maxpool = segment_max_pool(plot_cnn)

        subtitles_timedistributed = segment_time_distributed(subtitles_input)
        subtitles_cnn = segment_cnn_2(subtitles_timedistributed)
        subtitles_maxpool = segment_max_pool_2(subtitles_cnn)

        overview_dropout = SpatialDropout1D(0.40)(overview_maxpool)
        overview_pre_attention_output = Dense(172, name="OverviewPreAttnOutput")(overview_dropout)

        plot_dropout = SpatialDropout1D(0.40)(plot_maxpool)
        plot_pre_attention_output = Dense(172, name="PlotPreAttnOutput")(plot_dropout)

        subtitles_dropout = SpatialDropout1D(0.40, name="SubtitlesDropout")(subtitles_maxpool)
        subtitles_pre_attention_output = Dense(172, name="SubtitlesPreAttnOutput")(subtitles_dropout)

        attention_overview = AdditiveAttention(name="OverviewAttention")([overview_pre_attention_output, overview_maxpool])
        attention_plot = AdditiveAttention(name="PlotAttention")([plot_pre_attention_output, plot_maxpool])
        attention_subtitles = AdditiveAttention(name="SubtitlesAttention")([subtitles_pre_attention_output, subtitles_maxpool])

        overview_output = GlobalAveragePooling1D(name="GlobalAvgPoolOverview")(attention_overview)
        plot_output = GlobalAveragePooling1D(name="GlobalAvgPoolPlot")(attention_plot)
        subtitles_output = GlobalAveragePooling1D(name="GlobalAvgPoolSubitles")(attention_subtitles)

        concat_output = Concatenate(axis=-1, name="OutputConcatenate")([overview_output, plot_output, subtitles_output])
        dropput = Dropout(0.40)(concat_output)
        output = Dense(172, activation="sigmoid", name="Output")(dropput)

        model = Model([overview_input, plot_input, subtitles_input], output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adamax',
                      metrics=self.METRICS)

        print(sentence_model.summary())
        print(model.summary())
        self.sentence_model = sentence_model
        self.model = model
        if self.load_weights:
            self.sentence_model.load_weights("data/weights/sentence_model.h5")
            self.model.load_weights("data/weights/model.h5")
            self.vectorizer.load("data/weights/vectorizer.dat")
        return sentence_model, model

    def fit(self, data, labels, validation_data=None, validation_labels=None, batch_size=5, epochs=10):
        overview_input, plot_input, subtitles_input = self.preprocess(data)
        overview_validation_input, plot_validation_input, subtitles_validation_input = self.preprocess(validation_data)

        callback_actions = self.CallbackActions(main_model=self.model, sentence_model=self.sentence_model, vectorizer=self.vectorizer)

        checkpoint_path = "data/weights/checkpoints/cp-epoch_{epoch:02d}-accuracy_{accuracy:.3f}_precision_{precision:.3f}-recall_{recall:.3f}-auc_{auc:.3f}-mean_{mean:.3f}.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        self.model.fit([overview_input, plot_input, subtitles_input], labels, validation_data=([overview_validation_input, plot_validation_input, subtitles_validation_input], validation_labels), epochs=epochs, callbacks=[callback_actions, cp_callback], batch_size=batch_size)


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

    vectorizer = MultiVectorizer()
    auto_encoder_text = AutoEncoderTextModel(vectorizer=vectorizer)
    training_data_df, validation_data_df = auto_encoder_text.load_data("data/film_data.xlsx")
    auto_encoder_text.fit(training_data_df, auto_encoder_text.training_labels, validation_data=validation_data_df, validation_labels = auto_encoder_text.validation_labels, epochs=200, batch_size=5)


    print("Done")

    #auto_encoder_text.fit(X, y)

