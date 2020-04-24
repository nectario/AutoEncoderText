#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import math
import time
from collections import Counter
from operator import itemgetter

import tensorflow as tf
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tqdm._tqdm_notebook import tqdm_notebook

from common.MultiVectorizer import *
from common.data_utils import *

MAX_WORDS = 100
MAX_SENTENCES = 200

MAX_TEXT_LENGTH = 12500
tqdm_notebook.pandas()
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
print(K.image_data_format())
K.set_image_data_format('channels_last')

from bert import BertModelLayer
from bert.tokenization.bert_tokenization import FullTokenizer
from bert import params_from_pretrained_ckpt

import re

unknown_words = set()

class GenrePredictionModel():
    def __init__(self, vectorizer=None, load_weights=False, use_bert=True, bert_model_name=None):
        self.bert_model_name = bert_model_name

        if use_bert:
            self.vocabulary = set(line.strip() for line in open("D:/Development/Projects/bert_models/"+self.bert_model_name+"/vocab.txt", encoding="utf-8"))
            self.bert_tokenizer = FullTokenizer(vocab_file=os.path.join("D:/Development/Projects/bert_models/"+self.bert_model_name, "vocab.txt"))
            self.vectorizer = MultiVectorizer(tokenizer=self.bert_tokenizer, use_bert=use_bert)
        else:
            self.vectorizer = vectorizer

        self.load_weights = load_weights
        self.max_shape = None

        if self.load_weights:
            self.vectorizer = self.vectorizer.load("data/weights/vectorizer.dat")

        self.METRICS = [
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]

        self.checkpoint_path = "N:/weights/checkpoints/cp-epoch_{epoch:02d}-accuracy_{accuracy:.3f}_val_precision_{val_precision:.3f}-val_recall_{val_recall:.3f}-val_auc_{val_auc:.3f}.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        print("Checkpoint dir:",self.checkpoint_dir)

    def remove_unused_genres(self, genre_list, genres):
        if (type(genre_list) == str):
            genre_list = self.parse_str_labels(genre_list)
        genres_set = OrderedSet(list(genres))
        genre_list_set = OrderedSet(list(genre_list))
        output = list(genre_list_set.intersection(genres_set))
        return output

    def load_data(self, file_path, rows=None):

        data_df = pd.read_excel(file_path, nrows=rows)
        self.genres = self.load_current_genres(file_path="data/93_labels_data/new_93_genres.xlsx")

        train_ids = pd.read_excel("data/train_ids.xlsx")
        test_ids = pd.read_excel("data/test_ids.xlsx")

        train_ids = train_ids.set_index("Id")
        test_ids = test_ids.set_index("Id")
        data_df = data_df.set_index("Id")

        data_df["Id"] = data_df.index

        data_df = data_df[data_df.index.isin(train_ids.append(test_ids).index)][["Id","Title", "Labels", "Subtitles 1", "Subtitles 2"]]

        #data_df["Labels"] = data_df["Labels"].apply(self.remove_unused_genres, args=(self.genres,)).tolist()

        data_df.replace("vislted", "visited", inplace=True)
        data_df.replace(r'[^\x00-\x7F]+', ' ', inplace=True, regex=True)
        data_df.replace(0, "", inplace=True)
        data_df["Subtitles 2"].fillna("", inplace=True)
        data_df["Subtitles 1"].fillna("", inplace=True)
        data_df["Subtitles"] = data_df["Subtitles 1"] + data_df["Subtitles 2"]
        data_df = data_df[data_df.Subtitles != ""]

        #data_df.drop(["Subtitles 1", "Subtitles 2"], inplace=True, axis=1)

        training_data_df = data_df[data_df.index.isin(train_ids.index)]
        validation_data_df = data_df[data_df.index.isin(test_ids.index)]

        #training_data_df.to_excel("data/train_data.xlsx", index=False)
        #validation_data_df.to_excel("data/test_data.xlsx", index=False)

        mlb = MultiLabelBinarizer(classes=self.genres)

        training_data_df["List Labels"] = training_data_df["Labels"].apply(self.parse_str_labels)
        validation_data_df["List Labels"] = validation_data_df["Labels"].apply(self.parse_str_labels)

        training_binary_labels = mlb.fit_transform(training_data_df["List Labels"])
        validation_binary_labels = mlb.fit_transform(validation_data_df["List Labels"])

        self.training_labels = training_binary_labels
        self.validation_labels = validation_binary_labels

        return training_data_df, validation_data_df

    def load_current_genres(self, file_path, load_original=False):
        if load_original:
            with open("data/genres.pickle", "rb") as f:
                genres = pickle.load(f)
                return genres
        current_genres_df = pd.read_excel(file_path)
        genres = current_genres_df["Genre"].tolist()
        return genres

    def get_sentences(self, text):
        text = str(text if type(text) == str else "")
        sentences = sent_tokenize(text)
        return sentences

    def clean_text(self, text):

        atext = text.replace('-', ' ')
        atext = atext.lower()
        atext = atext.replace("there's", "there is")
        atext = atext.replace("she's", "she is")
        atext = atext.replace("he's", "he is")
        atext = atext.replace("I'm", "I am")
        atext = atext.replace("could've", "could have")
        atext = atext.replace("wasn't", "was not")
        atext = atext.replace("doesn't", "does not")
        atext = atext.replace("hadn't", "had not")
        atext = atext.replace("didn't", "did not")
        atext = atext.replace("isn't", "is not")
        atext = atext.replace("isnt", "is not")
        atext = atext.replace("didnt", "did not")
        atext = atext.replace("hadnt", "had not")
        atext = atext.replace("doesnt", "does not")
        atext = atext.replace("shes", "she is")
        atext = atext.replace("hes", "he is")


        clean_text = re.sub('[^A-Za-z0-9 ]+', '', atext)

        for word in clean_text.split():
            if word not in self.vocabulary:
                unknown_words.add(word)

        return clean_text

    def get_chunks(self, text):
        text = str(text if type(text) == str else "")
        text = text.replace("...",".")
        sentences = sent_tokenize(text)
        chunks = []
        chunk = ""
        lengths = []
        for i, sentence in enumerate(sentences):
            regx = r"subtitles [a-z0-9]+"
            regx_2 = r"^\s"
            sentence_text = re.sub(regx, "", sentence)
            sentence_text = re.sub(regx_2, "", sentence_text)
            chunk = chunk + " " + sentence_text if chunk != "" else sentence_text
            chunk = self.clean_text(chunk)
            if len(chunk) >= MAX_WORDS and  i != len(sentences) - 1:
                chunks.append(chunk)
                lengths.append(len(chunk))
                chunk = ""
            elif i == len(sentences) - 1:
                chunks.append(chunk)
                chunk = ""
        return chunks

    def parse_str_labels(self, str_labels):
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

    def pad_list_of_lists(self, array, fill_value=0.0, shape=(), debug=True):

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
        most_common_sents = Counter(sent_lens).most_common(50)
        most_common_words = Counter(word_lens).most_common(50)
        most_common_sents = max(list(zip(*most_common_sents))[0])
        most_common_words = max(list(zip(*most_common_words))[0])

        if debug:
            print("Max sentences:", max_sents)
            print("Max words:", max_words)
            print("Avg sentences:", avg_sents)
            print("Avg words:", avg_words)
            print("Most common sentences:", most_common_sents)
            print("Most common words:", most_common_words)

        chosen_words = MAX_WORDS
        chosen_sents = MAX_SENTENCES

        shape = (batch_size, chosen_sents, chosen_words)
        result = np.full(shape, fill_value, dtype="float32")
        for index, value in enumerate(array):
            if index == shape[0]:
                break
            for idx, row in enumerate(value):
                if idx == shape[1]:
                    break
                # result[index: len(value)] = value
                result[index, idx, :len(row) if len(row) < shape[2] else shape[2]] = row[:len(row) if len(row) < shape[2] else shape[2]]
        return result

    def pad(self, array, max_length=None, fill_value=0.0, debug=True):
        word_lens = []
        for i, text in enumerate(array):
            word_lens.append(len(text))

        max_words = max(word_lens)
        avg_words = np.mean(word_lens)
        most_common_words = Counter(word_lens).most_common(20)
        most_common_words = max(list(zip(*most_common_words))[0]) + 1000

        if debug:
            print("Max words:", max_words)
            print("Avg words:", avg_words)
            print("Most common words:", most_common_words)

        result = pad_sequences(array, maxlen=max_length, dtype="int32", padding='post', truncating='post')

        return result

    def preprocess_simple(self, data, max_length=None):
        subtitles_data = np.array(self.vectorizer.fit(data["Subtitles"].values, list_of_lists=False))
        #overview_data = np.array(self.vectorizer.fit(data["Overview"].values, list_of_lists=False))
        padded_data = self.pad(subtitles_data, max_length=max_length)
        return padded_data

    def preprocess(self, data, save_encoded_data=False, data_type=""):

        subtitles_data = self.vectorizer.fit(data["Subtitles"].apply(self.get_chunks).values, list_of_lists=True)

        if save_encoded_data:
            timestr = time.strftime("%m%d")

            subtitles_data = self.pad_list_of_lists(subtitles_data)
            self.save_pickle_data(subtitles_data, file_path="data"+data_type+"_subtitles_encoded_data.dat")

            return subtitles_data

        return self.pad_list_of_lists(subtitles_data)

    def save_pickle_data(self, overview_data, file_path=None):
        if file_path is None:
            raise ValueError("Please specify file path in save_pickle_data()!")

        with open(file_path, "wb") as handle:
            pickle.dump(overview_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle_data(self, file_path):
        with open(file_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            return data

    def flatten_layers(self, root_layer):
        if isinstance(root_layer, Layer):
            yield root_layer
        for layer in root_layer._layers:
            for sub_layer in self.flatten_layers(layer):
                yield sub_layer

    def freeze_bert_layers(self, l_bert):
        """
        Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
        """
        for layer in self.flatten_layers(l_bert):
            if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
                layer.trainable = True
            elif len(layer._layers) == 0:
                layer.trainable = False
            l_bert.embeddings_layer.trainable = False

    def create_learning_rate_scheduler(self, max_learn_rate=5e-5,
                                       end_learn_rate=1e-7,
                                       warmup_epoch_count=10,
                                       total_epoch_count=90):

        def lr_scheduler(epoch):
            if epoch < warmup_epoch_count:
                res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
            else:
                res = max_learn_rate * math.exp(
                    math.log(end_learn_rate / max_learn_rate) * (epoch - warmup_epoch_count + 1) / (total_epoch_count - warmup_epoch_count + 1))
            return float(res)

        learning_rate_scheduler = LearningRateScheduler(lr_scheduler, verbose=1)

        return learning_rate_scheduler

    def example(self):

        # Encode each timestep
        in_sentence = Input(shape=(None,),  dtype='int64', name="Input1")
        embedded_sentence = Embedding(1000, 300, trainable=False)(in_sentence)
        lstm_sentence = LSTM(300)(embedded_sentence)
        sentence_model = Model(in_sentence, lstm_sentence)

        section_input = Input(shape=(None, None), dtype='int64', name="Input2")
        section_encoded = TimeDistributed(sentence_model)(section_input)
        section_encoded = LSTM(300)(section_encoded)
        section_encoded = Dense(1)(section_encoded)
        section_model = Model(section_input, section_encoded)

        section_model.compile(loss='mse',
                      optimizer='adam',
                      metrics=['accuracy'])

        print(sentence_model.summary())
        print(section_model.summary())
        return section_model

    def bert_2(self,  bert_config_file=None, bert_ckpt_file=None):
        with tf.io.gfile.GFile(bert_config_file, "r") as reader:
            bert_params = params_from_pretrained_ckpt(bert_ckpt_file)
            l_bert = BertModelLayer.from_params(bert_params, name="bert")
            #l_bert.apply_adapter_freeze()
            #l_bert.embeddings_layer.trainable = False

        in_sentence = Input(shape=(150,), dtype='int64', name="Input1")

        bert_output = l_bert(in_sentence)

        lstm_output = GlobalAveragePooling1D()(bert_output)
        sentence_model = Model(in_sentence, lstm_output)

        section_input = Input(shape=(300, 150), dtype='int64', name="Input2")
        section_encoded = TimeDistributed(sentence_model)(section_input)
        section_encoded = LSTM(300)(section_encoded)
        section_encoded = Dense(21)(section_encoded)
        section_model = Model(section_input, section_encoded)

        section_model.compile(optimizer="adam",
                           loss="binary_crossentropy")

        sentence_model.summary()
        section_model.summary()

        return section_model



    def bert_model(self, max_seq_len, number_of_labels=93, adapter_size=64, bert_config_file=None, bert_ckpt_file=None, bert_model_name=None):

        with tf.io.gfile.GFile(bert_config_file, "r") as reader:
            bert_params = params_from_pretrained_ckpt(bert_ckpt_file)
            l_bert = BertModelLayer.from_params(bert_params, name="bert")

        sentence_input = Input(shape=(MAX_WORDS,), dtype='float32', name="sentence_input_ids")

        bert_output = l_bert(sentence_input)
        bert_output = GlobalAveragePooling1D()(bert_output)

        self.bert_model = Model(sentence_input, bert_output)

        chunk_input = Input(shape=(MAX_SENTENCES, MAX_WORDS), dtype='int32', name="SubtitlesInput")

        chunk_timedistributed = TimeDistributed(self.bert_model, name="TimeDistributedSegment")(chunk_input)
        chunk_cnn = Conv1D(256, 2, padding="same", strides=1, activation="relu", name="Segment2Conv1D")(chunk_timedistributed)
        cnn_output = MaxPooling1D(pool_size=3, name="Segment2MaxPool1D")(chunk_cnn)
        cnn_output = GlobalAveragePooling1D()(cnn_output)
        dense = Dense(512, activation="sigmoid", name="dense")(cnn_output)

        dropput = Dropout(0.10)(dense)

        output = Dense(number_of_labels, activation="sigmoid", name="Output")(dropput)

        self.model = Model(inputs=chunk_input, outputs=output)
        self.model.compile(optimizer="adam",
                      loss="binary_crossentropy", metrics=self.METRICS)

        self.bert_model.summary()
        self.model.summary()

        return self.bert_model, self.model


    def get_one_input_model(self, embedding_size=300, lstm_output_size=300, number_of_labels=21):

        print("Vocabulary Size:", self.vectorizer.get_vocabulary_size())

        subtitles_input = Input(shape=(None, None), dtype='int64', name="SubtitlesInput")
        sentence_input = Input(shape=(None,), dtype='int64', name="SentenceInput")

        embedded_sentence = Embedding(self.vectorizer.get_vocabulary_size(), embedding_size, trainable=True, name="Embedding")(sentence_input)
        spatial_dropout_sentence = SpatialDropout1D(0.10, name="SpatialDropoutSentence")(embedded_sentence)
        cnn_sentence = Conv1D(220, 3, padding="same", activation="relu", strides=1, name="Conv1DSentence")(spatial_dropout_sentence)
        max_pool_sentence = MaxPooling1D(pool_size=3, name="MaxPooling1DSentence")(cnn_sentence)

        sentence_encoding = Bidirectional(LSTM(lstm_output_size, recurrent_dropout=0.10))(max_pool_sentence)

        sentence_model = Model(sentence_input, sentence_encoding)

        segment_time_distributed = TimeDistributed(sentence_model, name="TimeDistributedSegment")
        segment_cnn_2 = Conv1D(number_of_labels, 3, padding="same", activation="relu", name="Segment2Conv1D")
        segment_max_pool_2 = MaxPooling1D(pool_size=3, name="Segment2MaxPool1D")

        subtitles_timedistributed = segment_time_distributed(subtitles_input)
        subtitles_cnn = segment_cnn_2(subtitles_timedistributed)
        subtitles_maxpool = segment_max_pool_2(subtitles_cnn)

        subtitles_dropout = SpatialDropout1D(0.10, name="SubtitlesDropout")(subtitles_maxpool)
        subtitles_pre_attention_output = Dense(number_of_labels, name="SubtitlesPreAttnOutput")(subtitles_dropout)

        attention_subtitles = Attention(name="SubtitlesAttention")([subtitles_pre_attention_output, subtitles_maxpool])

        subtitles_max_output = GlobalMaxPool1D(name="GlobalMaxPoolSubitles")(attention_subtitles)
        subtitles_avg_output = GlobalAveragePooling1D(name="GlobalAvgPoolSubitles")(attention_subtitles)

        concat_output = Concatenate(axis=-1, name="OutputConcatenate")([subtitles_max_output, subtitles_avg_output])
        dropput = Dropout(0.20)(concat_output)
        output = Dense(number_of_labels, activation="sigmoid", name="Output")(dropput)

        model = Model(subtitles_input, output)

        model.compile(loss='binary_crossentropy',
                      optimizer='adamax',
                      metrics=self.METRICS)




        print(sentence_model.summary())
        print(model.summary())
        self.sentence_model = sentence_model
        self.model = model
        if self.load_weights:
            sentence_weights = "data/weights/sentence_model.h5"
            weight_file = "data/genre_prediction.h5"
            print("Loading weights",weight_file)
            print("Loading weights", sentence_weights)
            self.sentence_model.load_weights(sentence_weights)
            self.model.load_weights(weight_file)

        self.model_vectors = Model(subtitles_input, subtitles_avg_output)

        return sentence_model, model


    def fit(self, data, labels, validation_data=None, load_encoded_data=False, use_subtitles_only=True, validation_labels=None, batch_size=5, epochs=10, save_encoded_data=False):

        if load_encoded_data:
            print("Loading encoded data for efficiency...")
            subtitles_input = self.load_pickle_data("data/train_subtitles_encoded_data.dat")
            subtitles_validation_input = self.load_pickle_data("data/test_subtitles_encoded_data.dat")
            validation_data = (subtitles_validation_input, validation_labels)
        else:
            print("Pre-processing training data...")
            subtitles_input = self.preprocess(data, save_encoded_data=save_encoded_data, data_type="train")

            if validation_data is not None:
                print("Pre-processing validation data...")
                subtitles_validation_input = self.preprocess(validation_data, save_encoded_data=save_encoded_data, data_type="test")
                validation_data = (subtitles_validation_input, validation_labels)

        self.sentence_model, self.model = self.get_one_input_model(number_of_labels=len(self.genres))

        callback_actions = self.CallbackActions(main_model=self.model, sentence_model=self.sentence_model, vectorizer=self.vectorizer)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        if use_subtitles_only:
            self.model.fit(subtitles_input, labels, validation_data=validation_data, epochs=epochs, callbacks=[callback_actions, cp_callback],
                           batch_size=batch_size)


    def fit_bert(self, data, labels, validation_data=None, validation_labels=None, batch_size=5, epochs=100, save_encoded_data=False):
        print("Pre-processing training data...")
        subtitles_input = self.preprocess(data)
        subtitles_validation_input = None
        if validation_data is not None:
            print("Pre-processing validation data...")
            subtitles_validation_input = self.preprocess(validation_data)
            validation_data = (subtitles_validation_input, validation_labels)
        self.max_shape = subtitles_input.shape

        with open('data/unknown_words.txt', 'w') as f:
            for word in unknown_words:
                f.write(word + '\n')

        print("Max Sentence Length:", MAX_WORDS)
        self.bert_model, self.model = self.bert_model(MAX_WORDS, bert_ckpt_file="D:/Development/Projects/bert_models/"+self.bert_model_name,
                                                          bert_config_file="D:/Development/Projects/bert_models/"+self.bert_model_name+"/bert_config.json",
                                                          number_of_labels=len(self.genres))

        callback_actions = self.CallbackActions(main_model=self.model, bert_model=self.bert_model, vectorizer=self.vectorizer)
        #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,save_weights_only=True, verbose=1)

        log_dir = "log"
        tensorboard_callback = TensorBoard(log_dir=log_dir)
        print("Batch Size:", batch_size)
        # model.fit(x=(data.train_x, data.train_x_token_types), y=data.train_y,

        self.model.fit(subtitles_input, labels,
                  validation_data=validation_data,
                  batch_size=batch_size,
                  shuffle=True,
                  epochs=epochs,
                  callbacks=[self.create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                            end_learn_rate=1e-7,
                                                            warmup_epoch_count=20,
                                                            total_epoch_count=epochs),
                             EarlyStopping(patience=20, restore_best_weights=True),
                             tensorboard_callback, callback_actions])

        self.model.save_weights('data/weights/model_bert.h5', overwrite=True)


    def evaluate(self, data=None, binary_labels=None, file_path=None, genre_column="Genre Labels", batch_size=2, output_vectors=False, data_type="",
                 load_encoded_data=False,
                 save_encoded_data=False, threshold=0.5):

        binary_predictions = None
        self.sentence_model, self.model = self.get_one_input_model(number_of_labels=len(self.genres))

        if file_path is not None:
            predictions_df = pd.read_excel(file_path)
        else:
            binary_predictions, predictions_df = self.predict(data, batch_size=batch_size, output_vectors=output_vectors, data_type=data_type,
                                                              load_encoded_data=load_encoded_data,
                                                              save_encoded_data=save_encoded_data, threshold=threshold)
        exact_match = []
        num_labels_matching = []
        single_miss = []
        single_miss_strict = []
        double_miss = []
        double_miss_strict = []
        num_labels = []
        predicted_num_labels = []
        accuracy_union = []
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

            if len(set_union) > 0:
                accuracy_union.append(round(len(set_intersection) / (len(set_union)), 4) * 100)
            else:
                accuracy_union.append(0)

            accuracy.append(round(len(set_intersection) / (len(genre_labels)), 4) * 100)

            diff = 0
            if len(genre_predictions) > 0 and len(set_intersection) > 0 and (len(genre_labels) != len(genre_predictions)):
                diff = len(genre_labels.difference(set_intersection))
            elif len(genre_predictions) > 0 and len(genre_labels) == len(genre_predictions):
                diff = np.absolute(len(genre_labels) - len(set_intersection))

            if diff == 1 and len(genre_labels) >= 1 and len(genre_predictions) >= 1:
                single_miss.append(True)
            else:
                single_miss.append(False)

            if diff == 2 and len(genre_labels) >= 2 and len(genre_predictions) >= 2:
                double_miss.append(True)
            else:
                double_miss.append(False)

            if diff == 1 and len(genre_labels) > 1 and len(genre_predictions) > 1:
                single_miss_strict.append(True)
            else:
                single_miss_strict.append(False)

            if diff == 2 and len(genre_labels) > 2 and len(genre_predictions) > 2:
                double_miss_strict.append(True)
            else:
                double_miss_strict.append(False)

        exact_match_avg = sum(exact_match) / len(exact_match)
        single_miss_avg = sum(single_miss) / len(single_miss)
        single_miss_strict_avg = sum(single_miss_strict) / len(single_miss_strict)
        double_miss_avg = sum(double_miss) / len(double_miss)
        double_miss_strict_avg = sum(double_miss_strict) / len(double_miss_strict)
        accuracy_union_avg = sum(accuracy_union) / len(accuracy_union)
        accuracy_avg = sum(accuracy) / len(accuracy)

        print("Threshold:", threshold)
        print("Exact Match:", exact_match_avg)
        print("Single Miss:", single_miss_avg)
        print("Single Miss Strict:", single_miss_strict_avg)
        print("Double Miss:", double_miss_avg)
        print("Double Miss Strict:", double_miss_strict_avg)
        print("Accuracy:", accuracy_avg)
        print("Accuracy Union:", accuracy_union_avg)

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
        predictions_df["Single Miss Strict"] = single_miss_strict
        predictions_df["Double Miss"] = double_miss
        predictions_df["Double Miss Strict"] = double_miss_strict
        predictions_df["Accuracy"] = accuracy
        predictions_df["Accuracy Union"] = accuracy_union
        predictions_df["Num Labels"] = num_labels
        predictions_df["Predicted Num Labels"] = predicted_num_labels
        predictions_df["Num Labels Matching"] = num_labels_matching
        predictions_df["Additional Labels"] = np.array(predicted_num_labels) - np.array(num_labels_matching)

        predictions_df.sort_values(by=["Exact Match", "Single Miss Strict", "Double Miss Strict", "Num Labels"], ascending=False, inplace=True)

        return predictions_df


    def predict(self, data, batch_size=1, output_vectors=False, output_subtitle_vectors=False, simple=True, data_type="", load_encoded_data=False, save_encoded_data=False, threshold=0.50):
        overview_encoded_data = None
        subtitle_encoded_data = None
        if data is not None:
            if load_encoded_data:
                print("Loading encoded data for efficiency...")
                print("data/subtitle_encoded_data.dat")
                subtitle_encoded_data = self.load_pickle_data("data/weights/subtitle_encoded_data.dat")
            else:
                if not simple:
                    overview_encoded_data, subtitle_encoded_data = self.preprocess(data, save_encoded_data=save_encoded_data)
                else:
                    subtitle_encoded_data =  self.preprocess_simple(data, max_length=MAX_TEXT_LENGTH)

            str_labels = data["Labels"].apply(
                self.to_str)

            ids = data["Id"]
            titles = data["Title"]

            print("Starting Predictions")

            if simple:
                X = subtitle_encoded_data
            else:
                X = [overview_encoded_data, subtitle_encoded_data]

            raw_predictions = self.model.predict(X, batch_size=batch_size, verbose=1)

            print("Finished with raw predictions...")

            predictions = []
            prediction_probs = []
            binary_predictions = []

            for i, prediction in enumerate(raw_predictions):
                indexes = [i for i, x in enumerate(prediction) if x >= threshold]
                binary_prediction = [1 if x >= threshold else 0 for i, x in enumerate(prediction)]

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
        def __init__(self, main_model=None, sentence_model=None, bert_model=None, vectorizer=None):
            self.main_model = main_model
            self.sentence_model = sentence_model
            self.vectorizer = vectorizer
            self.bert_model = bert_model

            return

        def on_train_begin(self, logs={}):
            return

        def on_train_end(self, logs={}):
            return

        def on_epoch_begin(self, epoch, logs={}):
            return

        def on_epoch_end(self, epoch, logs={}):
            self.main_model.save_weights("data/weights/main_model.h5")

            if self.bert_model is not None:
                self.bert_model.save_weights("data/weights/bert_model.h5")

            if self.sentence_model is not None:
                self.sentence_model.save_weights("data/weights/sentence_model.h5")

            #if epoch == 1:
                #self.vectorizer.save("N:/weights/vectorizer.dat")
            return

if __name__ == "__main__":

    evaluate = False
    train = True
    load_weights = False

    #vectorizer = MultiVectorizer(glove_path="D:/Development/Embeddings/Glove/glove.840B.300d.txt")

    #validation_data_filepath = "data/validation_data_2152.xlsx"

    #if evaluate:
    #    evaluation_df = genre_prediction.evaluate(validation_data_df, binary_labels=genre_prediction.validation_labels, batch_size=7)
    #    evaluation_df.to_excel("data/validation_evaluation.xlsx", index=False)

    bert = True

    if bert:
        genre_prediction = GenrePredictionModel(load_weights=load_weights, use_bert=True, bert_model_name="uncased_L-4_H-512_A-8")
        training_data_df, validation_data_df = genre_prediction.load_data("data/93_labels_data/new_gold_data.xlsx")


        genre_prediction.fit_bert(training_data_df, genre_prediction.training_labels, validation_data=validation_data_df,
                                                                        validation_labels=genre_prediction.validation_labels, epochs=1000, batch_size=1, save_encoded_data=True)
    else:
        vectorizer = MultiVectorizer()#glove_path="D:/Development/Embeddings/Glove/glove.840B.300d.txt")
        genre_prediction = GenrePredictionModel(vectorizer=vectorizer, load_weights=load_weights, use_bert=False)
        training_data_df, validation_data_df = genre_prediction.load_data("data/film_data_lots.xlsx")
        genre_prediction.fit(training_data_df, genre_prediction.training_labels, validation_data=validation_data_df, use_subtitles_only=True,
                              validation_labels=genre_prediction.validation_labels, epochs=1000, batch_size=5, load_encoded_data=True, save_encoded_data=True)

        #genre_prediction.fit(training_data_df, genre_prediction.training_labels, validation_data=validation_data_df, validation_labels = genre_prediction.validation_labels, epochs=1200, batch_size=2, save_encoded_data=True)

    print("Done")
