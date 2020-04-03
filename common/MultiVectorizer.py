
from gensim.corpora.dictionary import Dictionary
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from tqdm import tqdm
import spacy
from spacy.lang.en import English
from common.data_utils import convert_to_string
from orderedset import OrderedSet
#from spellchecker import Spellchecker
import pandas as pd
from nltk.corpus import stopwords

class MultiVectorizer():

    reserved = ["<PAD>", "<UNK>"]
    embedding_matrix = None
    embedding_word_vector = {}
    glove = False

    def __init__(self, reserved=None, min_occur=1, use_bert=False, glove_path=None, tokenizer=None, embedding_size=300):

        self.mi_occur = min_occur
        self.embedding_size = embedding_size
        self.use_bert = use_bert

        self.nlp = spacy.load("en")
        if tokenizer is None:
            self.tokenizer = English().Defaults.create_tokenizer(self.nlp)
        else:
            self.tokenizer = tokenizer

        if glove_path is not None:
            self.load_glove(glove_path)
            self.glove = True

        if reserved is not None:
            self.vocabulary = Dictionary([self.reserved.extend(reserved)])
        else:
            self.vocabulary = Dictionary([self.reserved])

    def get_vocabulary_size(self):
        return len(self.vocabulary.token2id.items())

    def load_glove(self, glove_file_path):
        f = open(glove_file_path, encoding="utf-8")
        for line in tqdm(f):
            value = line.split(" ")
            word = value[0]
            coef = np.array(value[1:], dtype='float32')
            self.embedding_word_vector[word] = coef
        f.close()

    def get_embedding_matrix(self):
        return self.embedding_matrix

    def is_word(self, string_value):
        if self.embedding_word_vector.get(string_value):
            return True

    def get_vocabulary(self):
        return self.vocabulary

    def get_word_id(self, word):
        return self.vocabulary.token2id[word]

    def get_word_from_id(self, index):
        return self.vocabulary.id2token[index]

    def fit_document(self, documents):
        document_tokens = []
        for document in documents:
            section_tokens = []
            for section in document:
                sentence_tokens = []
                for sentence in section:
                    tokens = self.tokenizer(sentence.lower())
                    word_str_tokens = list(map(convert_to_string, tokens))
                    sentence_tokens.append(word_str_tokens)
                    self.vocabulary.add_documents(sentence_tokens)
                section_tokens.append(sentence_tokens)
            document_tokens.append(section_tokens)
        return document_tokens

    def fit_bert_sentences(self, samples, remove_stop_words=True):
        output_tokens = []
        vocab = []
        stop_words = set(stopwords.words('english'))
        for sample in tqdm(samples):
            sentence_tokens = []
            for sentence in sample:
                tokens = self.tokenizer.tokenize(sentence.lower())
                tokens = [w for w in tokens if not w in stop_words]
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                sentence_tokens.append(tokens)
                vocab.append(tokens)
            output_tokens.append(sentence_tokens)
        #self.vocabulary.add_documents(vocab)
        return output_tokens

    def fit_samples_with_sentences(self, samples, remove_stop_words=True):
        output_tokens = []
        vocab = []
        for sample in tqdm(samples):
            sentence_tokens = []
            for sentence in sample:
                tokens = self.tokenizer(sentence.lower())
                if remove_stop_words:
                    tokens = [token for token in tokens if not token.is_stop]
                word_str_tokens = list(map(convert_to_string, tokens))
                sentence_tokens.append(word_str_tokens)
                vocab.append(word_str_tokens)
            output_tokens.append(sentence_tokens)
        self.vocabulary.add_documents(vocab)
        return output_tokens

    def fit(self, X, remove_stop_words=True, list_of_lists=False):
        if list_of_lists:
            if not self.use_bert:
                x_tokens = self.fit_samples_with_sentences(X,remove_stop_words=remove_stop_words) #self.fit_document(X)
            else:
                x_tokens = self.fit_bert_sentences(X, remove_stop_words=remove_stop_words)
        else:
            x_tokens = self.fit_text(X)

        self.vocabulary.filter_extremes(no_below=self.mi_occur, no_above=1.0, keep_tokens=self.reserved)
        unknown_words = []
        if self.glove:
            #spell = Spellchecker()
            print("Vocabulary Size:",self.get_vocabulary_size())
            self.embedding_matrix = np.zeros((self.get_vocabulary_size(), self.embedding_size))
            for word, i in tqdm(self.vocabulary.token2id.items()):
                if word == "<PAD>":
                    embedding_value = np.zeros((1, self.embedding_size))
                elif word == "<UNK>":
                    sd =  1/np.sqrt(self.embedding_size)
                    np.random.seed(seed=42)
                    embedding_value = np.random.normal(0, scale=sd, size=[1, self.embedding_size])
                else:
                    embedding_value = self.embedding_word_vector.get(word)
                    if embedding_value is None:
                        embedding_value = self.embedding_word_vector.get(self.correct_word(word))
                        if embedding_value is None:
                            unknown_words.append(word)
                            embedding_value = self.embedding_word_vector.get("<UNK>")

                if embedding_value is not None:
                    self.embedding_matrix[i] = embedding_value
        print("Number of unknown words:",len(unknown_words))
        unknown_words_df = pd.DataFrame()
        unknown_words_df["Unknown Words"] = unknown_words
        unknown_words_df.to_excel("data/unknown_words.xlsx", index=False)
        encoded_tokens = self.transform(x_tokens, list_of_lists=list_of_lists)
        return  encoded_tokens

    def fit_text(self, X, remove_stop_words=True):
        output_tokens = []
        for sample in tqdm(X):
            tokens = self.tokenizer(sample.lower())
            if remove_stop_words:
                tokens = [token for token in tokens if not token.is_stop]
            word_str_tokens = list(map(convert_to_string, tokens))
            output_tokens.append(word_str_tokens)
        self.vocabulary.add_documents(output_tokens)
        return output_tokens

    def correct_word(self, word):
        return word

    def transform(self, X, list_of_lists=False):
        if list_of_lists:
            if not self.use_bert:
                return self.transform_list_of_list(X)
            else:
                return self.transform_bert(X)
        else:
            return self.transform_text(X)

    def transform_list_of_list(self, samples):
        samples_tokens = []
        for sample in samples:
            encoded_tokens = self.transform_text(sample)
            samples_tokens.append(encoded_tokens)
        return samples_tokens

    def transform_document(self, documents):
        document_tokens = []
        for document in documents:
            section_tokens = []
            encoded_tokens = []
            for section in document:
                if type(section) == str:
                    encoded_tokens.append(section)
                    if len(encoded_tokens) == len(document):
                        section_tokens.append(encoded_tokens)
                        section_tokens = self.transform_text(section_tokens)
                else:
                    encoded_tokens = self.transform_text(section)
                    section_tokens.append(encoded_tokens)
            document_tokens.append(section_tokens)
        return document_tokens

    def transform_bert(self, samples):
        samples_tokens = []
        for sample in samples:
            encoded_sentences = []
            for sentence_tokens in sample:
                encoded_tokens = self.tokenizer.convert_tokens_to_ids(sentence_tokens)
                encoded_sentences.append(encoded_tokens)
            samples_tokens.append(encoded_sentences)
        return samples_tokens

    def transform_text(self, X):
        if hasattr(self, "limit"):
            return [[i if i < self.limit else self.reserved.index("<UNK>")
                     for i in self.vocabulary.doc2idx(x, unknown_word_index=self.reserved.index("<UNK>"))]
                    for x in X]
        else:
            return [self.vocabulary.doc2idx(x, unknown_word_index=self.reserved.index("<UNK>")) for x in X]

    def inverse_transform(self, X):
        return [[ self.vocabulary[i] for i in x ] for x in X]

    def save(self, file_path="./vecorizer.vec"):
        with open(file_path, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return file_path

    @classmethod
    def load(cls, file_path):
        with open(file_path, "rb") as handle:
            self = pickle.load(handle)
        return self