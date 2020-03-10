
from gensim.corpora.dictionary import Dictionary
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from tqdm import tqdm
import spacy
from spacy.lang.en import English
from common.data_utils import convert_to_string
from orderedset import OrderedSet
from spellchecker import SpellChecker

class MultiVectorizer():

    reserved = ["<PAD>", "<UNK>"]
    embedding_matrix = None
    embedding_word_vector = {}
    glove = False

    def __init__(self, reserved=None, min_occur=1, glove_path=None, tokenizer=None, embedding_size=300):

        self.mi_occur = min_occur
        self.embedding_size = embedding_size

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

    def fit(self, X, remove_stop_words=True):
        if type(X[0]) == list:
            x_tokens = self.fit_samples_with_sentences(X,remove_stop_words=remove_stop_words) #self.fit_document(X)
        else:
            x_tokens = self.fit_text(X)

        self.vocabulary.filter_extremes(no_below=self.mi_occur, no_above=1.0, keep_tokens=self.reserved)
        unknown_words = []
        if self.glove:
            spell = SpellChecker()
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
                        print("Will replace word:", word, spell.correction(word))
                        embedding_value = self.embedding_word_vector.get(spell.correction(word))
                        if embedding_value is None:
                            embedding_value = self.embedding_word_vector.get("<UNK>")
                            #print("Unknown word:",word)
                            unknown_words.append(word)

                if embedding_value is not None:
                    self.embedding_matrix[i] = embedding_value
        print("Number of unknown words:",len(unknown_words))
        return  self.transform(x_tokens)

    def fit_text(self, X):
        x_tokens = []
        for x in X:
            if x is not None:
                # x_tokens.append(word_tokenize(x.lower()))
                tokens = self.tokenizer(x.lower())
                word_str_tokens = list(map(convert_to_string, tokens))
                x_tokens.append(word_str_tokens)
                self.vocabulary.add_documents(x_tokens)
        return x_tokens

    def transform(self, X):
        return self.transform_list_of_list(X)

    def transform_list_of_list(self, samples):
        samples_tokens = []
        for sample in samples:
            encoded_tokens = self.transform_section(sample)
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
                        section_tokens = self.transform_section(section_tokens)
                else:
                    encoded_tokens = self.transform_section(section)
                    section_tokens.append(encoded_tokens)
            document_tokens.append(section_tokens)
        return document_tokens

    def transform_section(self, X):
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