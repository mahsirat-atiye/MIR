import re
import string
import json

import pandas as pnd
import numpy as np
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from phase2.config import Config


def my_log(x):
    if x == 0:
        return 0
    return np.log(x)


class TfIdf:
    def __init__(self, training_data, mode=None):
        """
        tf-idf from train data
        mode could be None, lemmatization, stemming, stopword_removal
        """
        self.train_data = training_data[:Config.TRAIN_TIME_SAVING]
        self.mode = mode

        self.postings = defaultdict(lambda: defaultdict(lambda: []))
        self.num_of_documents = len(training_data)
        self.documents_words = []
        self.term_frequency_in_doc = defaultdict(lambda: 1)
        self.all_terms = []
        self.word_to_id = {}
        self.df = {}

        self.construct_positional_indexes(self.train_data)
        # Learn the vocabulary dictionary and return term-document matrix.
        # Transform a count matrix to a normalized tf-idf representation
        #     The formula that is used to compute the tf-idf for a term t of a document d
        #     in a document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf is
        #     computed as idf(t) = log [ n / df(t) ] + 1 , where
        #     n is the total number of documents in the document set and df(t) is the
        #     document frequency of t; the document frequency is the number of documents
        #     in the document set that contain the term t. The effect of adding "1" to
        #     the idf in the equation above is that terms with zero idf, i.e., terms
        #     that occur in all documents in a training set, will not be entirely
        #     ignored.
        self.term_document_tf_idf_matrix = self.build_tf()

    def get_tfidf_vector(self, doc):
        """
        computes tf idf for a complete doc.
        """
        doc_body = doc[Config.BODY]
        doc_title = doc[Config.TITLE]
        return self.get_tfidf_vector_util(doc_body, doc_title)

    def get_tfidf_vector_util(self, doc_body, doc_title):
        vector = [0] * len(self.all_terms)
        # Transform documents to document-term matrix.
        # Extract token counts out of raw text documents using the vocabulary fitted
        document_words = self.prepare_text(doc_body)
        title_words = self.prepare_text(doc_title)
        num_of_document_words = len(document_words)
        num_of_title_words = len(title_words)
        num_of_title_document_words = num_of_document_words + num_of_title_words
        if num_of_title_document_words != 0:
            f = 1 / num_of_title_document_words
            for word in document_words:
                if word in self.word_to_id.keys():
                    vector[self.word_to_id[word]] += f
            for word in title_words:
                if word in self.word_to_id.keys():
                    vector[self.word_to_id[word]] += f
        for i, word in enumerate(self.df):
            vector[i] /= 1 + my_log(self.num_of_documents / self.df[word])
        return np.array(vector)

    def get_tf_idf_vector_for_docs(self, docs):
        docs_body = docs[Config.BODY].values
        docs_title = docs[Config.TITLE].values
        return np.array([self.get_tfidf_vector_util(doc_body, doc_title) for doc_body, doc_title in
                         list(zip(docs_body, docs_title))])

    def prepare_text_mode_None(self, text):
        punctuations = string.punctuation
        text = text.lower()
        text = re.sub('\d+', ' ', text)
        text = text.translate(str.maketrans(punctuations, ' ' * len(punctuations)))

        text = text.strip()
        words = word_tokenize(text)
        return words

    def prepare_text(self, text):
        words = self.prepare_text_mode_None(text)

        if self.mode == 'lemmatization':
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(w) for w in words]
            return words

        elif self.mode == 'stemming':
            stemmer = PorterStemmer()
            words = [stemmer.stem(w) for w in words]
            return words

        elif self.mode == 'stopword_removal':
            words = [w for w in words if w not in stopwords.words()]
            return words

        else:
            return words

    def construct_positional_indexes(self, documents):
        documents_body = documents[Config.BODY].values
        documents_title = documents[Config.TITLE].values

        for i, document in enumerate(documents_body):
            document_words = self.prepare_text(documents_body[i])
            title_words = self.prepare_text(documents_title[i])
            num_of_document_words = len(document_words)
            num_of_title_words = len(title_words)
            num_of_title_document_words = num_of_document_words + num_of_title_words

            if num_of_title_document_words == 0:
                continue
            f = 1 / num_of_title_document_words
            for word in document_words:
                self.term_frequency_in_doc[word] += f
            f = 1 / num_of_title_document_words
            for word in title_words:
                self.term_frequency_in_doc[word] += f

            title_words.extend(document_words)
            document_words = title_words
            self.documents_words.append(document_words)

        for i, document_words in enumerate(self.documents_words):
            doc_id = i
            for j, document_word in enumerate(document_words):
                self.postings[document_word][doc_id].append(j)

    def build_tf(self):
        print("Building tf idf table...")
        self.all_terms = list(self.postings.keys())
        self.word_to_id = {t: i for i, t in enumerate(self.all_terms)}
        tf = np.zeros(shape=(self.num_of_documents, len(self.all_terms)))
        # all terms
        # d
        # o
        # c
        # u
        # m
        # e
        # n
        # t
        # s
        for word in self.postings.keys():
            self.df[word] = len(self.postings[word].keys())
            for document_including_word in self.postings[word].keys():
                NTN_idf = 1 + my_log(self.num_of_documents / self.df[word])
                tf[document_including_word][self.word_to_id[word]] = len(
                    self.postings[word][document_including_word]) * NTN_idf

        print("tf idf table built!")
        return tf


if __name__ == '__main__':
    # Demo

    with open(Config.TRAIN_DATA) as training_data_file:
        training_data = json.loads(training_data_file.read())
    train_data = pnd.read_json(json.dumps(training_data))
    tfidf = TfIdf(train_data)
    print(train_data.shape)

    a = tfidf.get_tf_idf_vector_for_docs(train_data[-5:])

    print(a)
