import json

import numpy as np
import pandas as pnd

from phase2.config import Config
from phase2.evaluator import accuracy, recall, precision, f1score, confusion_matrix
from phase2.tf_idf import TfIdf

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Naive Bayes')


class NaiveBayesClassifier:
    def __init__(self, data, validation_data, tfidf):
        self.vocab_conditional_probabilities = []
        self.data = data[:Config.TRAIN_TIME_SAVING]
        self.validation_data = validation_data[:Config.VALIDATION_TIME_SAVING]
        self.tfidf = tfidf

        self.n = len(self.data)
        self.train_data = self.data

        self.train_docs = self.train_data[Config.BODY]
        self.validation_docs = self.validation_data[Config.BODY]

        self.train_labels = self.train_data[Config.CATEGORY].values
        self.validation_labels = self.validation_data[Config.CATEGORY].values

        self.prob_of_classes = [np.sum(self.train_labels == i) / self.n for i in range(1, Config.NUM_OF_CLASSES + 1)]

        self.vocab2id = self.tfidf.word_to_id

        self.vocab_class_occurrences = [[0] * Config.NUM_OF_CLASSES for _ in range(len(self.vocab2id))]

        for class_id, doc in zip(self.train_labels, self.train_docs):
            for vocab in self.tfidf.prepare_text(doc):
                if vocab not in self.vocab2id:
                    continue
                self.vocab_class_occurrences[self.vocab2id[vocab]][class_id - 1] += 1

        self.class_total_vocabs = [0] * Config.NUM_OF_CLASSES
        for vocab_occurrence in self.vocab_class_occurrences:
            for i, class_population in enumerate(vocab_occurrence):
                self.class_total_vocabs[i] += class_population

        self.report()

    def vocab_probabilities(self, alpha):
        V = len(self.vocab2id)
        return [[
            (vco[i] + alpha) / (self.class_total_vocabs[i] + alpha * V) for i in range(Config.NUM_OF_CLASSES)
        ] for vco in self.vocab_class_occurrences]

    def predict(self, doc):
        class_scores = [np.log(p) for p in self.prob_of_classes]
        for c in range(Config.NUM_OF_CLASSES):
            for vocab in self.tfidf.prepare_text(doc):
                if vocab not in self.vocab2id:
                    continue
                class_scores[c] += np.log(self.vocab_conditional_probabilities[self.vocab2id[vocab]][c])

        return np.argmax(class_scores) + 1

    def report(self):
        best_alpha = self.evaluate_validation_for_alphas([1, 2, 3])
        logger.info("Best value for alpha: {}".format(best_alpha))

        self.evaluate(self.train_docs, self.train_labels, best_alpha, name='Train')
        self.evaluate(self.validation_docs, self.validation_labels, best_alpha, name='Validation')

    def set_alpha(self, alpha):
        self.vocab_conditional_probabilities = self.vocab_probabilities(alpha)

    def evaluate(self, docs, labels, alpha, name):
        self.set_alpha(alpha)
        predictions = []
        true_labels = labels
        for doc in docs:
            predictions.append(self.predict(doc))

        accuracy_ = accuracy(true_labels, predictions)
        per_class_recall = recall(true_labels, predictions)
        per_class_precision = precision(true_labels, predictions)
        macro_f1 = f1score(true_labels, predictions)
        confusion_matrix_ = confusion_matrix(true_labels, predictions)

        np.set_printoptions(precision=3)
        logger.info(" On " + name + " Data")
        logger.info("alpha = {}\n"
                    "\tAccuracy = {:.3f}\n"
                    "\tRecall_per class = {}\n"
                    "\tPrecision_per class = {}\n"
                    "\tmacro_F1 = {:.3f}\n"
                    "\tconfusion matrix = {}".format(alpha, accuracy_, per_class_recall, per_class_precision,
                                                     macro_f1, confusion_matrix_))
        return accuracy_

    def evaluate_validation_for_alphas(self, alphas):
        accuracy_list = []
        for alpha in alphas:
            accuracy_list.append(self.evaluate(self.validation_docs, self.validation_labels, alpha, name='Validation'))
        return alphas[np.argmax(accuracy_list)]


if __name__ == '__main__':
    # Demo
    with open(Config.TRAIN_DATA) as training_data_file:
        training_data = json.loads(training_data_file.read())
    training_data = pnd.read_json(json.dumps(training_data))

    with open(Config.VALIDATION_DATA) as test_data_file:
        test_data = json.loads(test_data_file.read())
    testing_data = pnd.read_json(json.dumps(test_data))

    tfidf = TfIdf(training_data, 'lemmatization')
    nb_clf = NaiveBayesClassifier(training_data, testing_data, tfidf)
