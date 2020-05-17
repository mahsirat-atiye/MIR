import json

import numpy as np
import pandas as pnd

from sklearn.svm import LinearSVC

from phase2.config import Config
from phase2.evaluator import accuracy, recall, precision, f1score, confusion_matrix
from phase2.tf_idf import TfIdf

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SVM")


class SVMClassifier:
    def __init__(self, train_data, validation_data, tfidf):
        self.train_data = train_data[:Config.TRAIN_TIME_SAVING]
        self.validation_data = validation_data[:Config.VALIDATION_TIME_SAVING]
        self.tfidf = tfidf

        self.train_labels = self.train_data[Config.CATEGORY].values
        self.validation_labels = self.validation_data[Config.CATEGORY].values

        self.train_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.train_data)
        self.validation_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.validation_data)

        self.report()

    def predict(self, doc):
        doc_vector = self.tfidf.get_tfidf_vector(doc)
        return self.model.predict([doc_vector])[0]

    def report(self):
        self.model, best_C = self.report_on_validation([0.5, 1.0, 1.5])

        self.evaluate(self.train_docs_matrix, self.train_labels, best_C, 'Train', model=self.model)
        self.evaluate(self.validation_docs_matrix, self.validation_labels, best_C, 'Validation', model=self.model)

    def build_model(self, C):
        model = LinearSVC(C=C)
        model.fit(self.train_docs_matrix, self.train_labels)
        return model

    def evaluate(self, docs_matrix, labels, C, name, model=None):
        if not model:
            model = self.build_model(C)
        predictions = model.predict(docs_matrix).astype(int)
        true_labels = labels.astype(int)

        accuracy_ = accuracy(true_labels, predictions)
        per_class_recall = recall(true_labels, predictions)
        per_class_precision = precision(true_labels, predictions)
        macro_f1 = f1score(true_labels, predictions)
        confusion_matrix_ = confusion_matrix(true_labels, predictions)
        np.set_printoptions(precision=3)
        logger.info(" On " + name + " Data")
        logger.info("C = {}:\n"
                    "\tAccuracy = {:.3f}\n"
                    "\tRecall_per class = {}\n"
                    "\tPrecision_per class = {}\n"
                    "\tmacro_F1 = {}\n"
                    "\tconfusion matrix = {}\n".format(C, accuracy_,
                                                       per_class_recall, per_class_precision,
                                                       macro_f1, confusion_matrix_))

        return model, accuracy_

    def report_on_validation(self, Cs):
        accuracy_list = []
        model_list = []
        for C in Cs:
            model, accuracy_ = self.evaluate(self.validation_docs_matrix, self.validation_labels, C,
                                             name='Validation')
            accuracy_list.append(accuracy_)
            model_list.append(model)

        best_idx = np.argmax(accuracy_list)
        return model_list[best_idx], Cs[best_idx]


if __name__ == '__main__':
    with open(Config.TRAIN_DATA) as training_data_file:
        training_data = json.loads(training_data_file.read())
    training_data = pnd.read_json(json.dumps(training_data))

    with open(Config.VALIDATION_DATA) as test_data_file:
        test_data = json.loads(test_data_file.read())
    testing_data = pnd.read_json(json.dumps(test_data))

    tfidf = TfIdf(training_data)
    svm_clf = SVMClassifier(training_data, testing_data, tfidf)
