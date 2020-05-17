import json

import numpy as np
import pandas as pnd

from sklearn.ensemble import RandomForestClassifier as RFC

from phase2.config import Config
from phase2.evaluator import accuracy, recall, precision, f1score, confusion_matrix
from phase2.tf_idf import TfIdf

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Random Forest")


class RandomForestClassifier:
    def __init__(self, train_data, validation_data, tfidf):
        self.train_data = train_data[:Config.TRAIN_TIME_SAVING]
        self.validation_data = validation_data[:Config.VALIDATION_TIME_SAVING]
        self.tfidf = tfidf

        self.train_labels = self.train_data[Config.CATEGORY].values
        self.test_labels = self.validation_data[Config.CATEGORY].values

        self.train_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.train_data)
        self.test_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.validation_data)

        # if predict be called before evaluation
        self.model = RFC(max_features=10, max_depth=40)
        self.model.fit(self.train_docs_matrix, self.train_labels)

        self.report()

    def predict(self, doc):
        doc_vector = self.tfidf.get_tfidf_vector(doc)
        return self.model.predict([doc_vector])[0]

    def report(self):
        best_max_depthـbest_max_feature, best_model = self.evaluate_validation_for_max_depth_max_features([(40, 10), (60, 15)])
        best_max_depth, best_max_feature = best_max_depthـbest_max_feature
        logger.info("Best value for max depth: {} , max feature: {}".format(best_max_depth, best_max_feature))
        self.evaluate(self.train_docs_matrix, self.train_labels, best_max_depth, best_max_feature, 'Train',
                      model=best_model)
        self.evaluate(self.test_docs_matrix, self.test_labels, best_max_depth, best_max_feature, 'Validation',
                      model=best_model)

    def evaluate(self, docs_matrix, labels, max_depth, max_feature, name, model=None):
        if model is None:
            model = RFC(max_features=max_feature, max_depth=max_depth)
            model.fit(self.train_docs_matrix, self.train_labels)
        predictions = model.predict(docs_matrix)
        true_labels = labels

        accuracy_ = accuracy(true_labels, predictions)
        per_class_recall = recall(true_labels, predictions)
        per_class_precision = precision(true_labels, predictions)
        macro_f1 = f1score(true_labels, predictions)
        confusion_matrix_ = confusion_matrix(true_labels, predictions)
        np.set_printoptions(precision=3)
        logger.info(" On " + name + " Data")
        logger.info("\tAccuracy = {:.3f}\n"
                    "\tRecall_per class = {}\n"
                    "\tPrecision_per class = {}\n"
                    "\tmacro_F1 = {:.3f}\n"
                    "\tconfusion matrix = {}".format(accuracy_, per_class_recall, per_class_precision,
                                                     macro_f1, confusion_matrix_))
        return accuracy_, model

    def evaluate_validation_for_max_depth_max_features(self, depths_features):
        accuracy_list = []
        models_list = []
        for max_depth, max_feature in depths_features:
            accuracy_, model = self.evaluate(self.test_docs_matrix, self.test_labels, max_depth, max_feature, 'Test')
            accuracy_list.append(accuracy_)
            models_list.append(model)
        return depths_features[np.argmax(accuracy_list)], models_list[np.argmax(accuracy_list)]


if __name__ == '__main__':
    # Demo
    with open(Config.TRAIN_DATA) as training_data_file:
        training_data = json.loads(training_data_file.read())
    training_data = pnd.read_json(json.dumps(training_data))

    with open(Config.VALIDATION_DATA) as test_data_file:
        test_data = json.loads(test_data_file.read())
    testing_data = pnd.read_json(json.dumps(test_data))

    tfidf = TfIdf(training_data)

    rf_clf = RandomForestClassifier(training_data, testing_data, tfidf)
