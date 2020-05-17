import logging
import json
import numpy as np
import pandas as pnd

from phase2.config import Config
from phase2.evaluator import accuracy, recall, precision, confusion_matrix, f1score
from phase2.tf_idf import TfIdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KNN")


def most_common_class(list_of_K_classes):
    return max(set(list_of_K_classes), key=list_of_K_classes.count)


class KNNClassifier:
    def __init__(self, train_data, validation_data, tfidf, distance_mode='euclidean_distance'):
        self.train_data = train_data[:Config.TRAIN_TIME_SAVING]
        self.validation_data = validation_data[:Config.VALIDATION_TIME_SAVING]
        self.tfidf = tfidf
        self.distance_mode = distance_mode

        self.train_labels = self.train_data[Config.CATEGORY].values
        self.validation_labels = self.validation_data[Config.CATEGORY].values

        self.train_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.train_data)
        self.validation_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.validation_data)

        self.report()

    def report(self):
        best_k = self.evaluate_validation_for_ks([1, 3, 5])
        logger.info("Best value for k: {}".format(best_k))
        self.evaluate(self.train_docs_matrix, self.train_labels, best_k, name='Train')
        self.evaluate(self.validation_docs_matrix, self.validation_labels, best_k, name='Validation')

    def predict(self, doc, k):

        doc_vector = self.tfidf.get_tfidf_vector(doc)
        distances = self.pairwise_distances(self.train_docs_matrix, doc_vector)
        if self.distance_mode == 'euclidean_distance':
            nearest_docs = np.argsort(distances)[:k]
        else:
            nearest_docs = np.argsort(distances)[-k:]
        nearest_docs_classes = self.train_labels[nearest_docs].tolist()
        return most_common_class(nearest_docs_classes)

    def euclidean_distance(self, vec_1, vec_2):
        # return type sparse matrix
        diff = vec_1 - vec_2
        # dist = diff * diff.transpose()
        # sum_dist = sum(dist)
        # result = np.sqrt(sum_dist)
        result = np.linalg.norm(diff)
        return (result)

    def norm(self, vec):
        # return type float
        # sum_vec2 = sum(vec * vec.transpose())
        result = np.linalg.norm(vec)
        return result

    def cosine_similarity(self, vec_1, vec_2):
        mul = vec_1 * vec_2.transpose()
        result = np.linalg.norm(mul)
        #
        # sum_vec1_mul_vec2 = (sum(vec_1 * vec_2.transpose()))
        # result = sum_vec1_mul_vec2 / (self.norm(vec_1) * (self.norm(vec_2)))
        return (result)

    def pairwise_distances(self, new_docs_vector, old_docs_vector):

        if self.distance_mode == 'euclidean_distance':
            f = self.euclidean_distance
        else:
            f = self.cosine_similarity

        D = [[0.0] * old_docs_vector.shape[0] for _ in range((new_docs_vector.shape[0]))]
        for i, new_doc_vector in enumerate(new_docs_vector):
            for j, old_doc_vector in enumerate(old_docs_vector):
                D[i][j] = f(new_doc_vector, old_doc_vector)
        return D

    def evaluate(self, docs, labels, k, name):
        distances = self.pairwise_distances(docs, self.train_docs_matrix)
        predictions = []
        true_labels = []


        for i, d in enumerate(distances):
            if self.distance_mode == 'euclidean_distance':
                nearest_docs = np.argsort(d)[:k]
            else:
                nearest_docs = np.argsort(d)[-k:]

            nearest_docs_classes = self.train_labels[nearest_docs].tolist()
            predicted_class = most_common_class(nearest_docs_classes)
            true_class = labels[i]
            predictions.append(predicted_class)
            true_labels.append(true_class)

        accuracy_ = accuracy(true_labels, predictions)
        per_class_recall = recall(true_labels, predictions)
        per_class_precision = precision(true_labels, predictions)
        confusion_matrix_ = confusion_matrix(true_labels, predictions)
        macro_f1 = f1score(true_labels, predictions)
        np.set_printoptions(precision=3)
        logger.info("On " + name + " Data")
        logger.info("K = {}:\n"
                    "\tAccuracy = {:.3f}\n"
                    "\tRecall_per class = {}\n"
                    "\tPrecision_per class = {}\n"
                    "\tmacro_F1 = {:.3f}\n"
                    "\tconfusion matrix = {}".format(k, accuracy_, per_class_recall, per_class_precision,
                                                     macro_f1, confusion_matrix_))
        return accuracy_

    def evaluate_validation_for_ks(self, ks):
        accuracy_list = []
        for k in ks:
            accuracy_list.append(self.evaluate(self.validation_docs_matrix, self.validation_labels, k,
                                               name="Validation"))
        return ks[np.argmax(accuracy_list)]


if __name__ == '__main__':
    # Demo
    # reading data
    with open(Config.TRAIN_DATA) as training_data_file:
        training_data = json.loads(training_data_file.read())
    training_data = pnd.read_json(json.dumps(training_data))

    with open(Config.VALIDATION_DATA) as test_data_file:
        test_data = json.loads(test_data_file.read())
    testing_data = pnd.read_json(json.dumps(test_data))

    tfidf = TfIdf(training_data)
    knn_clf = KNNClassifier(training_data, testing_data, tfidf)

