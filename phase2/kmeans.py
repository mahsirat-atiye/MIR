import json
from collections import defaultdict

import numpy as np
import pandas as pnd

from phase2.config import Config
from phase2.tf_idf import TfIdf

from sklearn.manifold import TSNE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns



class K_Means:
    def __init__(self, train_data, validation_data, tfidf, k=4, tol=0.001, max_iter=300):
        self.train_data = train_data[:Config.TRAIN_TIME_SAVING]
        self.validation_data = validation_data[:Config.VALIDATION_TIME_SAVING]
        self.tfidf = tfidf
        self.k = k
        self.tol = tol
        self.max_iter = max_iter




        self.train_labels = self.train_data[Config.CATEGORY].values
        self.predicted_train_labels = [0] * len(self.train_labels)
        self.validation_labels = self.validation_data[Config.CATEGORY].values

        self.train_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.train_data)
        self.validation_docs_matrix = self.tfidf.get_tf_idf_vector_for_docs(self.validation_data)
        self.centroids_matrix = self.train_docs_matrix[:self.k]

        self.report()

    def euclidean_distance(self, vec_1, vec_2):
        diff = vec_1 - vec_2
        result = np.linalg.norm(diff)
        # dist = diff * diff.transpose()
        # sum_dist = sum(dist)
        # result = np.sqrt(sum_dist)
        return (result)

    def pairwise_distances(self, new_docs_vector, old_docs_vector):

        D = [[0.0] * old_docs_vector.shape[0] for _ in range((new_docs_vector.shape[0]))]
        for i, new_doc_vector in enumerate(new_docs_vector):
            for j, old_doc_vector in enumerate(old_docs_vector):
                D[i][j] = self.euclidean_distance(new_doc_vector, old_doc_vector)
        return D

    def check_end_of_fitness(self, old_centroid_matrix, new_centroid_matrix):
        optimized = True
        for i in range(self.k):
            current_centroid = new_centroid_matrix[i]
            original_centroid = old_centroid_matrix[i]
            change_in_centroid = (current_centroid - original_centroid) * 100
            for i, dimension in enumerate(change_in_centroid):
                if original_centroid[i] == 0:
                    if current_centroid[i] == 0:
                        change_in_centroid[i] = 0
                    else:
                        change_in_centroid[i] = 1

                else:
                    change_in_centroid[i] = dimension / original_centroid[i]

            whole_change = (np.sum(change_in_centroid))
            print(whole_change)
            if whole_change > self.tol:
                optimized = False
            return optimized

    def fit(self):

        for i in range(self.max_iter):
            self.classifications = defaultdict(lambda: [])
            distances = self.pairwise_distances(self.train_docs_matrix, self.centroids_matrix)
            for i, d in enumerate(distances):
                nearest_centroid = np.argmin(d)
                self.predicted_train_labels[i] = nearest_centroid
                self.classifications[nearest_centroid].append(self.train_docs_matrix[i])

            prev_centroids = self.centroids_matrix

            for classification in self.classifications:
                self.centroids_matrix[classification] = np.average(self.classifications[classification], axis=0)

            if self.check_end_of_fitness(prev_centroids, self.centroids_matrix):
                return

    def predict(self, doc):
        doc_vector = self.tfidf.get_tfidf_vector(doc)
        distance = self.pairwise_distances(doc_vector, self.centroids_matrix)
        nearest_centroid = np.argmin(distance)
        return nearest_centroid

    def report(self):
        self.fit()
        dim_reduction = TSNE(n_components=2, n_iter=250)
        transformed_features = dim_reduction.fit_transform(self.train_docs_matrix)

        num_classes = len(np.unique(self.train_labels))
        palette = np.array(sns.color_palette("hls", num_classes))
        c_ = ['r', 'b', 'green', 'orange']
        colors = [c_[pred] for pred in self.predicted_train_labels]

        plt.scatter(x=transformed_features[:, 0], y=transformed_features[:, 1], c=colors, lw=1, s=4)
        plt.show()


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
    km_cstr = K_Means(training_data, testing_data, tfidf, k=4, tol=0.001, max_iter=30)

