from typing import List, Dict
import json
import pandas as pnd

from phase2.config import Config
from phase2.naive_bayes import NaiveBayesClassifier
from phase2.tf_idf import TfIdf

global nb_clf


def train(training_docs: List[Dict]):
    global nb_clf

    training_data = pnd.read_json(json.dumps(training_docs))

    tfidf = TfIdf(training_data)
    nb_clf = NaiveBayesClassifier(training_data, training_data, tfidf)
    nb_clf.set_alpha(alpha=1)


def classify(doc: Dict) -> int:
    global nb_clf

    return nb_clf.predict(doc[Config.BODY])
