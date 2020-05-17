from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# from sklearn.metrics import confusion_matrix
import numpy as np

from phase2.config import Config

K = Config.NUM_OF_CLASSES


def confusion_matrix(true_labels, predicted_labels):

    result = np.zeros((K, K))
    for i, true_class in enumerate(true_labels):
        result[true_class - 1][predicted_labels[i] - 1] += 1

    return result


def accuracy(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    true_predictions = 0
    for i in range(K):
        true_predictions += cm[i][i]
    sum = 0
    for i in range(K):
        for j in range(K):
            sum += cm[i][j]

    return true_predictions / sum


def recall(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    recall_per_class = [0.0] * K

    for i in range(K):
        sum = 0
        for j in range(K):
            sum += cm[i][j]
        if sum == 0:
            recall_per_class[i] = 0
        else:
            recall_per_class[i] = cm[i][i] / sum
    return recall_per_class


def precision(true_labels, predicted_labels):
    cm = confusion_matrix(true_labels, predicted_labels)
    precision_per_class = [0.0] * K

    for j in range(K):
        sum = 0
        for i in range(K):
            sum += cm[i][j]
        if sum == 0:
            precision_per_class[j] = 0
        else:
            precision_per_class[j] = cm[j][j] / sum
    return precision_per_class


def f1score(true_labels, predicted_labels):
    p = precision(true_labels, predicted_labels)
    r = recall(true_labels, predicted_labels)
    f1_per_class = [0.0] * K
    for i in range(len(f1_per_class)):
        if (p[i] + r[i]) == 0:
            f1_per_class[i] = 0
        else:
            f1_per_class[i] = 2 * p[i] * r[i] / (p[i] + r[i])

    return sum(f1_per_class) / len(f1_per_class)


if __name__ == '__main__':
    # Demo
    y_true = [3, 1, 3, 3, 1, 2]
    y_pred = [1, 1, 3, 3, 1, 3]
    print(confusion_matrix(y_true, y_pred))
    print("Acc:", accuracy(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))
    print("Recall: ", recall(y_true, y_pred))
    print(recall_score(y_true, y_pred, average=None))
    print("Precision: ", precision(y_true, y_pred))
    print(precision_score(y_true, y_pred, average=None))
    print("Macro f1", f1score(y_true, y_pred))
    print(f1_score(y_true, y_pred, average='macro'))
