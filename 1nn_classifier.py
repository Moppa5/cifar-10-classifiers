import cifar_common
import numpy as np


# Function/wrapper for running 1-NN classifier
def run_1nn_classifier():
    trdata, trlabels = cifar_common.load_training_data()
    test_data, test_labels = cifar_common.load_test_data()
    pred_labels = []

    for i in range(test_data.shape[0]):
        pred_labels.append(cifar10_classifier_1nn(test_data[i], trdata, trlabels))
        cifar_common.print_progress(int(100 * i / test_data.shape[0]))

    cifar_common.class_acc(pred_labels, test_labels)


# 1-NN classifier
def cifar10_classifier_1nn(x, trdata, trlabels):
    distances = []
    for i in range(trdata.shape[0]):
        # Manhattan distance
        distances.append(np.sum(np.abs(trdata[i]-x)))

    minimum = np.argmin(distances)
    return trlabels[minimum]


def main():
    print("### 1-NN Classifier ###")
    run_1nn_classifier()


main()