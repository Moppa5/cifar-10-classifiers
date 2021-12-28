import pickle
import numpy as np
import random

# Specify your CIFAR-10 folder path
cifar_folder_path = "C:/Users/sants/Downloads/cifar-10-batches-py/"


# Load the CIFAR-10 data
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


# Print progress of the classification process
def print_progress(prog):
    print('\r[{0}] {1}%'.format('#' * (prog // 10), prog), end="")


# Load CIFAR-10 test data
def load_test_data():
    datadict = unpickle(cifar_folder_path + 'test_batch')
    test_data = datadict["data"]
    test_labels = datadict["labels"]
    # Convert, transpose etc.
    test_data = test_data.reshape(len(test_data), 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    test_labels = np.array(test_labels)

    return test_data, test_labels


# Load CIFAR-10 training data
def load_training_data():
    tr_data = None
    tr_labels = []
    # Load the data_batch_N files
    for i in range(1, 6):
        tr_data_dict = unpickle(cifar_folder_path + 'data_batch_{}'.format(i))
        if i == 1:
            tr_data = tr_data_dict["data"]
        else:
            tr_data = np.vstack((tr_data, tr_data_dict["data"]))

        tr_labels += tr_data_dict["labels"]

    tr_data = tr_data.reshape(len(tr_data), 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    tr_labels = np.array(tr_labels)

    return tr_data, tr_labels


# Calculate the classification accuracy
def class_acc(pred, gt):
    classification_count = len(pred)
    classification_err = 0

    for index in range(0, classification_count):
        if pred[index] != gt[index]:
            classification_err += 1

    accuracy_err = classification_err / classification_count
    accuracy = 1.0 - accuracy_err
    print("\nClassification accuracy for the given set: {:3f}".format(accuracy))


# Random classifier
def random_classifier(x):
    random_label = random.randint(0, 9)
    return random_label
