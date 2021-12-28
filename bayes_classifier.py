import numpy as np
import cifar_common
from skimage.transform import resize
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt


def cifar10_color(X):
    Xp = np.zeros((X.shape[0], 1, 1, 3), dtype="float")

    for i in range(X.shape[0]):
        Xp[i] = resize(X[i], (1, 1, 3))
        cifar_common.print_progress(int(100 * i / X.shape[0]))

    return Xp.reshape(Xp.shape[0], 3)


# For later use
def cifar10_2x2_color(X, size=2):
    resized = np.zeros((X.shape[0], size, size, 3), dtype="float")

    for i in range(X.shape[0]):
        resized[i] = resize(X[i], (size, size, 3))
        cifar_common.print_progress(int(100 * i / X.shape[0]))

    return resized.reshape(resized.shape[0], size * size * 3)


def cifar10_naivebayes_learn(Xp, Y):
    # Init the dimensions
    mu = np.zeros((10, 3))
    sigma = np.zeros((10, 3))
    p = np.zeros((10, 1))

    for i in range(10):
        # store imgs per loop
        class_images = []

        for j in range(Xp.shape[0]):
            # Store the sample to this class if label match
            if Y[j] == i:
                class_images.append(Xp[j])
        class_images = np.asarray(class_images)

        if class_images.shape[0] != 0:
            mu[i] = np.mean(class_images, axis=0)
            sigma[i] = np.std(class_images, axis=0)
            p[i] = class_images.shape[0] / Xp.shape[0]

        cifar_common.print_progress(int(10 * i))

    return mu, sigma, p


def cifar10_classifier_naivebayes(x, mu, sigma, p):
    probabilities = []

    for i in range(10):
        # Probability for every class
        prob = norm.pdf(x[0], mu[i][0], sigma[i][0]) * norm.pdf(x[1], mu[i][1], sigma[i][1]) \
                      * norm.pdf(x[2], mu[i][2], sigma[i][2]) * p[i] / np.sum(norm.pdf(x[0], mu[0], sigma[0])
                      * norm.pdf(x[1], mu[1], sigma[1]) * norm.pdf(x[2], mu[2], sigma[2]) * p)

        probabilities.append(prob)

    max_label = np.argmax(probabilities)
    return max_label


def run_naive_bayes(tr_data, tr_labels, test_data, test_labels):
    print("\n--- Task 1: Naive bayes 1x1 ---")
    mu, sigma, p = cifar10_naivebayes_learn(tr_data, tr_labels)
    naive_bayes_pred = []

    for i in range(test_data.shape[0]):
        naive_bayes_pred.append(cifar10_classifier_naivebayes(test_data[i], mu, sigma, p))
        cifar_common.print_progress(int(100 * i / test_data.shape[0]))
    cifar_common.class_acc(naive_bayes_pred, test_labels)


def cifar10_bayes_learn(Xf, Y):
    mu = np.zeros((10, len(Xf[0])))
    sigma = np.zeros((10, len(Xf[0]), len(Xf[0])))
    p = np.zeros((10, 1))

    for i in range(10):
        # store imgs per loop
        class_images = []

        for j in range(Xf.shape[0]):
            # Store the sample to this class if label match
            if Y[j] == i:
                class_images.append(Xf[j])
        class_images = np.asarray(class_images)

        if class_images.shape[0] != 0:
            mu[i] = np.mean(class_images, axis=0)
            sigma[i] = np.cov(class_images, rowvar=0)
            p[i] = class_images.shape[0] / Xf.shape[0]

        cifar_common.print_progress(int(10 * i))

    return mu, sigma, p


def cifar10_classifier_bayes(x, mu, sigma, p):
    probabilities = []

    for i in range(10):
        # Probability for every class
        prob = multivariate_normal.pdf(x, mu[i], sigma[i]) * p[i]
        probabilities.append(prob)

    max_label = np.argmax(probabilities)
    return max_label


def run_bayes_multivariate(tr_data, tr_labels, test_data, test_labels):
    mu, sigma, p = cifar10_bayes_learn(tr_data, tr_labels)
    bayes_pred = []

    for i in range(test_data.shape[0]):
        bayes_pred.append(cifar10_classifier_bayes(test_data[i], mu, sigma, p))
        cifar_common.print_progress(int(100 * i / test_data.shape[0]))
    cifar_common.class_acc(bayes_pred, test_labels)


def main():
    # Load training data
    tr_data, tr_labels = cifar_common.load_training_data()
    test_data, test_labels = cifar_common.load_test_data()
    # Resize the data
    Xp = cifar10_color(tr_data)
    test_data = cifar10_color(test_data)

    # Naive bayes
    run_naive_bayes(Xp, tr_labels, test_data, test_labels)

    # Bayes going through 1x1 2x2 4x4...
    print("--- Task 2 & 3: Different sizes ---")
    dimension_list = [1, 2, 4, 8, 16, 32]

    for d in dimension_list:
        print("Bayes: " + str(d) + "x" + str(d))
        tr_data, tr_labels = cifar_common.load_training_data()
        test_data, test_labels = cifar_common.load_test_data()
        Xp = cifar10_2x2_color(tr_data, d)
        test_data = cifar10_2x2_color(test_data, d)
        run_bayes_multivariate(Xp, tr_labels, test_data, test_labels)


main()
