import numpy as np
import cifar_common
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Flatten, Input
import keras.optimizers


# Convert int labels to one-hot vectors
def to_categorical(labels):
    one_hots = []

    for label_num in labels:
        one_hot = []
        for i in range(0, 10):
            if i != label_num:
                one_hot.append(0)
            else:
                one_hot.append(1)

        one_hots.append(one_hot)

    return np.array(one_hots)


# Convolutional model
def improved_model():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model

    return model


def basic_model():
    model = Sequential()
    # Input layer, images
    model.add(Input(shape=(32, 32, 3)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="sigmoid"))

    return model


def load_model(model_type='basic'):

    if model_type == 'conv':
        return improved_model()

    return basic_model()


def main():
    tr_data, tr_labels = cifar_common.load_training_data()
    test_data, test_labels = cifar_common.load_test_data()
    # Scale the data as we are using TensorFlow
    tr_data = tr_data / 255.0
    test_data = test_data / 255.0
    # Convert to categorical (one-hot vectors)
    tr_labels = to_categorical(tr_labels)
    test_labels = to_categorical(test_labels)

    model = load_model()
    # Optimize & compile
    keras.optimizers.SGD(lr=0.001, momentum=0.9)
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

    num_of_epochs = 30
    model.fit(tr_data, tr_labels, epochs=num_of_epochs, verbose=1)
    # Run the evaluation & print results
    _, acc = model.evaluate(test_data, test_labels, verbose=0)
    print('> %.3f' % (acc * 100.0))


main()
