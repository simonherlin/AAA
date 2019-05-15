import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from keras import layers
from keras import backend as K
from keras.models import Sequential
import matplotlib.pyplot as plt


def format_data(df):
    """
    Return only data serve for working and display important information
    :param df:
    :return:  4 array -> 2 trainning data, 2 label data
    """
    nb_classe = 2
    count_row = df.shape[0]
    count_col = df.shape[1]

    count_neg = df[df.label == 'neg']
    count_pos = df[df.label == 'pos']
    print('nb rows ', count_row)
    print('nb neg ', len(count_neg))
    print('nb pos ', len(count_pos))

    t = df[df.label != 'unsup']
    t = t.drop(t.columns[0], axis=1)
    del t['file']
    del t['type']
    print('nb total pos neg ', len(t))

    mapping = {'neg': 0, 'pos': 1}
    t = t.replace({'label': mapping})
    t = shuffle(t)
    sentences = t['review'].values
    y = t['label'].values
    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y)
    print('train ', len(sentences_train))
    print('test ', len(sentences_test))

    print('nb neg train', np.count_nonzero(y_train == 0))
    print('nb pos train', np.count_nonzero(y_train == 1))

    print('nb neg test', np.count_nonzero(y_test == 0))
    print('nb pos test', np.count_nonzero(y_test == 1))
    return sentences_test, sentences_train, y_test, y_train


def vect_data(train, test):
    """
    Data transform for working
    :param train:
    :param test:
    :return: 2 vector train and test
    """
    vect = CountVectorizer(encoding='iso-8859-1')
    vect.fit(train)
    # print(vect.vocabulary_)
    v_train = vect.transform(train)
    v_test = vect.transform(test)
    print('v_train shape ', v_train.shape)
    print('v_test shape ', v_test.shape)
    return v_test, v_train


def neural_network(v_train, y_train, v_test, y_test):
    """
    Keras neural network
    :param v_train:
    :param y_train:
    :param v_test:
    :param y_test:
    :return: history result
    """
    input_dim = v_train.shape[1]
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(v_train, y_train,
                         epochs=10,
                         verbose=True,
                         validation_data=(v_test, y_test),
                         batch_size=10)
    loss, accuracy = model.evaluate(v_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(v_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    return history


def plot_history(history):
    """
    plot history result
    Display result
    :param history:
    :return:
    """
    plt.style.use('ggplot')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    K.tensorflow_backend._get_available_gpus()
    df = pd.read_csv("./films.csv", encoding="ISO-8859-1")
    x_test, x_train, y_test, y_train = format_data(df)
    v_test, v_train = vect_data(x_train, x_test)
    # print(v_test)
    history = neural_network(v_train, y_train, v_test, y_test)
    plot_history(history)
