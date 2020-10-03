import numpy as np
import random
import _pickle as cPickle
import gzip
import os
import sys

def load_mnist(path, num_training=50000, num_test=10000, cnn=False, one_hot=True):
    f = gzip.open(os.path.join(path, "mnist.pkl.gz"), 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='iso-8859-1')
    f.close()
    X_train, y_train = training_data
    X_val, y_val = validation_data
    X_test, y_test = test_data

    if cnn:
        shape = (-1, 1, 28, 28)
        X_train = X_train.reshape(shape)
        X_test = X_test.reshape(shape)
        X_val = X_val.reshape(shape)

    if one_hot:
        y_train = one_hot_encode(y_train, 10)
        y_test = one_hot_encode(y_test, 10)
        y_val = one_hot_encode(y_val, 10)
    
    (X_train, y_train)= np.concatenate([X_train, X_val])[:num_training], np.concatenate([y_train, y_val]) [:num_training]
    (X_test, y_test) = X_test[:num_test], y_test[:num_test]
    return (X_train, y_train), (X_test, y_test)

def load_cifar10(path, num_training=1000, num_test=1000):
    Xs, ys = [], []
    for batch in range(1, 6):
        f = open(os.path.join(path, "data_batch_{0}".format(batch)), 'rb')
        data = cPickle.load(f, encoding='iso-8859-1')
        f.close()
        X = data["data"].reshape(10000, 3, 32, 32).astype("float64")
        y = np.array(data["labels"])
        Xs.append(X)
        ys.append(y)
    f = open(os.path.join(CIFAR10_PATH, "test_batch"), 'rb')
    data = cPickle.load(f, encoding='iso-8859-1')
    f.close()
    X_train, y_train = np.concatenate(Xs), np.concatenate(ys)
    X_test = data["data"].reshape(10000, 3, 32, 32).astype("float")
    y_test = np.array(data["labels"])
    X_train, y_train = X_train[range(num_training)], y_train[range(num_training)]
    X_test, y_test = X_test[range(num_test)], y_test[range(num_test)]
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train)
    X_train /= 255.0
    X_test /= 255.0
    return (X_train, y_train), (X_test, y_test)

def one_hot_encode(y, num_class):
    m = y.shape[0]
    onehot = np.zeros((m, num_class), dtype="int32")
    for i in range(m):
        idx = y[i]
        onehot[i][idx] = 1
    return onehot

def shuffle_data(X, Y):
    n = X.shape[1]
    # Pour minimiser le "overfitting", on mélange les données
    data = np.concatenate([X, Y], axis=1)
    np.random.shuffle(data)
    return np.split(data, [n], axis=1)
