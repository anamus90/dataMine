__author__ = "Sumana Srivatsa"
__email__ = "sumana.srivatsa@bsse.ethz.ch"

import numpy as np
from numpy import asarray

# Number of random Fourier features (RFF)
mFeats = 8000
# Gamma for RFF with RBF kernel
gamma = 5

# Random Fourier Feature transformations
# Sample w from from p(w) and b from unif[0,2*pi] and transform z = sqrt(2/m)cos(wx+b)
np.random.seed(456)
w = np.transpose(np.sqrt(2 * gamma) * np.random.normal(size=(mFeats, 400)))
b = 2 * np.pi * np.random.rand(mFeats)

def transform(X):

    transformed = np.sqrt(2.0 / mFeats) * np.cos(np.dot(X, w) + b)
    # makes the data more sparse
    transformed = (transformed - np.mean(transformed, 0)) / np.std(transformed, 0)

    return transformed

# Calculate the gradient of the hinge loss function
def gradient_loss_function(w, y, X, Xy):
    gradient = np.zeros(X.shape[1])

    for i, (y_i, x_i) in enumerate(zip(y, X)):
        if y_i * np.dot(x_i, w) < 1.:
            gradient -= Xy[i]

    return gradient

# Getting weight vector using adagrad
def ADA_grad(labels, features, w0, gradient_loss_function,stepSize=5e-3, fudgeFactor=1e-6, maxIter=1000, thresh=1e-4):
    gradChain = np.zeros(w0.shape[0])
    w = w0
    feats_labels = features * labels[:, None]
    t = 0

    while t <= maxIter:
        wOld = np.copy(w)
        t = t + 1
        gradient = gradient_loss_function(w, labels, features, feats_labels)
        gradChain += np.square(gradient)
        adjustGrad = gradient / (fudgeFactor + np.sqrt(gradChain))
        w = w - (stepSize * adjustGrad)
        if np.sum(np.abs(w - wOld)) < thresh:
            break

    return w

# Mapper function
# Get the labels and features for each mini-batch and returns weights for each mini-batch after SGD
def mapper(key, value):
    labels = np.zeros(len(value))
    features = np.zeros([len(value), len(value[0].split()) - 1])

    for i in range(len(value)):
        tempArr = value[i].split()
        labels[i] = tempArr[0]
        features[i] = tempArr[1:]

    features = transform(features)
    w0 = np.zeros(mFeats)

    w = ADA_grad(labels,features,w0,gradient_loss_function)

    yield 0, w


# The reducer function returns the average value
def reducer(key, values):
    yield np.sum(asarray(values), axis=0) / len(values)