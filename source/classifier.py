import numpy as np
import random

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import chainer
from chainer import Variable

from utils.ArgumentParser import ArgumentParser
from utils.Serializer import Serializer
from models.VAE import VAE

import pickle
import sys


"""
Paths
"""
out_model_dir = 'out_models'

"""
Settings
"""
serializer = Serializer(out_model_dir)
parser = ArgumentParser()
gpuid = parser.get_argument('gpuid')
n_epoch = parser.get_argument('epoch')
n_latent = parser.get_argument('dims')
batchsize = parser.get_argument('batchsize')

"""
Data loading
"""
height = pickle.load(open('height.pkl'))
height['data'] = height['data'].astype(np.float32)
height['data'] /= np.max(height['data'])
height['target'] = height['target'].astype(np.int32)

N_train = 800
x_train, x_test = np.split(height['data'],   [N_train])
y_train, y_test = np.split(height['target'], [N_train])
N_test = y_test.size

"""
Others
"""
if gpuid >= 0:
    xp = cuda.cupy
    cuda.get_device(gpuid).use()
else:
    xp = np

"""
SVM
"""
estimator = LinearSVC(C=100.0)


def encode(x, y, model):
    size = len(x)
    encoded_vector = np.zeros((size, n_latent*2+1))
    for i in xrange(0, size, batchsize):
        _x = Variable(xp.asarray(x[i:i + batchsize]))
        _y = np.asarray(y[i:i + batchsize])
        mu, ln_var = model.encode(_x)
        mu = mu.data
        ln_var = ln_var.data
        _x = np.c_[mu, ln_var, _y]
        encoded_vector[i:i + batchsize] = _x
    return encoded_vector

def train_svm(features, targets):
    estimator.fit(features, targets)

def predict(features):
    return estimator.predict(features)

def evaluation(targets, predictions):
    print "# Evaluation"
    print " - Male: 0"
    print " - Female: 1"
    print classification_report(targets, predictions)


def main():
    # load trained model
    vae = VAE(n_in=1, n_latent=n_latent, n_h=64)
    serializer.load(vae, n_epoch)

    # encode
    encoded_train = encode(x_train, y_train, vae)
    encoded_test = encode(x_test, y_test, vae)
    train_features = encoded_train[:,:-1]
    train_targets = encoded_train[:,-1].astype(np.int32)
    test_features = encoded_test[:,:-1]
    test_targets = encoded_test[:,-1].astype(np.int32)

    # train SVC
    train_svm(train_features, train_targets)

    # test
    results = predict(test_features)

    # evaluation
    evaluation(test_targets, results)

    return 0


# Entry point
if __name__ == '__main__':
    sys.exit(main())
