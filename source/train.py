import numpy as np
import random

import chainer
from chainer import cuda
from chainer import Variable
from chainer import optimizers
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence

from utils.ArgumentParser import ArgumentParser
from utils.Logger import Logger
from utils.Serializer import Serializer
from utils.ComputationalGraph import ComputationalGraph

from models.VAE import VAE

import pickle
import sys
import os
import signal


"""
Paths
"""
out_model_dir = 'out_models'

"""
Settings
"""
parser = ArgumentParser()
gpuid = parser.get_argument('gpuid')
n_epoch = parser.get_argument('epoch')
n_latent = parser.get_argument('dims')
batchsize = parser.get_argument('batchsize')

logger = Logger(n_epoch)
serializer = Serializer(out_model_dir)
computational_graph_generator = ComputationalGraph('cg')

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
Signal handler
"""
signal.signal(signal.SIGINT, logger.handler)

"""
Others
"""
if gpuid >= 0:
    xp = cuda.cupy
    cuda.get_device(gpuid).use()
else:
    xp = np

try:
    os.mkdir(out_model_dir)
except:
    pass

"""
Show information
"""
logger.show_information(gpuid, n_latent, batchsize, n_epoch)
logger.show_dataset_information(x_train, x_test)
logger.ok()


def train(model, epoch0=0):
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    for epoch in xrange(epoch0, n_epoch):
        logger.print_epoch(epoch)

        # training
        perm = np.random.permutation(N_train)
        for i in xrange(0, N_train, batchsize):
            x = Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
            mu, ln_var = model.encode(x)
            rec_loss = 0
            for l in xrange(1):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, model.decode(z, sigmoid=False)) / batchsize
            reg_loss = gaussian_kl_divergence(mu, ln_var) / batchsize
            loss = rec_loss + reg_loss
            optimizer.zero_grads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            logger.save_loss(reg_loss.data, rec_loss.data, train=True)

        # evaluation
        for i in xrange(0, N_test, batchsize):
            x = Variable(xp.asarray(x_train[i:i + batchsize]), volatile='on')
            mu, ln_var = model.encode(x)
            rec_loss = 0
            for l in xrange(1):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, model.decode(z, sigmoid=False)) / batchsize
            reg_loss = gaussian_kl_divergence(mu, ln_var) / batchsize
            loss = rec_loss + reg_loss
            logger.save_loss(reg_loss.data, rec_loss.data, train=False)

        logger.epoch_end()

    logger.terminate()
    serializer.save(model, optimizer, epoch+1)

    # everything works well
    return 0

def main():
    vae = VAE(n_in=1, n_latent=n_latent, n_h=64)
    if gpuid >= 0: vae.to_gpu()
    return train(vae)

"""
Entry point
"""
if __name__ == '__main__':
    sys.exit(main())
