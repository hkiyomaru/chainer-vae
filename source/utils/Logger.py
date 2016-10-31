import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import sys


class Logger:
    def __init__(self, epoch):
        self.n_epoch = epoch
        self.prepare_graph()
        self.current_loss = [0, 0]
        self.current_rec_loss = [0, 0]

    def prepare_graph(self):
        self.y_train = np.zeros(self.n_epoch)
        self.y_test = np.zeros(self.n_epoch)
        self.rec_y_train = np.zeros(self.n_epoch)
        self.rec_y_test = np.zeros(self.n_epoch)

    def show_information(self, gpuid, dims, minibatch_size, epoch):
        print "GPU:", gpuid
        print "# latent varable dimention:", dims
        print "# minibatch-size:", minibatch_size
        print "# epoch:", epoch

    def show_dataset_information(self, train, test):
        train_size = len(train)
        test_size = len(test)
        dims = len(train[0])
        print "DATASET"
        print "#", dims, "parameters"
        print "# TRAIN:", train_size, "samples"
        print "# TEST:", test_size, "samples"

    def ok(self):
        print "\nREADY.\n"

    def print_epoch(self, epoch):
        self.epoch = epoch
        self.epoch_loss = [0, 0]
        self.epoch_rec_loss = [0, 0]
        print "epoch:", epoch

    def save_loss(self, loss, rec_loss, train=True):
        if train: index = 0
        else: index = 1
        self.current_loss[index] = loss
        self.current_rec_loss[index] = rec_loss
        self.epoch_loss[index] += self.current_loss[index]
        self.epoch_rec_loss[index] += self.current_rec_loss[index]

    def epoch_end(self):
        self.y_train[self.epoch] = self.epoch_loss[0]
        self.rec_y_train[self.epoch] = self.epoch_rec_loss[0]
        self.y_test[self.epoch] = self.epoch_loss[1]
        self.rec_y_test[self.epoch] = self.epoch_rec_loss[1]

        print "TRAIN"
        print "# mean loss:", self.epoch_loss[0]
        print "# mean reconstruction loss:", self.epoch_rec_loss[0]
        print "TEST"
        print "# mean loss:", self.epoch_loss[1]
        print "# mean reconstruction loss:", self.epoch_rec_loss[1]

    def terminate(self):
        print "\nDONE."
        self.show_loss_graph()

    def show_loss_graph(self):
        y = np.asarray([self.y_train, self.y_test, self.rec_y_train, self.rec_y_test])
        y = y.T
        df = pd.DataFrame(y, columns=['y_train', 'y_test', 'rec_t_train', 'rec_y_test'])
        df.plot(title='loss value transition')
        plt.show()

    def handler(self, signum, frame):
        print "Keyboard interrupt."
        self.show_loss_graph()
        sys.exit(1)
