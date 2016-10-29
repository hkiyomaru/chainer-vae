import numpy as np
from matplotlib import pyplot as plt

import sys


class Logger:
    def __init__(self, n_epoch):
        self.n_epoch = n_epoch
        self.prepare_graph()

    def prepare_graph(self):
        self.x = range(self.n_epoch)
        self.y = np.zeros(self.n_epoch)

    def show_dataset_information(self, detaset):
        pass

    def print_epoch(self, epoch):
        self.epoch = epoch
        self.epoch_loss = np.float32(0)
        print "epoch:", epoch

    def save_loss(self, loss):
        self.current_loss = loss
        self.epoch_loss += self.current_loss

    def epoch_end(self):
        self.y[self.epoch] = self.epoch_loss

    def show_loss_graph(self):
        plt.title("Loss value transition")
        plt.plot(self.x, self.y)
        plt.show()

    def handler(self, signum, frame):
        self.show_loss_graph()
        sys.exit(1)
