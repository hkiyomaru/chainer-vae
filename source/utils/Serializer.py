import chainer
from chainer import serializers


class Serializer:
    def __init__(self, out_model_dir):
        self.out_model_dir = out_model_dir

    def save(self, model, state, epoch):
        serializers.save_hdf5("%s/model_ie_%d.h5"%(self.out_model_dir, epoch), model)
        serializers.save_hdf5("%s/state_ie_%d.h5"%(self.out_model_dir, epoch), state)
