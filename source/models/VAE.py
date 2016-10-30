import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L


class VAE(chainer.Chain):
    def __init__(self, n_in, n_latent, n_h):
        super(VAE, self).__init__(
            # encoder
            e1 = L.Linear(n_in, n_h),
            e2_mu = L.Linear(n_h, n_latent),
            e2_ln_var = L.Linear(n_h, n_latent),
            # decoder
            d1 = L.Linear(n_latent, n_h),
            d2 = L.Linear(n_h, n_in),
        )

    def __call__(self, x, sigmoid=True):
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.e1(x))
        mu = self.e2_mu(h1)
        ln_var = self.e2_ln_var(h1)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.d1(z))
        h2 = self.d2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2
