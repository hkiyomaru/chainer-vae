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

    def get_loss_function(self, C=1.0, k=1, train=True):
        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu.data)

            rec_loss = 0
            for l in xrange(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False)) / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss + C * gaussian_kl_divergence(mu, ln_var) / batchsize
        return lf
