import matplotlib.pyplot as plt
import numpy as np


class N:

    def __init__(self, mean, var):
        self.mu = mean
        self.sig = np.sqrt(var)

    def sample(self, size):
        return self.sig * np.random.randn(size) + self.mu


class LearnableUniNormal:

    def __init__(self):
        self.mu = 0.0
        self.log_sig = 0.0
        self.mu_grad = 0.0
        self.log_sig_grad = 0.0

    def likelihood(self, data):
        z = (data - self.mu)/np.exp(self.log_sig)
        return np.power(2.0*np.pi, -0.5) * np.exp(-0.5*z) / np.exp(self.log_sig)

    def fit(self, data):
        self.mu_grad = ((self.mu - data)/np.exp(2.0*self.log_sig)).mean()
        self.log_sig_grad = (1.0 - np.square(data - self.mu) * np.exp(-2.0 * self.log_sig)).mean()
        self.mu -= 0.01 * self.mu_grad
        self.log_sig -= 0.01*self.log_sig_grad


unknown_mean = 3.0
unknown_var = 10.0

unknown = N(mean=unknown_mean, var=unknown_var)
batch_size = 100
model = LearnableUniNormal()
for i in range(10000):
    batch = unknown.sample(batch_size)
    lh = model.likelihood(batch)
    print('avg likelihood : ', lh.mean())
    print('model.mean:', model.mu, ' model.var:', np.exp(2.0 * model.log_sig))
    model.fit(batch)
