import numpy as np
import matplotlib.pyplot as plt


class Normal:

    def __init__(self, mu, var):
        self.mu = mu
        self.var = var

    def prob(self, _x):
        return np.power(2.0*np.pi*self.var, -0.5) * np.exp(-np.square(self.mu - _x)/(2.0 * self.var))

    def sample(self, size):
        return np.sqrt(self.var) * np.random.randn(size) + self.mu


def hypothesis(_x):
    return np.abs(_x)

target = Normal(0, 1)
sampler = Normal(4, 2)

x = sampler.sample(size=500000)
Eg_wh = target.prob(x)*hypothesis(x)/sampler.prob(x)

print('importance : ', Eg_wh.mean())
x = target.sample(size=100000)
print('monte carlo : ', hypothesis(x).mean())


M = 3
proposal = Normal(0, 2)
auxiliary = np.random.rand(50000)
x = proposal.sample(50000)
reject_prob = target.prob(x)/(M*proposal.prob(x))
v = []
for i in range(len(x)):
    if auxiliary[i] < reject_prob[i]:
        v.append(x[i])
print('rejection : ',hypothesis(v).mean())
