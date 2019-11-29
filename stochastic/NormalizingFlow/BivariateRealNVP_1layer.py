import matplotlib.pyplot as plt
import numpy as np


class BiVariateGaussian:

    def __init__(self, mean, cov):
        self.mu = np.array(mean)
        self.chol = np.linalg.cholesky(np.array(cov))

    def sample(self, size):
        rv = np.random.randn(size*2).reshape(size, 2)
        for i in range(size):
            rv[i] = self.chol.dot(rv[i]) + self.mu
        return rv


std = BiVariateGaussian(mean=[0.0, 0.0], cov=[[1.0, 0.0],
                                              [0.0, 1.0]])


class Affine1D:
    def __init__(self):
        self.scale = 1.0
        self.bias = 0.0
        self.cache = 0.0

    def fn(self, data):
        return self.scale * data + self.bias

    def forward(self, data):
        self.cache = data
        return self.fn(data)

    def feed(self, grad):
        self.bias -= 0.001 * grad.mean()
        self.scale -= 0.001 * (grad * self.cache).mean()


def split(data):
    x1, x2 = np.hsplit(data, 2)
    return x1.reshape(-1), x2.reshape(-1)


class BiVariateRealNVP:

    def __init__(self):
        self.s = Affine1D()
        self.t = Affine1D()

    def likelihood(self, data):
        x1, x2 = split(data)
        z1 = x1
        det = np.exp(1.0 * self.s.fn(z1))
        z2 = (x2 - self.t.fn(z1)) / det
        p_z = np.exp(-0.5 * (np.square(z1) + np.square(z2)))/(2.0*np.pi)
        return p_z/det

    def fit(self, data):
        x1, x2 = split(data)
        z1 = x1
        s_x1 = self.s.forward(x1)
        t_x1 = self.t.forward(x1)
        t_grad = (t_x1 - x2) * np.exp(-2.0 * s_x1)
        s_grad = -np.square(x2 - t_x1) * np.exp(-2.0 * s_x1) + 1.0
        self.t.feed(t_grad)
        self.s.feed(s_grad)

    def sample(self, n_data):
        rv = std.sample(n_data)
        for i in range(n_data):
            rv[i][1] = rv[i][1] * np.exp(self.s.fn(rv[i][0])) + self.t.fn(rv[i][0])
        return rv


unknown = BiVariateGaussian(mean=[0.0, 0.0], cov=[[1.0, 0.5], [0.5, 1.0]])
model = BiVariateRealNVP()
batch_size = 100
for i in range(10000):
    batch = unknown.sample(batch_size)
    likelihood = model.likelihood(batch)
    model.fit(batch)
    print('likelihood : ', likelihood.mean())

rv = model.sample(10000)
batch = unknown.sample(10000)
plt.scatter(rv[:, 0], rv[:, 1], s=1)

plt.scatter(batch[:, 0], batch[:, 1], s=1)
plt.show()