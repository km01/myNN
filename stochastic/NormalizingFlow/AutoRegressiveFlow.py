import numpy as np
import matplotlib.pyplot as plt


def data_sampling(n_data):
    xp = (np.random.rand(n_data)) - 0.5
    yp = (np.random.randn(n_data) * 0.01) + np.sqrt(0.25 - np.square(xp))
    return xp, yp



class Affine1D:
    def __init__(self):
        self.scale = 1.0
        self.bias = 0.0
        self.s_grad = 0.0
        self.b_grad = 0.0
        self.cache = 0.0

    def fn(self, data):
        return self.scale * data + self.bias

    def forward(self, data):
        self.cache = data
        return self.fn(data)

    def zero_grad(self):
        self.b_grad = 0.0
        self.s_grad = 0.0

    def step(self, lr):
        self.bias -= lr * self.b_grad
        self.scale -= lr * self.s_grad

    def backward(self, grad):
        self.b_grad += grad.mean()
        self.s_grad += (grad * self.cache).mean()
        return grad * self.scale


class Ar:

    def __init__(self):
        self.mu_1 = 0.0
        self.tau_1 = 0.0
        self.grad_mu_1 = 0.0
        self.grad_tau_1 = 0.0
        self.nn_mu_2 = Affine1D()
        self.nn_tau_2 = Affine1D()

        self.cache_z1 = 0.0
        self.cache_z2 = 0.0
        self.cache_tau_2 = 0.0
        self.cache_mu_2 = 0.0

    def inv_forward(self, x1, x2):
        self.cache_z1 = (x1 - self.mu_1)/np.exp(self.tau_1)
        self.cache_mu_2 = self.nn_mu_2.forward(x1)
        self.cache_tau_2 = self.nn_tau_2.forward(x1)
        self.cache_z2 = (x2 - self.cache_mu_2)/np.exp(self.cache_tau_2)
        det = np.exp(self.tau_1 + self.cache_tau_2)
        return self.cache_z2, self.cache_z1, det

    def inv_backward(self, grad_z2, grad_z1):
        self.grad_tau_1 += (1.0 - (grad_z1 * self.cache_z1)).mean()
        self.grad_mu_1 -= (grad_z1/np.exp(self.tau_1)).mean()
        gx1_0 = grad_z1/np.exp(self.tau_1)
        tau_2_grad = (1.0 - (grad_z2 * self.cache_z2))
        mu_2_grad = -(grad_z2 / np.exp(self.cache_tau_2))
        gx1_1 = self.nn_mu_2.backward(mu_2_grad)
        gx1_2 = self.nn_tau_2.backward(tau_2_grad)
        grad_x2 = grad_z2/np.exp(self.cache_tau_2)
        grad_x1 = gx1_0 + gx1_1 + gx1_2
        return grad_x1, grad_x2

    def zero_grad(self):
        self.grad_mu_1 = 0.0
        self.grad_tau_1 = 0.0
        self.nn_mu_2.zero_grad()
        self.nn_tau_2.zero_grad()

    def step(self, lr):
        self.mu_1 -= lr * self.grad_mu_1
        self.tau_1 -= lr * self.grad_tau_1
        self.nn_mu_2.step(lr)
        self.nn_tau_2.step(lr)

    def transpose(self, z2, z1):
        x1 = np.exp(self.tau_1)*z1 + self.mu_1
        x2 = np.exp(self.nn_tau_2.forward(x1))*z2 + self.nn_mu_2.forward(x1)
        return x1, x2

class AutoRegressiveFlow:

    def __init__(self, n_layer):
        self.ar = []
        for l in range(n_layer):
            self.ar.append(Ar())

    def fit(self, x1, x2):
        z1, z2 = x1, x2
        norm = 1.0
        for l in reversed(range(len(self.ar))):
            z1, z2, det = self.ar[l].inv_forward(z1, z2)
            norm *= det

        lh = np.exp(-0.5*(np.square(z1) + np.square(z2)))/(2.0 * np.pi)
        lh = lh/norm

        for l in reversed(range(len(self.ar))):
            self.ar[l].zero_grad()

        grad_z1, grad_z2 = z1, z2
        for l in range(len(self.ar)):
            grad_z1, grad_z2 = self.ar[l].inv_backward(grad_z1, grad_z2)
        for l in reversed(range(len(self.ar))):
            self.ar[l].step(0.0001)
        return lh.mean()

    def sampling(self, n_data):
        x1 = np.random.randn(n_data)
        x2 = np.random.randn(n_data)
        for l in range(len(self.ar)):
            x1, x2 = self.ar[l].transpose(x1, x2)
        return x1, x2

model = AutoRegressiveFlow(n_layer=6)
for i in range(40000):
    a, b = data_sampling(n_data=128)
    print(model.fit(a, b))

x, y = model.sampling(2000)
plt.scatter(x, y, s=1)
a, b = data_sampling(2000)
plt.scatter(a,b,s=1)
plt.show()
