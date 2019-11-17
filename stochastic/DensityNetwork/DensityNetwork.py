import numpy as np
import matplotlib.pyplot as plt


class Unit:

    def __init__(self):
        self.lr = 0.0
        pass

    def forward(self, x):
        pass

    def backward(self, out_grad):
        pass

    def zero_grad(self):
        pass

    def step(self):
        pass

    def set_lr(self, lr):
        self.lr = lr



class ReLU(Unit):

    def __init__(self):
        super().__init__()
        self.rectified = 0

    def forward(self, x):
        self.rectified = (x > 0)
        return self.rectified * x

    def backward(self, grad):
        return self.rectified * grad

class Dense(Unit):

    def __init__(self, weight):
        super().__init__()
        self.w = weight
        self.bias = np.zeros(shape=(1, weight.shape[1]), dtype=np.float)
        self.w_grad = np.zeros(shape=self.w.shape, dtype=np.float)
        self.b_grad = np.zeros(shape=self.bias.shape, dtype=np.float)
        self.input_container = 0

    def forward(self, x):
        self.input_container = x
        return np.matmul(self.input_container, self.w) + self.bias

    def backward(self, grad):
        self.b_grad += grad
        self.w_grad += np.matmul(np.transpose(self.input_container), grad)
        return np.matmul(grad, np.transpose(self.w))

    def zero_grad(self):
        self.w_grad.fill(0.0)
        self.b_grad.fill(0.0)

    def step(self):
        self.w -= self.lr*self.w_grad
        self.bias -= self.lr*self.b_grad


class Bundle(Unit):

    def __init__(self, layer_list):
        super().__init__()
        self.layer_list = layer_list

    def forward(self, x):
        for k in range(0, len(self.layer_list)):
            x = self.layer_list[k].forward(x)
        return x

    def backward(self, grad):
        for k in reversed(range(len(self.layer_list))):
            grad = self.layer_list[k].backward(grad)
        return grad

    def zero_grad(self):
        for k in range(0, len(self.layer_list)):
            self.layer_list[k].zero_grad()

    def step(self):
        for k in range(0, len(self.layer_list)):
            self.layer_list[k].step()

    def set_lr(self, lr):
        self.lr = lr
        for k in range(0, len(self.layer_list)):
            self.layer_list[k].set_lr(self.lr)


def mu_distribution(_x):
    return _x*(_x + 3)*0.03


def var_distribution(_x):
    return (_x - 0.3)* (_x - 0.3)*0.01


def generate(_x):
    noise = np.random.randn(_x.size).reshape(_x.shape)
    mean = mu_distribution(_x)
    stddev = np.sqrt(var_distribution(_x))
    return stddev*noise + mean


def xavier(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2.0/(n_in + n_out))


def gaussian_log_likelihood(params, y):
    mu = params[0][0]
    log_sig = params[0][1]
    rv = y[0][0]
    beta = np.exp(-2.0*log_sig)
    llh = -0.5*(np.log(2.0*np.pi) + log_sig + np.square(mu - rv) * beta)
    grad = params
    grad[0][0] = -(mu - rv) * beta
    grad[0][1] = -0.5 + np.square(mu - rv) * beta
    return llh, grad


half_range = 1.0
d_size = 50000

x = 2 * half_range * (np.random.rand(d_size) - 0.5).reshape(-1, 1, 1)
y = generate(x)


nn = Bundle([Dense(xavier(1, 10)), ReLU(), Dense(xavier(10, 10)), ReLU(), Dense(xavier(10, 10)),
             ReLU(), Dense(xavier(10, 2))])
nn.set_lr(0.00001)

for i in range(10):
    mean_llh = 0.0
    for k in range(len(x)):
        nn.zero_grad()
        llh, llh_grad = gaussian_log_likelihood(nn.forward(x[k]), y[k])
        nn.backward(-llh_grad)
        mean_llh += llh
        nn.step()
    print('log likelihood : ', mean_llh/len(x))

r = np.linspace(-half_range, half_range, 100)
mu = np.empty(100, dtype=np.float)
var = np.empty(100, dtype=np.float)
for k in range(100):
    p = nn.forward(r[k].reshape(1, 1))
    mu[k] = p[0][0]
    var[k] = np.exp(p[0][1]*2.0)

plt.scatter(x,y,s=1)
plt.plot(r, mu, c='r', label='predicted mu')
plt.plot(r, var, c='y', label='predicted var')
plt.plot(r, mu_distribution(r), c='m', label='true mean')
plt.plot(r, var_distribution(r), c='k', label='true var')

plt.legend()
plt.show()
