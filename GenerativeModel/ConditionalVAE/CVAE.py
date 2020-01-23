import torchvision.datasets.mnist as mnist
import numpy as np
import matplotlib.pyplot as plt

download_root = './MNIST_DATASET'
train_dataset = mnist.MNIST(download_root, transform=None, train=True, download=True)
imgs = np.add(np.divide(np.array(train_dataset.data),255.0), -0.5).reshape(60000, 1, 28 * 28)
label = np.eye(10)[np.array(train_dataset.targets)].reshape(-1, 1, 10)

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



class SeLU(Unit):

  def __init__(self):
    super().__init__()
    self.aph = 1.6732
    self.lmd = 1.0507
    self.rectified = 0
    self.exp_x = 0
    self.leaked = 0

  def forward(self, x):
    self.rectified = (x>=0)
    self.leaked = (x<0)
    self.exp_x = np.exp(x)
    return self.lmd*(self.rectified*x + self.leaked*(self.aph*(self.exp_x - 1.0)))

  def backward(self, grad):
    return self.lmd*(self.rectified*grad + self.leaked*self.aph*self.exp_x*grad)


class Sigmoid(Unit):

    def __init__(self):
        super().__init__()
        self.sig = 0

    def forward(self, x):
        self.sig = 1.0 / (1.0 + np.exp(-x))
        return self.sig

    def backward(self, grad):
        return self.sig * (1.0 - self.sig) * grad


class Tanh(Unit):

    def __init__(self):
        super().__init__()
        self.e_2x = 0

    def forward(self, x):
        self.e_2x = np.exp(2*x)
        return (self.e_2x - 1.0)/(self.e_2x + 1.0)

    def backward(self, grad):
        return grad*(4.0*self.e_2x)/np.square(self.e_2x + 1.0)


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


class VariationalEncoder(Unit):

    def __init__(self, bundle):
        super().__init__()
        self.net = bundle
        self.mu = 0
        self.log_sig = 0
        self.sig = 0
        self.rv = 0

    def sample(self, x):
        self.mu, self.log_sig = np.hsplit(self.net.forward(x), 2)
        self.sig = np.exp(self.log_sig)
        self.rv = np.random.randn(self.mu.shape[0], self.mu.shape[1])
        return self.sig*self.rv + self.mu

    def re_sample(self):
        self.rv = np.random.randn(self.mu.shape[0], self.mu.shape[1])
        return self.sig*self.rv + self.mu

    def kld_std(self):
        return 0.5*(np.square(self.sig) + np.square(self.mu) - 2.0 * self.log_sig - 1.0).mean()

    def backward(self, grad):
        mu_grad = grad + self.mu
        ls_grad = grad * self.rv * self.sig + (np.square(self.sig) - 1.0)
        return self.net.backward(np.concatenate((mu_grad, ls_grad), axis=1))


def l2_loss(pred, ans):
    loss = np.square(pred - ans).mean()
    grad = 0.5 * (pred - ans)
    return loss, grad


def xavier(n_in, n_out):
    return np.random.randn(n_in, n_out) * np.sqrt(2.0/(n_in + n_out))

z_size = 4
encoder = Bundle([Dense(xavier(784 + 10, 128)), SeLU(), Dense(xavier(128, 128)), SeLU(), Dense(xavier(128, z_size + z_size))])
sampler = VariationalEncoder(encoder)
decoder = Bundle([Dense(xavier(z_size + 10, 128)), SeLU(), Dense(xavier(128, 128)), SeLU(), Dense(xavier(128, 784))])
encoder.set_lr(0.0001)
decoder.set_lr(0.0001)
zero = np.zeros(shape=(1, 10), dtype=np.float)

for n in range(10):
    mean_loss = 0
    mean_kldv = 0
    for i in range(len(imgs)):
        encoder.zero_grad()
        decoder.zero_grad()
        z = sampler.sample(np.concatenate((imgs[i], label[i]),axis=1))
        loss, grad = l2_loss(decoder.forward(np.concatenate((z, label[i]), axis=1)), imgs[i])
        sampler.backward(decoder.backward(grad)[:, 0:z_size])
        encoder.step()
        decoder.step()
        kl = sampler.kld_std()
        mean_loss += loss
        mean_kldv += kl
    print('iteration : ', n+1, 'l2 : ', mean_loss/len(imgs), 'kl_std : ', mean_kldv/len(imgs))


idx = np.random.randint(1,10000)
plt.title('original')
plt.imshow(imgs[idx].reshape(28, 28))
plt.show()
x = np.concatenate((imgs[idx], label[idx]), axis=1)
v = sampler.sample(x)
for j in range(4):
  plt.subplot(1, 4, j+1)
  plt.axis('off')
  z = np.concatenate((v, label[idx]), axis=1)
  plt.imshow(decoder.forward(z).reshape(28, 28))
plt.show()
