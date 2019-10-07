import torchvision.datasets.mnist as mnist
import numpy as np
import matplotlib.pyplot as plt


download_root = './MNIST_DATASET'
train_dataset = mnist.MNIST(download_root, transform=None, train=True, download=True)
imgs = np.add(np.divide(np.array(train_dataset.data),255.0), -0.5).reshape(60000, 28*28, 1)


class Unit:

    def __init__(self, n_in, n_out):
        self.input_size = n_in
        self.output_size = n_out
        self.input_container = None

    def forward(self, x):
        pass

    def backward(self, out_grad):
        pass

    def zero_grad(self):
        pass

    def feed(self, lr):
        pass


class Dense(Unit):

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.w = np.random.rand(self.output_size, self.input_size) * np.sqrt(
            2.0 / (self.output_size + self.input_size)) - np.sqrt(2.0 / (self.output_size + self.input_size)) * 0.5
        self.w_grad = np.zeros((self.output_size, self.input_size), dtype=np.float)
        self.bias = np.zeros((self.output_size, 1), dtype=np.float)
        self.bias_grad = np.zeros((self.output_size, 1), dtype=np.float)

    def forward(self, x):
        self.input_container = x
        return np.matmul(self.w, self.input_container) + self.bias

    def backward(self, grad):
        self.bias_grad = grad
        self.w_grad = np.matmul(grad, np.transpose(self.input_container))
        return np.matmul(np.transpose(self.w), grad)

    def feed(self, learning_rate):
        self.bias -= learning_rate * self.bias_grad
        self.w -= learning_rate * self.w_grad

    def zero_grad(self):
        self.bias_grad.fill(0.0)
        self.w_grad.fill(0.0)


class Sigmoid(Unit):

    def __init__(self, input_size):
        super().__init__(input_size, input_size)

    def forward(self, x):
        self.input_container = x
        return 1.0 / (1.0 + np.exp(-self.input_container))

    def backward(self, grad):
        return (np.exp(-self.input_container) / (
                    (1 + np.exp(-self.input_container)) * (1 + np.exp(-self.input_container)))) * grad


class Bundle(Unit):

    def __init__(self, layer_list):
        super().__init__(layer_list[0].input_size, layer_list[-1].output_size)
        self.layer_list = layer_list

    def forward(self, x):
        for i in range(0, len(self.layer_list)):
            x = self.layer_list[i].forward(x)
        return x

    def backward(self, grad):
        for i in reversed(range(len(self.layer_list))):
            grad = self.layer_list[i].backward(grad)
        return grad

    def feed(self, learning_rate):
        for i in range(0, len(self.layer_list)):
            self.layer_list[i].feed(learning_rate)

    def zero_grad(self):
        for i in range(0, len(self.layer_list)):
            self.layer_list[i].zero_grad()


encoder = Bundle([Dense(784, 32), Sigmoid(32)])
decoder = Bundle([Dense(32, 784)])


def mse_loss(pred, ans):

    loss_grad = (pred - ans)
    loss = 0.5 * np.square(pred - ans).sum()
    return loss, loss_grad


for i in range(2):
    mean_loss = 0
    for data in imgs:
        encoder.zero_grad()
        decoder.zero_grad()
        encoded = encoder.forward(data)
        decoded = decoder.forward(encoded)
        loss, grad = mse_loss(decoded, data)
        encoder.backward(decoder.backward(grad))
        encoder.feed(0.0005)
        decoder.feed(0.0005)
        mean_loss += loss
    mean_loss = mean_loss/len(imgs)
    print('loss : ', mean_loss)


def visualize(origin, generated):
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(origin)
    plt.subplot(1, 2, 2)
    plt.title('Generated')
    plt.imshow(generated)
    plt.show()


while True:

    flag = input('try[y/n]? : ')
    if flag == 'y':
        img = imgs[np.random.randint(100, 50000)]
        decoded = decoder.forward(encoder.forward(img))
        visualize(img.reshape(28, 28), decoded.reshape(28, 28))


    else:
        break















