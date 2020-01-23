#include "Unit.h"
#include "Optimizer.h"
#include "DataFrame.h"
#include "Viewer.h"
int main() {

	string path = "C:\\Data\\";
	DataSet mnist(path + "mnist_train_100.csv", 0, 1);
	Labelset label(path + "mnist_train_100.csv", 0);
	mnist.mnist_scaling(-0.5, 0.5);
	cout << "DATA LOAD SUCCESSED" << endl;

	int latent_size = 8*4*4;
	Sequential encoder;
	encoder.push(new Conv2D(Shape(1, 28, 28), Shape(1, 2, 2), Shape(4, 27, 27), 1, 1)); // 27 = 1 + (28 - 2)/1
	encoder.push((new LeakyReLU(4*27*27))->setSlope(0.01));

	encoder.push(new Conv2D(Shape(4, 27, 27), Shape(4, 3, 3), Shape(6, 13, 13), 2, 2)); // 13 = 1 + (27 - 3)/2
	encoder.push((new LeakyReLU(6 * 13 * 13))->setSlope(0.01));

	encoder.push(new Conv2D(Shape(6, 13, 13), Shape(6, 3, 3), Shape(8, 6, 6), 2, 2)); // 6 = 1 + (13 - 3)/2
	encoder.push((new LeakyReLU(8 * 6 * 6))->setSlope(0.01));
	
	encoder.push(new Conv2D(Shape(8, 6, 6), Shape(8, 3, 3), Shape(8, 4, 4), 1, 1)); // 4 = 1 + (6 - 3)/1

	encoder.push(new Tanh(latent_size));
	
	encoder.setCache(1);

	Sequential decoder;

	decoder.push(new Conv2D_Transposed(Shape(8, 4, 4), Shape(8, 3, 3), Shape(8, 6, 6), 1, 1)); // 7 = 1 + (7 - 3)/2
	decoder.push((new LeakyReLU(8 * 6 * 6))->setSlope(0.05));
	
	decoder.push(new Conv2D_Transposed(Shape(8, 6, 6), Shape(6, 3, 3), Shape(6, 13, 13), 2, 2)); // 7 = 1 + (14 - 2)/2
	decoder.push((new LeakyReLU(6 * 13 * 13))->setSlope(0.1));

	decoder.push(new Conv2D_Transposed(Shape(6, 13, 13), Shape(4, 3, 3), Shape(4, 27, 27), 2, 2));
	decoder.push((new LeakyReLU(4 * 27 * 27))->setSlope(0.2));

	decoder.push(new Conv2D_Transposed(Shape(4, 27, 27), Shape(1, 2, 2), Shape(1, 28, 28), 1, 1));
	decoder.setCache(1);

	Optimizer optim(encoder, decoder);
	optim.learning_rate = 0.001;
	Darr y = alloc(784);
	Darr z = alloc(latent_size);
	Darr y_grad = alloc(784);
	double loss = 0.0;
	DarrList img = km::alloc(100, 784);
	for (int ep = 0; ep < 1000; ep++) {
		loss = 0.0;
		for (int i = 0; i < 100; i++) {
			optim.zero_grad();
			encoder.charge(mnist.data[i], 0);
			encoder.forward(decoder.in_port[0], 0);
			decoder.forward(y, 0);
			loss += km::LossFn::L2Loss(y_grad, y, mnist.data[i], 784);
			decoder.backward(y_grad, 0);
			encoder.backward(decoder.grad_port, 0);
			optim.step();
		}
		cout << "loss : " << loss/100.0 << endl;
	}
	for (int i = 0; i < 100; i++) {
		encoder.charge(mnist.data[i], 0);
		encoder.forward(decoder.in_port[0], 0);
		decoder.forward(img[i], 0);
	}
	Viewer viewer(10, 10, 28, 28, img);
	viewer.show();
	km::free(img, 100);
	km::free(y_grad);
	km::free(y);
	km::free(z);
	cout << "end" << endl;
}
