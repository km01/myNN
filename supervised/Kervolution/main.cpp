#include "nn.h"
#include "optimizer.h"
#include "DataFrame.h"

/*
	author : kimin Jeong
	kimin353@gmail.com
	kimin353@cau.ac.kr
*/

using namespace std;
int main() {
	string path = "C:\\ai_data\\";
	DataSet mnist(path + "mnist_train_100.csv", 0, 1);
	Labelset label(path + "mnist_train_100.csv", 0);
	int batch_size = 4;
	ImgScaler scaler(-0.5,0.5);
	scaler.scale(mnist.data, mnist.n_rows,mnist.n_cols);
	batch_loader dispenser(&mnist, &label);
	dispenser.alloc_batch_storage(batch_size);
	dispenser.shuffle_order(); dispenser.next_batch();

	nn* net = new nn(N_Layer(13)); { // converage very slow

		/* Kervolution(GuassianRBF) -> BN -> ReLU */
		net->layer[0] = new L2NormKernel3D(shape(1, 28, 28), shape(1, 6, 6), shape(4, 12, 12), 2, 2);
		net->layer[1] = new GaussianRBF(net->layer[0]->output_size);
		net->layer[2] = new BatchNormalizer(shape(4, 12, 12));
		net->layer[3] = new ReLU(net->layer[0]->output_size);

		/* Kervolution(GuassianRBF) -> BN -> ReLU */
		net->layer[4] = new L2NormKernel3D(shape(4, 12, 12), shape(4, 4, 4), shape(8, 5, 5), 2, 2);
		net->layer[5] = new GaussianRBF(net->layer[4]->output_size);
		net->layer[6] = new BatchNormalizer(shape(8, 5, 5));
		net->layer[7] = new ReLU(net->layer[4]->output_size);

		/* Kervolution(GuassianRBF) -> BN -> ReLU */
		net->layer[8] = new L2NormKernel3D(shape(8, 5, 5), shape(8, 3, 3), shape(12, 3, 3), 1, 1);
		net->layer[9] = new GaussianRBF(net->layer[8]->output_size);
		net->layer[10] = new BatchNormalizer(shape(12, 3, 3));
		net->layer[11] = new ReLU(net->layer[8]->output_size);

		net->layer[12] = new kernel3D(shape(12, 3, 3), shape(12, 3, 3), shape(10, 1, 1), 1, 1);
	} net->publish(); net->alloc(batch_size);

	double learning_rate = 0.005 / batch_size;
	optimizer* optim = new optimizer(net); optim->zero_grad();
	double** dLoss = km_2d::alloc(batch_size, net->output_size);
	int nb_batch = mnist.n_rows / batch_size;
	double loss = 0.0;
	double accuarcy = 0.0;
	optim->setLearingRate(learning_rate);
	optim->use_AdaptiveMomentum(0.9, 0.999);
	for (int ep = 0; ep < 100; ep++) {
		loss = 0.0;
		accuarcy = 0.0;
		for (int b = 0; b < nb_batch; b++) {
			dispenser.next_batch();
			optim->zero_grad();
			net->predict(dispenser.mini_x);
			loss += km_2d::CEloss(dLoss, net->output_port, dispenser.mini_y, batch_size, net->output_size);
			km_2d::argmax(net->argmax, net->output_port, batch_size, net->output_size);
			accuarcy += km::accuarcy(net->argmax, dispenser.mini_y, batch_size);
			net->backward(dLoss);
			optim->step();
		}
		cout << "epoch " << ep << " - ";
		cout << "Loss : " << loss / nb_batch << " ---- ";
		cout << "acc : " << accuarcy / nb_batch << endl;
		double test_acc = 0.0;
		for (int i = 0; i < mnist.n_rows; i++) {
			if (label.label[i] == net->one_predict(mnist.data[i])) {
				test_acc += 1.0;
			}
		}
		cout << "test acc : " << test_acc / mnist.n_rows << endl;
	}
	delete optim;
	delete net;
}