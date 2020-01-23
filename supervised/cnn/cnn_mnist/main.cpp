#include "model.h"
#include "optimizer.h"
#include "DataFrame.h"


using namespace std;
int main() {
	string path = "C:\\ai_data\\";
	DataSet mnist(path + "mnist_train_100.csv", 0, 1);
	Labelset label(path + "mnist_train_100.csv", 0);


	mnist.mnist_scaling(-0.5,0.5);
	int batch_size = 8;
	batch_loader dispenser(&mnist, &label);
	dispenser.alloc_batch_storage(batch_size);
	dispenser.shuffle_order(); dispenser.next_batch();

	classifier* net = new classifier(N_Layer(7)); {
		// out = 1 + (in - Filt)/stride
		net->layer[0] = new kernel3D(shape(1, 28, 28), shape(1, 6, 6), shape(6, 12, 12), 2, 2);//12 = 1 + (28-6)/2 
		net->layer[1] = new ReLU(net->layer[0]->output_size);
		net->layer[2] = new kernel3D(shape(6, 12, 12), shape(6, 4, 4), shape(12, 5, 5), 2, 2);// 5 = 1 + (12-4)/2 
		net->layer[3] = new ReLU(net->layer[2]->output_size);
		net->layer[4] = new kernel3D(shape(12, 5, 5), shape(12, 3, 3), shape(18, 3, 3), 1, 1);//3 = 1 + (5-3)/1 
		net->layer[5] = new ReLU(net->layer[4]->output_size);
		net->layer[6] = new kernel3D(shape(18, 3, 3), shape(18, 3, 3), shape(10, 1, 1), 1, 1);//1 = 1+(3-3)/1

	} net->publish(); net->alloc(batch_size);
	
	double learning_rate = 0.02/batch_size;


	optimizer* optim = new optimizer(net, learning_rate); optim->zero_grad();

	double** dLoss = km_2d::alloc(batch_size, net->output_size);
	int nb_batch = mnist.n_rows / batch_size;
	double loss = 0.0;
	double accuarcy = 0.0;
	for (int ep = 0; ep < 100; ep++) {
		loss = 0.0;
		accuarcy = 0.0;
		for (int b = 0; b < nb_batch; b++) {
			dispenser.next_batch();
			net->forward(dispenser.mini_x);
			optim->zero_grad();
			loss += net->Bulit_in_CELoss(dLoss, dispenser.mini_y);
			accuarcy += km::accuarcy(net->argmax, dispenser.mini_y, net->output_size);
			net->backward(dLoss);
			optim->step();
		}
		cout << "session " << ep <<" - ";
		cout << "Loss : " << loss / nb_batch <<" ---- ";
		cout << "acc : " << accuarcy / nb_batch <<endl;

	}

	delete optim;
	delete net;
	km_2d::free(dLoss, batch_size);
}