#include "model.h"
#include "optimizer.h"
#include "DataFrame.h"
/*
	author : kimin Jeong
	kimin353@gmail.com
	kimin353@cau.ac.kr

*/

using namespace std;
int main(){
	string path = "C:\\ai_data\\Iris_Data\\";
	DataSet iris(path + "data.csv", 1, 0);
	Labelset label(path +"answer.csv", 1);
	int batch_size = 10;
	scaler scaler(iris.n_cols);
	scaler.get_minmax(iris.data, iris.n_rows);
	scaler.scale(iris.data, iris.n_rows, -0.5, 0.5);

	batch_loader dispenser(&iris, &label);
	dispenser.alloc_batch_storage(batch_size);
	dispenser.shuffle_order(); dispenser.next_batch();

	classifier* net = new classifier(N_Layer(5)); {
		net->layer[0] = new fully_connected(4, 16);
		net->layer[1] = new caps_block(4, 4, 6, 8);
		net->layer[2] = new caps_block(6, 8, 4, 8);
		net->layer[3] = new caps_block(4, 8, 3, 2);
		net->layer[4] = new fully_connected(6, 3);

	} net->publish(); net->alloc(batch_size);
	
	double learning_rate = 0.1/ batch_size;
	optimizer* optim = new optimizer(net, learning_rate); optim->zero_grad();
	double** dLoss = km_2d::alloc(batch_size, net->output_size);
	int nb_batch = iris.n_rows / batch_size;
	double loss = 0.0;
	double accuarcy = 0.0;

	net->forward(dispenser.mini_x);
	cout << net->softmax_output[0][0] << " " << net->softmax_output[0][1] <<" "<< net->softmax_output[0][2] << endl;
	for (int ep = 0; ep < 5000; ep++) {
		loss = 0.0;
		accuarcy = 0.0;
		for (int b = 0; b < nb_batch; b++) {
			dispenser.next_batch();
			net->forward(dispenser.mini_x);
			optim->zero_grad();
			loss += net->Bulit_in_CELoss(dLoss, dispenser.mini_y);
			accuarcy += km::accuarcy(net->argmax, dispenser.mini_y, batch_size);
			net->backward(dLoss);
			optim->step();
		}
		cout << "epoch " << ep <<" - ";
		cout << "Loss : " << loss / nb_batch <<" ---- ";
		cout << "acc : " << accuarcy / nb_batch <<endl;

	}
	//delete optim;
	//delete net;
	//km_2d::free(dLoss, batch_size);
}