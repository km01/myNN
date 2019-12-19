#include "mynn_core.h"
#include "unit.h"
#include "dataframe.h"
#include "loss_function.h"
#include "model.h"
#include "optimizer.h"

double mean_accuarcy(int* pred, int* label, int& batch_size) {
	double mean_acc = 0.0;
	for (int i = 0; i < batch_size; i++) {
		if (pred[i] == label[i]) {
			mean_acc += 1.0;
		}
	}
	mean_acc = mean_acc / (double)batch_size;
	return mean_acc;
}

/*
 * by kimin Jeong, kimin227@naver.com
 */

int main() {
	/* data setting */
	string path = "";
	DataSet Iris(path+"data.csv");
	Labelset label(path+"answer.csv");
	Iris.min_max_scaling();
	int batch_size = 8;
	batch_loader dispenser(&Iris, &label);
	dispenser.alloc_batch_storage(batch_size);
	dispenser.shuffle_order();
	int nb_batch = Iris.n_rows / batch_size;

	/* model setting */
	multi_layer_network mlp(number_of_units(5));
	mlp.layer[0] = new bn_perceptrons(4, 50, RELU, true);
	mlp.layer[1] = new perceptrons(50, 50, RELU, true);
	mlp.layer[2] = new bn_perceptrons(50, 50, RELU, true);
	mlp.layer[3] = new perceptrons(50, 50, RELU, true);
	mlp.layer[4] = new bn_perceptrons(50, 3, SOFTMAX, true);
	mlp.alloc_single_memory();
	mlp.alloc_batch_memory(batch_size);

	/* loss function */
	cross_entropy loss_fn;


	/* optimizer setting*/
	optimizer adam(mlp);
	adam.set_learning_rate(0.01 / (double)batch_size);
	adam.use_adaptive_momentum(0.9, 0.999);
	//adam.use_RMSprop(0.999);
	//adam.use_momentum(0.9);

	/* f i t t i n g */
	double mean_loss = 0.0;
	double mean_acc = 0.0;
	double learning_Rate = 0.01;
	double** dLoss = km_2d::alloc(batch_size, mlp.output_size); // for storing loss prime

	for (int iter = 0; iter < 500; iter++) {
		mean_loss = 0.0;
		mean_acc = 0.0;
		for (int i = 0; i < nb_batch; i++) {
			dispenser.next_batch();
			mlp.batch_feed_forward(dispenser.mini_x);
			mean_loss += loss_fn.batch_loss_prime(dLoss, mlp.batch_output, dispenser.mini_y, batch_size, mlp.output_size);
			mlp.zero_grad();
			mlp.batch_feed_backward(dLoss);
			adam.step();
			mean_acc += mean_accuarcy(mlp.batch_argmax, dispenser.mini_y, batch_size);
		}
		cout << "mean_loss : " << mean_loss / nb_batch << endl;
		cout << "mean_acc : " << mean_acc / nb_batch << endl;
	}

	/* t e s t i n g */
	mean_loss = 0.0;
	mean_acc = 0.0;
	for (int i = 0; i < Iris.n_rows; i++) {
		mlp.single_feed_forward(Iris.data[i]);
		mean_loss += loss_fn.single_loss(mlp.single_output, label.label[i], mlp.output_size);
		if (label.label[i] == mlp.single_argmax) {
			mean_acc += 1.0;
		}
	}
	cout << "test mode total_loss : " << mean_loss / Iris.n_rows << endl;
	cout << "test mode total_acc : " << mean_acc / Iris.n_rows << endl;

	km_2d::free(dLoss, batch_size);
}