#include "mynn_core.h"
#include "unit.h"
#include "dataframe.h"
#include "loss_function.h"
#include "model.h"
double sfmx_accuracy(double**batch_pred, int* batch_label, const int& batch_size, const int& len) {
	double mean_acc = 0.0;
	double max;
	int argmax;
	for (int b = 0; b < batch_size; b++) {
		max = batch_pred[b][0];
		argmax = 0;
		for (int i = 1; i < len; i++) {
			if (max < batch_pred[b][i]) {
				argmax = i;
				max = batch_pred[b][argmax];
			}
		}
		if (argmax == batch_label[b]) {
			mean_acc += 1.0;
		}
	}
	return mean_acc / (double)batch_size;
}

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
int main() {
	string path = "";
	DataSet Iris(path+"data.csv");
	Labelset label(path+"answer.csv");
	Iris.min_max_scaling();

	int batch_size = 10;
	batch_loader dispenser(&Iris, &label);
	dispenser.alloc_batch_storage(batch_size);
	dispenser.shuffle_order();
	multi_layer_network mlp(number_of_units(5));
	mlp.layer[0] = new perceptrons(4, 100, RELU, true);
	mlp.layer[1] = new perceptrons(100, 50, RELU, true);
	mlp.layer[2] = new perceptrons(50, 50, RELU, true);
	mlp.layer[3] = new perceptrons(50, 50, RELU, true);
	mlp.layer[4] = new perceptrons(50, 3, SOFTMAX, true);
	mlp.alloc_sample_memory();

	mlp.alloc_batch_memory(batch_size);
	int nb_batch = Iris.n_rows / batch_size;

	cross_entropy loss_fn;
	double mean_loss = 0.0;
	double mean_acc = 0.0;

	double learning_Rate = 0.01;

	double** dLoss = km_2d::alloc(batch_size, mlp.output_size);
	for (int iter = 0; iter < 100; iter++) {
		mean_loss = 0.0;
		mean_acc = 0.0;
		for (int i = 0; i < nb_batch; i++) {
			dispenser.next_batch();
			mlp.batch_feed_forward(dispenser.mini_x);
			mean_loss += loss_fn.batch_loss_prime(dLoss, mlp.batch_output, dispenser.mini_y, batch_size, mlp.output_size);
			mlp.zero_grad();
			mlp.batch_feed_backward(dLoss);
			mlp.feed_grad_itself(learning_Rate / (double)batch_size);
			mean_acc += mean_accuarcy(mlp.batch_argmax, dispenser.mini_y, batch_size);
		}
		cout << "mean_loss : " << mean_loss / nb_batch << endl;
		cout << "mean_acc : " << mean_acc / nb_batch << endl;

	}
	km_2d::free(dLoss, batch_size);
}