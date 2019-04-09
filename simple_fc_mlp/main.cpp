#include "mynn_core.h"
#include "unit.h"
#include "dataframe.h"
#include "loss_function.h"

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
int main() {
	string path = "";
	DataSet Iris(path+"data.csv");
	Labelset label(path+"answer.csv");
	Iris.min_max_scaling();

	int batch_size = 8;
	batch_loader dispenser(&Iris, &label);
	dispenser.alloc_batch_storage(batch_size);

	unit* one_fc = new perceptrons(4, 3, SOFTMAX, true);

	one_fc->alloc_batch_storage(batch_size);

	int nb_batch = Iris.n_rows / batch_size;
	double** predict = km_2d::alloc(batch_size,one_fc->output_size);

	cross_entropy loss_fn;
	double** delta = km_2d::alloc(batch_size, one_fc->output_size);
	for (int i = 0; i < 10000; i++) {
		double loss = 0.0;
		double acc = 0.0;
		for (int b = 0; b < nb_batch; b++) {
			one_fc->grad_zero();
			one_fc->batch_load_inputs(dispenser.mini_x);
			one_fc->batch_forward_prop(predict);
			loss += loss_fn.batch_loss_prime(delta, predict, dispenser.mini_y, batch_size, one_fc->output_size);
			one_fc->batch_back_prop(delta);
			one_fc->feed_grad_itself(0.001);
			acc += sfmx_accuracy(predict, dispenser.mini_y, batch_size, one_fc->output_size);
		}
		
		cout << "avg loss : " <<loss/(double)nb_batch << endl;
		cout << "avg acc : " << acc / (double)nb_batch << endl << endl;
	}

	km_2d::free(delta, one_fc->batch_size);
	km_2d::free(predict, one_fc->batch_size);
}