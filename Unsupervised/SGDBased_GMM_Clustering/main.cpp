#include "CustomDataSet.h"
#include "Distribution.h"
using namespace darr;
int main() {

	int batch_size = 100;
	int K = 6;
	double learning_rate = 0.01;
	int num_epochs = 10000;
	
	CustomDataSet dataset;
	v2D data = alloc(batch_size, 2);

	GaussianMixtureModel model(K, 2);

	for (int k = 0; k < K; k++) {
		cout << model.prior_cache[k] << " ";
	}
	cout << endl;
	for (int k = 0; k < K; k++) {
		cout << model.models[k]->mean[0] << " " << model.models[k]->mean[1] << endl;
	}
	for (int k = 0; k < K; k++) {
		cout << model.models[k]->pre_stddev[0] << " " << model.models[k]->pre_stddev[1] << endl;
	}

	for (int iter = 0; iter < num_epochs; iter++) {
		dataset.getData(data, batch_size);
		model.zero_grad();
		cout <<" iteration : "<<iter+1<< "	average_GMM_log_likelihood : "  << model.fit2(data, batch_size) << endl;
		model.step(learning_rate/(double)batch_size);
	}

	cout << " --------------------------------" << endl;
	cout << K << "-GMM " << endl;
	cout << endl;
	cout << "prior" << endl;
	for (int k = 0; k < K; k++) {
		cout << model.prior_cache[k] << " ";
	}
	cout << endl;
	cout << endl;
	cout << "mean" << endl;

	for (int k = 0; k < K; k++) {
		cout << model.mean_cache[k][0] << " " << model.mean_cache[k][1] << endl;
	}
	cout << endl;
	cout << "stddev" << endl;
	for (int k = 0; k < K; k++) {
		cout << model.stddev_cache[k][0] << " " << model.stddev_cache[k][1] << endl;
	}
}