#pragma once
#include "nn.h"

class optimizer {
public:
	bool optim_alloced;
	double** params;
	double** grad;
	int* n_params;
	int n_bundle;
	double learning_rate;
	optimizer(nn* trainee) { 
		optim_alloced = false; 
		collect(trainee);
	}
	void zero_grad() {
		for (int i = 0; i < n_bundle; i++) {
			for (int j = 0; j < n_params[i]; j++) {
				grad[i][j] = 0.0;
			}
		}
	}

	void setLearingRate(const double& lr) {
		learning_rate = lr;
	}

	void step() {
		for (int i = 0; i < n_bundle; i++) {
			for (int j = 0; j < n_params[i]; j++) {
				params[i][j] -= learning_rate * grad[i][j];
			}
		}
	}

	void collect(nn* trainee) {
		if (optim_alloced) {
			delete[] params;
			delete[] grad;
			delete[] n_params;
		}
		vector<double*> p_bag;
		vector<double*> g_bag;
		vector<int> len_bag;
		trainee->delegate(p_bag, g_bag, len_bag);
		n_bundle = p_bag.size();
		params = new double*[n_bundle];
		grad = new double*[n_bundle];
		n_params = new int[n_bundle];
		for (int n = 0; n < n_bundle; n++) {
			n_params[n] = len_bag[n];
			params[n] = p_bag[n];
			grad[n] = g_bag[n];
		}
		optim_alloced = true;
	}

	~optimizer(){
		if (optim_alloced) {
			delete[] params;
			delete[] grad;
			delete[] n_params;
		}
	}
};