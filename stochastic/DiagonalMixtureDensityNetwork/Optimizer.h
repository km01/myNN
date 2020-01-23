#pragma once
#include "Unit.h"

class Optimizer {
public:
	double** weight = nullptr;
	double** grad = nullptr;
	int* weight_len = nullptr;
	int n_weight = 0;
	double learning_rate = 0.0;

	Optimizer(Unit& trainee) {
		vector<double*> p_bag;
		vector<double*> g_bag;
		vector<int> len_bag;
		trainee.takeParams(p_bag, g_bag, len_bag);
		n_weight = (int)len_bag.size();
		weight = new double* [n_weight];
		grad = new double* [n_weight];
		weight_len = new int[n_weight];
		for (int i = 0; i < n_weight; i++) {
			weight_len[i] = len_bag[i];
			weight[i] = p_bag[i];
			grad[i] = g_bag[i];
		}
	}

	Optimizer(Unit& trainee1, Unit& trainee2) {
		vector<double*> p_bag;
		vector<double*> g_bag;
		vector<int> len_bag;
		trainee1.takeParams(p_bag, g_bag, len_bag);
		trainee2.takeParams(p_bag, g_bag, len_bag);
		n_weight = (int)len_bag.size();
		weight = new double* [n_weight];
		grad = new double* [n_weight];
		weight_len = new int[n_weight];
		for (int i = 0; i < n_weight; i++) {
			weight_len[i] = len_bag[i];
			weight[i] = p_bag[i];
			grad[i] = g_bag[i];
		}
	}

	void zero_grad() {
		for (int i = 0; i < n_weight; i++) {
			for (int j = 0; j < weight_len[i]; j++) {
				grad[i][j] = 0.0;
			}
		}
	}

	void step() {
		for (int i = 0; i < n_weight; i++) {
			for (int j = 0; j < weight_len[i]; j++) {
				weight[i][j] -= learning_rate * grad[i][j];
			}
		}
	}

	void setLearningRate(const double& learning_rate) {
		this->learning_rate = learning_rate;
	}

	~Optimizer() {
		delete[] weight;
		delete[] grad;
		delete[] weight_len;
	}
};