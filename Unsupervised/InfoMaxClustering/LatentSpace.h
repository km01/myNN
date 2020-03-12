#pragma once
#include "Core.h"

class Categorical_Distribution {
public:
	darr::v base = nullptr;
	darr::v base_grad = nullptr;
	double regularizer = 0.0;
	darr::v exp_cache = nullptr;
	int n_category = 0;

	Categorical_Distribution(const int& n_category) {
		this->n_category = n_category;
		base = darr::alloc(n_category);
		km::setZero(base, n_category);
		base_grad = darr::alloc(n_category);
		exp_cache = darr::alloc(n_category);
	}

	void forward(darr::v& params) {
		double denom = 0.0;
		for (int i = 0; i < n_category; i++) {
			exp_cache[i] = exp(base[i]);
			denom += exp_cache[i];
		}
		regularizer = 1.0 / denom;
		for (int i = 0; i < n_category; i++) {
			params[i] = exp_cache[i] * regularizer;
		}
	}

	void backward(darr::v& params_grad) {
		double regularizer_grad = 0.0;
		for (int i = 0; i < n_category; i++) {
			regularizer_grad += exp_cache[i] * params_grad[i];
		}
		for (int i = 0; i < n_category; i++) {
			base_grad[i] += (params_grad[i] - regularizer * regularizer_grad) * regularizer * exp_cache[i];
		}
	}

	void zero_grad() {
		for (int i = 0; i < n_category; i++) {
			base_grad[i] = 0.0;
		}
	}

	void step(const double& learning_rate) {
		for (int i = 0; i < n_category; i++) {
			base[i] -= learning_rate * base_grad[i];
		}
	}
};

class Diag_Normal_distribution {
public:
	darr::v mean = nullptr;
	darr::v mean_grad = nullptr;
	darr::v pre_stddev = nullptr;
	darr::v pre_stddev_grad = nullptr;
	int n_variables = 0;

	Diag_Normal_distribution(const int& n_variables) {
		this->n_variables = n_variables;
		mean = darr::alloc(n_variables);
		km::setZero(mean, n_variables);
		mean_grad = darr::alloc(n_variables);
		pre_stddev = darr::alloc(n_variables);
		km::setZero(mean, n_variables);
		pre_stddev_grad = darr::alloc(n_variables);
	}

	void forward(darr::v& mu, darr::v& sigma) {
		for (int i = 0; i < n_variables; i++) {
			mu[i] = mean[i];
			if (pre_stddev[i] < 0.0) {
				sigma[i] = exp(pre_stddev[i]);
			}
			else {
				sigma[i] = 1.0 + pre_stddev[i];
			}
		}

	}

	void backward(darr::v& mu_grad, darr::v& sigma_grad) {
		for (int i = 0; i < n_variables; i++) {
			mean_grad[i] += mu_grad[i];
			if (pre_stddev[i] < 0.0) {
				pre_stddev_grad[i] += exp(pre_stddev[i]) * sigma_grad[i];
			}
			else {
				pre_stddev_grad[i] += sigma_grad[i];
			}
		}
	}

	void zero_grad() {
		for (int i = 0; i < n_variables; i++) {
			mean_grad[i] = 0.0;
			pre_stddev_grad[i] = 0.0;
		}
	}

	void step(const double& learning_rate) {
		for (int i = 0; i < n_variables; i++) {
			mean[i] -= learning_rate * mean_grad[i];
			pre_stddev[i] -= learning_rate * pre_stddev_grad[i];

		}
	}
};