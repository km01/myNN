#pragma once
#include "Core.h"

class Categorical_Distribution {
public:
	darr::v base = nullptr;
	darr::v base_grad = nullptr;
	double regularizer = 0.0;
	darr::v exp_cache = nullptr;
	int n_category = 0;
	Categorical_Distribution(){}
	Categorical_Distribution(const int& n_category) {
		create(n_category);
	}

	void create(const int& n_category) {
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
	Diag_Normal_distribution(){}
	Diag_Normal_distribution(const int& n_variables) {
		create(n_variables);
	}

	void create(const int& n_variables) {
		this->n_variables = n_variables;
		mean = darr::alloc(n_variables);
		km::normal_sampling(mean, n_variables, 0.0, 1.0);
		mean_grad = darr::alloc(n_variables);
		pre_stddev = darr::alloc(n_variables);
		km::setZero(pre_stddev, n_variables);
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

class GaussianMixtureModel {
public:
	int n_category = 0;
	int n_variables = 0;
	darr::v prior_grad = nullptr;
	darr::v prior_cache = nullptr;
	darr::v2D mean_cache = nullptr;
	darr::v2D stddev_cache = nullptr;
	darr::v2D mean_grad = nullptr;
	darr::v2D stddev_grad = nullptr;

	darr::v2D likelihood_cache = nullptr;
	darr::v2D likelihood_grad = nullptr;
	darr::v model_likelihood_cache = nullptr;
	darr::v model_posterior = nullptr;


	Categorical_Distribution prior;
	Diag_Normal_distribution** models = nullptr;
	GaussianMixtureModel(const int& n_category, const int& n_variables) {
		this->n_category = n_category;
		this->n_variables = n_variables;
		prior.create(n_category);
		models = new Diag_Normal_distribution*[n_category];
		prior_cache = darr::alloc(n_category);
		prior_grad = darr::alloc(n_category);
		mean_cache = darr::alloc(n_category, n_variables);
		mean_grad = darr::alloc(n_category, n_variables);
		stddev_cache = darr::alloc(n_category, n_variables);
		stddev_grad = darr::alloc(n_category, n_variables);
		likelihood_cache = darr::alloc(n_category, n_variables);
		likelihood_grad = darr::alloc(n_category, n_variables);


		model_likelihood_cache = darr::alloc(n_category);
		model_posterior = darr::alloc(n_category);
		for (int i = 0; i < n_category; i++) {
			models[i] = new Diag_Normal_distribution(n_variables);
		}
	}
	void zero_grad() {
		prior.zero_grad();
		for (int i = 0; i < n_category; i++) {
			models[i]->zero_grad();
		}
	}

	void step(const double& learning_rate) {
		prior.step(learning_rate);
		for (int i = 0; i < n_category; i++) {
			models[i]->step(learning_rate);
		}
	}

	double fit(darr::v2D const& data, const int& n_data) {
		double gmm_avg_log_likelihood = 0.0;
		double p_x = 0.0;
		km::setZero(prior_grad, n_category);
		km::setZero(mean_grad, n_category, n_variables);
		km::setZero(stddev_grad, n_category, n_variables);
		prior.forward(prior_cache);
		for (int k = 0; k < n_category; k++) {
			models[k]->forward(mean_cache[k], stddev_cache[k]);
		}

		for (int n = 0; n < n_data; n++) {
			p_x = 0.0;
			for (int k = 0; k < n_category; k++) {
				model_likelihood_cache[k] = 1.0;
				for (int i = 0; i < n_variables; i++) {
					likelihood_cache[k][i] = km::univariate_gaussian_likelihood(data[n][i], mean_cache[k][i], stddev_cache[k][i]);
					model_likelihood_cache[k] *= likelihood_cache[k][i];
				}
				p_x += model_likelihood_cache[k] * prior_cache[k];
			}
			for (int k = 0; k < n_category; k++) {
				model_posterior[k] = prior_cache[k] * model_likelihood_cache[k] / p_x;
				prior_grad[k] -= model_likelihood_cache[k] / p_x;
				for (int i = 0; i < n_variables; i++) {
					likelihood_grad[k][i] = (model_likelihood_cache[k]/likelihood_cache[k][i]) * ( -prior_cache[k] / p_x ) ;
					mean_grad[k][i] -= likelihood_grad[k][i] * 
						(likelihood_cache[k][i] * (mean_cache[k][i] - data[n][i]) * 
						(mean_cache[k][i] - data[n][i]) * (mean_cache[k][i] - data[n][i]) / 
						(stddev_cache[k][i] * stddev_cache[k][i]));
					stddev_grad[k][i] += likelihood_grad[k][i] * (likelihood_cache[k][i] / stddev_cache[k][i])
						* (km::square((mean_cache[k][i] - data[n][i])/stddev_cache[k][i]) - 1.0);
				}
			}
			gmm_avg_log_likelihood += p_x;
		}
		
		prior.backward(prior_grad);
		for (int k = 0; k < n_category; k++) {
			models[k]->backward(mean_grad[k], stddev_grad[k]);
		}
		return gmm_avg_log_likelihood/(double)n_data;
	}

	double fit2(darr::v2D const& data, const int& n_data) {
		double gmm_avg_log_likelihood = 0.0;
		double p_x = 0.0;
		km::setZero(prior_grad, n_category);
		km::setZero(mean_grad, n_category, n_variables);
		km::setZero(stddev_grad, n_category, n_variables);
		prior.forward(prior_cache);
		for (int k = 0; k < n_category; k++) {
			models[k]->forward(mean_cache[k], stddev_cache[k]);
		}

		for (int n = 0; n < n_data; n++) {
			p_x = 0.0;
			for (int k = 0; k < n_category; k++) {
				model_likelihood_cache[k] = 1.0;
				for (int i = 0; i < n_variables; i++) {
					likelihood_cache[k][i] = km::univariate_gaussian_likelihood(data[n][i], mean_cache[k][i], stddev_cache[k][i]);
					model_likelihood_cache[k] *= likelihood_cache[k][i];
				}
				p_x += model_likelihood_cache[k] * prior_cache[k];
			}
			for (int k = 0; k < n_category; k++) {
				model_posterior[k] = prior_cache[k] * model_likelihood_cache[k] / p_x;
			}
			int argmax = km::argmax(model_posterior, n_category);
			prior_grad[argmax] -= 1.0 / prior_cache[argmax];
			for (int i = 0; i < n_variables; i++) {
				mean_grad[argmax][i] += (mean_cache[argmax][i] - data[n][i]) / (stddev_cache[argmax][i] * stddev_cache[argmax][i]);
				stddev_grad[argmax][i] += (1.0 - km::square((mean_cache[argmax][i] - data[n][i]) / stddev_cache[argmax][i])) / stddev_cache[argmax][i];
			}
			gmm_avg_log_likelihood += p_x;
		}

		prior.backward(prior_grad);
		for (int k = 0; k < n_category; k++) {
			models[k]->backward(mean_grad[k], stddev_grad[k]);
		}
		return gmm_avg_log_likelihood / (double)n_data;
	}

};