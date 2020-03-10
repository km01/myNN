#pragma once
#include "Core.h"

class LatentCodeSpace {
public:
	int n_group = 0;
	int centroid_len = 0;
	darr::v2D centroid_mean = nullptr;
	darr::v2D centroid_mean_grad = nullptr;

	darr::v2D centroid_pre_stddev = nullptr;
	darr::v2D centroid_pre_stddev_grad = nullptr;

	darr::v pre_categorical_prior = nullptr;
	darr::v pre_categorical_prior_grad = nullptr;
	darr::v categorical_prior_cache = nullptr;
	darr::v categorical_prior_grad_cache = nullptr;


	darr::v2D likelihood_each_variables_cache = nullptr;
	darr::v likelihood_cache = nullptr;
	

	LatentCodeSpace(const int& n_group, const int& centroid_len) {
		this->n_group = n_group;
		this->centroid_len = centroid_len;
		
		centroid_mean = darr::alloc(this->n_group, this->centroid_len);
		centroid_mean_grad = darr::alloc(this->n_group, this->centroid_len);
		centroid_pre_stddev = darr::alloc(this->n_group, this->centroid_len);
		centroid_pre_stddev_grad = darr::alloc(this->n_group, this->centroid_len);
		pre_categorical_prior = darr::alloc(this->n_group);
		pre_categorical_prior_grad = darr::alloc(this->n_group);
		categorical_prior_cache = darr::alloc(this->n_group);
		categorical_prior_grad_cache = darr::alloc(this->n_group);
		likelihood_each_variables_cache = darr::alloc(this->n_group, this->centroid_len);
		likelihood_cache = darr::alloc(this->n_group);

		for (int k = 0; k < n_group; k++) {
			km::normal_sampling(centroid_mean[k], centroid_len, 0.0, 1.0);
			km::setZero(centroid_pre_stddev[k], centroid_len);
		}
		km::normal_sampling(pre_categorical_prior, n_group, 0.0, 1.0);
	}

	void zero_grad() {
		for (int k = 0; k < n_group; k++) {
			pre_categorical_prior_grad[k] = 0.0;
			for (int i = 0; i < centroid_len; i++) {
				centroid_mean_grad[k][i] = 0.0;
				centroid_pre_stddev_grad[k][i] = 0.0;
			}
		}
	}

	void show_categorical_prior() {
		double categorical_prior_denom = 0.0; //omega
		for (int k = 0; k < n_group; k++) {
			categorical_prior_cache[k] = exp(pre_categorical_prior[k]);
			categorical_prior_denom += categorical_prior_cache[k];
		}
		cout << " categorical prior of clustering field :" << endl;
		for (int k = 0; k < n_group; k++) {
			cout << categorical_prior_cache[k] / categorical_prior_denom << "	"<<endl;
		}

		for (int k = 0; k < n_group; k++) {
			cout << "mean[" << k << "]:";
			for (int i = 0; i < centroid_len; i++) {
				cout << centroid_mean[k][i] << " ";
			}
			cout << endl;
			cout << "stddev[" << k << "]:";
			for (int i = 0; i < centroid_len; i++) {
				if (centroid_pre_stddev[k][i] < 0.0) {
					cout << exp(centroid_pre_stddev[k][i])<< " ";
				}
				else {
					cout << centroid_pre_stddev[k][i] + 1.0 << " ";
				}
			}
			cout << endl;
		}

		cout << endl;

	}
	void step(const double& learning_rate) {
		for (int k = 0; k < n_group; k++) {
			pre_categorical_prior[k] -= learning_rate *  pre_categorical_prior_grad[k];
			for (int i = 0; i < centroid_len; i++) {
				centroid_mean[k][i] -= learning_rate * centroid_mean_grad[k][i];
				centroid_pre_stddev[k][i] -= learning_rate * centroid_pre_stddev_grad[k][i];
			}
		}
	}

	double calculate(darr::v const& code) {
		double stddev = 0.0;
		double stddev_grad = 0.0;
		double categorical_prior_denom = 0.0;

		for (int k = 0; k < n_group; k++) {
			likelihood_cache[k] = 1.0;
			categorical_prior_cache[k] = exp(pre_categorical_prior[k]);
			categorical_prior_denom += categorical_prior_cache[k];
			for (int i = 0; i < centroid_len; i++) {
				if (centroid_pre_stddev[k][i] < 0) {
					stddev = exp(centroid_pre_stddev[k][i]);
				}
				else {
					stddev = centroid_pre_stddev[k][i] + 1.0;
				}
				likelihood_each_variables_cache[k][i] = km::univariate_gaussian_pdf(code[i], centroid_mean[k][i], stddev);
				likelihood_cache[k] *= likelihood_each_variables_cache[k][i];
			}
		}

		double max_joint = -1.0;
		int argmax = 0;
		double p_z = 0.0;
		for (int k = 0; k < n_group; k++) {
			categorical_prior_cache[k] = categorical_prior_cache[k] / categorical_prior_denom;
			p_z += categorical_prior_cache[k] * likelihood_cache[k];
			if (max_joint < categorical_prior_cache[k] * likelihood_cache[k]) {
				max_joint = categorical_prior_cache[k] * likelihood_cache[k];
				argmax = k;
			}
			categorical_prior_grad_cache[k] = 0.0;
		}

		for (int i = 0; i < centroid_len; i++) {
			if (centroid_pre_stddev[argmax][i] < 0) {
				stddev = exp(centroid_pre_stddev[argmax][i]);
			}
			else {
				stddev = centroid_pre_stddev[argmax][i] + 1.0;
			}
			centroid_mean_grad[argmax][i] += (centroid_mean[argmax][i] - code[i]) / (stddev * stddev);
			stddev_grad = (1.0 - km::square((code[i] - centroid_mean[argmax][i]) / stddev)) / stddev;
			if (centroid_pre_stddev[argmax][i] < 0) {
				centroid_pre_stddev_grad[argmax][i] += stddev_grad * stddev;
			}
			else {
				centroid_pre_stddev_grad[argmax][i] += stddev_grad;
			}
		}
		categorical_prior_grad_cache[argmax] += -1.0 / categorical_prior_cache[argmax];
		for (int k = 0; k < n_group; k++) {
			categorical_prior_grad_cache[k] += likelihood_cache[k] / p_z;
		}

		double likelihood_grad = 0.0;
		for (int k = 0; k < n_group; k++) {
			for (int i = 0; i < centroid_len; i++) {
				if (centroid_pre_stddev[k][i] < 0) {
					stddev = exp(centroid_pre_stddev[k][i]);
				}
				else {
					stddev = centroid_pre_stddev[k][i] + 1.0;
				}
				likelihood_grad = categorical_prior_cache[k] / p_z;
				likelihood_grad *= likelihood_cache[k];
				centroid_mean_grad[k][i] += likelihood_grad * (code[i] - centroid_mean[k][i])/stddev*stddev;
				stddev_grad = likelihood_grad * (1.0 / stddev)  * (km::square((code[i] - centroid_mean[k][i]) / stddev) - 1.0);

			}
		}


		for (int k = 0; k < n_group; k++) {
			for (int k_iter = 0; k_iter < n_group; k_iter++) {
				if (k == k_iter) {
					pre_categorical_prior_grad[k_iter] += categorical_prior_grad_cache[k] * exp(pre_categorical_prior[k_iter]) * (categorical_prior_denom - exp(pre_categorical_prior[k]) / (categorical_prior_denom * categorical_prior_denom));
				}
				else {
					pre_categorical_prior_grad[k_iter] += categorical_prior_grad_cache[k] * exp(pre_categorical_prior[k_iter]) * (-exp(pre_categorical_prior[k]) / (categorical_prior_denom * categorical_prior_denom));
				}
			}
		}
		return log(p_z) - log(likelihood_cache[argmax]) - log(categorical_prior_cache[argmax]);
	}
};