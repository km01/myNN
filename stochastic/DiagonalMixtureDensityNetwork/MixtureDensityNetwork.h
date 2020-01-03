#pragma once
#include "Unit.h"

class DiagCovMDN : public Sequential {
public: 

	int n_group = 0; int n_dim = 0;
	Darr nn_out = nullptr;
	Darr mixing_coeff = nullptr;
	DarrList mean = nullptr;
	DarrList stddev = nullptr;

	Darr nn_out_grad = nullptr;
	Darr conditional = nullptr;

	double lh_constant = 0;
	void setMDN(const int& n_group, const int& n_dim) {
		this->n_group = n_group;
		this->n_dim = n_dim;
		lh_constant = pow(2.0 * 3.141592, -0.5 * n_dim);
		nn_out = alloc(n_group * n_dim * 2 + n_group);
		mixing_coeff = alloc(n_group);
		mean = alloc(n_group, n_dim);
		stddev = alloc(n_group, n_dim);
		nn_out_grad = alloc(n_group * n_dim * 2 + n_group);
		conditional = alloc(n_group);
	}
	~DiagCovMDN() {
		free(nn_out);
		free(mixing_coeff);
		free(mean, n_group);
		free(stddev, n_group);
		free(nn_out_grad);
		free(conditional);
	}
	virtual void setCache(const int& cache_size) {
		for (int i = 0; i < sequential_len; i++) {
			list[i]->setCache(cache_size);
		}
		in_port = list[0]->in_port;
	}

	double fit(Darr& x, Darr& y) {

		static double denom = 0.0;
		static double det = 0.0;
		static double mahala = 0.0;
		charge(x, 0);
		forward(nn_out, 0);
		denom = 0.0;
		for (int k = 0; k < n_group; k++) {
			mixing_coeff[k] = exp(nn_out[k]);
			denom += mixing_coeff[k];
		}
		for (int k = 0; k < n_group; k++) {
			mixing_coeff[k] /= denom;
		}
		
		double likelihood_of_y = 0.0;
	
		for (int k = 0; k < n_group; k++) {
			det = 1.0;
			mahala = 0.0;
			for (int i = 0; i < n_dim; i++) {
				mean[k][i] = nn_out[n_group + k * n_dim + i];
				stddev[k][i] = exp(nn_out[n_group + n_group * n_dim + k * n_dim + i]);
				det *= stddev[k][i]*stddev[k][i];
				mahala += (y[i] - mean[k][i]) * (y[i] - mean[k][i]) / (stddev[k][i]*stddev[k][i]);
			}
			conditional[k] = lh_constant * pow(det, -0.5) * exp(-0.5 * mahala);
			likelihood_of_y += mixing_coeff[k] * conditional[k];
		}

		for (int k = 0; k < n_group; k++) {
			nn_out_grad[k] = mixing_coeff[k] * (1.0 - (conditional[k] / likelihood_of_y));
			for (int i = 0; i < n_dim; i++) {
				nn_out_grad[n_group + k * n_dim + i] =  (conditional[k] * (mean[k][i] - y[i]))/ (stddev[k][i] * likelihood_of_y);
				nn_out_grad[n_group + n_group * n_dim + k * n_dim + i] 
					= (1.0 - ((mean[k][i] - y[i]) / stddev[k][i]) * ((mean[k][i] - y[i]) / stddev[k][i])) * 
					(conditional[k] /( likelihood_of_y*stddev[k][i]));
			}
		}
		backward(nn_out_grad, 0);
		return likelihood_of_y;
	}
};