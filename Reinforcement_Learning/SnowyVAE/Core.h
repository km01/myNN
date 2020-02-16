#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <time.h>
#include <cmath>
#include <vector>
#include <cassert>

using namespace std;

random_device rdev;
mt19937 rEngine(rdev());
uniform_real_distribution<> U(0.0, 1.0);
normal_distribution<> STD(0.0, 1.0);
using h_w = pair<int, int>;
using c_h_w = pair<int, h_w>;

namespace darr {

	using v = double*;
	using v2D = double**;

	v alloc(const int& len) {
		return new double[len];
	}

	void free(v& darr) {
		if (darr != nullptr) {
			delete[] darr;
			darr = nullptr;
		}
	}

	void resize(v& darr, const int& len) {
		free(darr);
		darr = alloc(len);
	}


	v2D alloc(const int& list_len, const int& len) {
		v2D list = new double* [list_len];
		for (int i = 0; i < list_len; i++) {
			list[i] = new double[len];
		}
		return list;
	}

	void free(v2D& list, const int& list_len) {
		if (list != nullptr) {
			for (int i = 0; i < list_len; i++) {
				free(list[i]);
			}
			delete[] list;
			list = nullptr;
		}
	}

	void resize(v2D& list, const int& n_old, const int& n_new, const int& len) {
		free(list, n_old);
		list = alloc(n_new, len);
	}
}

namespace iarr {

	using v = int*;
	using v2D = int**;

	v alloc(const int& len) {
		return new int[len];
	}

	void free(v& iarr) {
		if (iarr != nullptr) {
			delete[] iarr;
			iarr = nullptr;
		}
	}

	void resize(v& iarr, const int& len) {
		free(iarr);
		iarr = alloc(len);
	}


	v2D alloc(const int& list_len, const int& len) {
		v2D list = new int* [list_len];
		for (int i = 0; i < list_len; i++) {
			list[i] = new int[len];
		}
		return list;
	}

	void free(v2D& list, const int& list_len) {
		if (list != nullptr) {
			for (int i = 0; i < list_len; i++) {
				free(list[i]);
			}
			delete[] list;
			list = nullptr;
		}
	}

	void resize(v2D& list, const int& n_old, const int& n_new, const int& len) {
		free(list, n_old);
		list = alloc(n_new, len);
	}
}


namespace km {

	double CELoss_2D(double*& loss_grad, double* const& raw_output, double* const& target, const int& n_channel, const int& plain_size) {
		double loss = 0.0;
		double base = 0.0;
		double sfmx = 0.0;
		for (int i = 0; i < plain_size; i++) {
			base = 0.0;
			for (int c = 0; c < n_channel; c++) {
				loss_grad[c * plain_size + i] = exp(raw_output[c * plain_size + i]);
				base += loss_grad[c * plain_size + i];
			}
			for (int c = 0; c < n_channel; c++) {
				sfmx = loss_grad[c * plain_size + i] / base;
				loss_grad[c * plain_size + i] = sfmx - target[c * plain_size + i];
				loss -= target[c * plain_size + i] * log(sfmx);
			}
		}
		return loss;
	}

	int randint(const int& divider) {
		return rand() % divider;
	}

	double square(const double& x) {
		return x * x;
	}

	int argmax(double* const& src, const int& len) {
		int max_idx = 0;
		for (int i = 1; i < len; i++) {
			if (src[max_idx] < src[i]) {
				max_idx = i;
			}
		}
		return max_idx;
	}

	bool bernoulli_sampling(const double& true_prob) {
		if (U(rEngine) < true_prob) {
			return true;
		}
		return false;
	}

	int multinoulli_sampling(double* const& prob_vec, const int& len) {
		double rand0to1 = U(rEngine);
		double amount = 0.0;
		for (int i = 0; i < len - 1; i++) {
			amount += prob_vec[i];
			if (amount > rand0to1) {
				return i;
			}
		}
		return len - 1;
	}

	void normal_sampling(double*& arr, const int& len, const double& mean, const double& variance) {
		double stddev = sqrt(0.3333333333333333 / variance);
		for (int i = 0; i < len; i++) {
			arr[i] = ((2.0 * U(rEngine) - 1.0) / stddev) + mean;
		}
	}

	void normal_sampling2(double*& arr, const int& len, const double& mean, const double& variance) {
		for (int i = 0; i < len; i++) {
			arr[i] = sqrt(variance) * STD(rEngine) + mean;
		}
	}

	void setZero(double* const& arr, const int& len) {
		for (int i = 0; i < len; i++) {
			arr[i] = 0.0;
		}
	}

}

