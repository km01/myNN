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

namespace km {

	random_device rdev;
	mt19937 rEngine(rdev());
	uniform_real_distribution<> U(0.0, 1.0);
	normal_distribution<> STD(0.0, 1.0);

	using h_w = pair<int, int>;
	using c_h_w = pair<int, h_w>;
	using Darr = double*;
	using DarrList = double**;
	using Iarr = int*;
	using IarrList = int**;

	Darr alloc(const int& len) {
		return new double[len];
	}

	void free(Darr& darr) {
		if (darr != nullptr) {
			delete[] darr;
		}
	}

	void resize(Darr& darr, const int& len) {
		free(darr);
		darr = alloc(len);
	}


	DarrList alloc(const int& list_len, const int& len) {
		DarrList list = new double* [list_len];
		for (int i = 0; i < list_len; i++) {
			list[i] = new double[len];
		}
		return list;
	}

	void free(DarrList& list, const int& list_len) {
		if (list != nullptr) {
			for (int i = 0; i < list_len; i++) {
				free(list[i]);
			}
			delete[] list;
		}
	}

	void resize(DarrList& list, const int& n_old, const int& n_new, const int& len) {
		free(list, n_old);
		list = alloc(n_new, len);
	}

	void setNormal(DarrList& target, const int& list_size, const int& target_len, const double& mean, const double& stddev) {
		for (int i = 0; i < list_size; i++) {
			for (int j = 0; j < target_len; j++) {
				target[i][j] = stddev * STD(rEngine) + mean;
			}
		}
	}

	void setNormal(Darr& target, const int& target_len, const double& mean, const double& stddev) {
		for (int i = 0; i < target_len; i++) {
			target[i] = stddev * STD(rEngine) + mean;
		}
	}

	void setZero(Darr& target, const int& target_len) {
		for (int i = 0; i < target_len; i++) {
			target[i] = 0.0;
		}
	}

	void copy(Darr& dst, Darr const& src, const int& target_len) {
		for (int i = 0; i < target_len; i++) {
			dst[i] = src[i];
		}
	}

	namespace LossFn {
		double CELoss(Darr& grad, Darr const& pred, const int& target, const int& len) {
			double loss = 0.0;
			double base = 0.0;
			for (int i = 0; i < len; i++) {
				grad[i] = exp(pred[i]);
				base += grad[i];
			}
			for (int i = 0; i < len; i++) {
				grad[i] /= base;
				if (target == i) {
					loss = -log(grad[i]);
					grad[i] -= 1.0;
				}
			}
			return loss;
		}
		
		double L2Loss(Darr& grad, Darr const& pred, const Darr& target, const int& len) {
			double loss = 0.0;
			for (int i = 0; i < len; i++) {
				grad[i] = 0.5 * (pred[i] - target[i]);
				loss += (pred[i] - target[i]) * (pred[i] - target[i]);
			}
			return loss;
		}

		double BCELoss(Darr& grad, Darr const& pred, const Darr& target, const int& len) {
			double loss = 0.0;
			for (int i = 0; i < len; i++) {
				loss += -target[i] * log(pred[i]) - (1.0 - target[i]) * log(1.0 - pred[i]);
				grad[i] = -(target[i] / pred[i]) - (1.0 - target[i]) / (pred[i] - 1.0);
			}
			return loss;
		}
	}

	Iarr ialloc(const int& len) {
		return new int[len];
	}

	void free(Iarr& iarr) {
		if (iarr != nullptr) {
			delete[] iarr;
		}
	}

	void resize(Iarr& iarr, const int& len) {
		free(iarr);
		iarr = ialloc(len);
	}


	IarrList ialloc(const int& list_len, const int& len) {
		IarrList list = new int* [list_len];
		for (int i = 0; i < list_len; i++) {
			list[i] = new int[len];
		}
		return list;
	}

	void free(IarrList& list, const int& list_len) {
		if (list != nullptr) {
			for (int i = 0; i < list_len; i++) {
				free(list[i]);
			}
			delete[] list;
		}
	}

	void resize(IarrList& list, const int& n_old, const int& n_new, const int& len) {
		free(list, n_old);
		list = ialloc(n_new, len);
	}

	void setZero(Iarr& target, const int& target_len) {
		for (int i = 0; i < target_len; i++) {
			target[i] = 0;
		}
	}

	bool bernoulli_sampling(const double& true_prob) {
		if (km::U(rEngine) < true_prob) {
			return true;
		}
		return false;
	}

	int randint(const int& divider) {
		return rand() % divider;
	}

	int argmax(Darr const& src, const int& len) {
		int max_idx = 0;
		for (int i = 1; i < len; i++) {
			if (src[max_idx] < src[i]) {
				max_idx = i;
			}
		}
		return max_idx;
	}


	int multinomial_sampling(Darr const& prob_vec, const int& len) {
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

}