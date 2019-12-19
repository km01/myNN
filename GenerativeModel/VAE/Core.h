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

	using Darr = double*;
	using DarrList = double**;

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
		DarrList list = new double*[list_len];
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

	namespace LossFn {
		double CELoss(Darr& grad, Darr& pred, const int& target, const int& len) {
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
		double L2Loss(Darr& grad, const Darr& pred, const Darr& target, const int& len) {
			double loss = 0.0;
			for (int i = 0; i < len; i++) {
				grad[i] = 0.5 * (pred[i] - target[i]);
				loss += (pred[i] - target[i]) * (pred[i] - target[i]);
			}
			return loss;
		}
	}
}