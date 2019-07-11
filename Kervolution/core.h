#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include <time.h>
#include <cmath>
#include <vector>
#include <cassert>
#define epsilon 0.00000001
using namespace std;

namespace km {
	random_device rdev;
	mt19937 rEngine(rdev());
	const double rhalf = 1.0;
	uniform_real_distribution<> uni_gen(-rhalf, rhalf);
	uniform_real_distribution<> rand0to1(0.0, 1.0);
	const double uni_gen_var = 0.333333333;
	void shuffle(int*& order_list, const int& size) {
		int rand_idx;
		int temp;
		for (int i = 0; i < size; i++) {
			order_list[i] = i;
		}
		for (int iter = 0; iter < 10; iter++) {
			for (int i = 0; i < size; i++) {
				rand_idx = std::rand() % size;
				temp = order_list[rand_idx];
				order_list[rand_idx] = order_list[i];
				order_list[i] = temp;
			}
		}
	}
	int randint(const int& divider) {
		return rand() % divider;
	}
	int argmax(double *const& src, const int& len) {
		int max_idx = 0;
		for (int i = 1; i < len; i++) {
			if (src[max_idx] < src[i]) {
				max_idx = i;
			}
		}
		return max_idx;
	}
	double accuarcy(int* &a, int* &b, const int& len) {
		double acc = 0.0;
		for (int i = 0; i < len; i++) {
			if (a[i] == b[i]) {
				acc += 1.0;
			}
		}
		return acc / (double)len;
	}
}
namespace km_1d {

	void fill_random(double*& arr, const int& len) {
		for (int i = 0; i < len; i++) {
			arr[i] = km::uni_gen(km::rEngine);
		}
	}

	void softmax(double*& dest, double* const& src, const int& len) {

		double base = 0.0;
		for (int i = 0; i < len; i++) {
			dest[i] = exp(src[i]);
			base += dest[i];
		}
		for (int i = 0; i < len; i++) {
			dest[i] = dest[i] / base;
		}
	}
	void swap(double*& arr1, double*& arr2, const int& len) {
		double temp;
		for (int i = 0; i < len; i++) {
			temp = arr1[i];
			arr1[i] = arr2[i];
			arr2[i] = temp;
		}
	}
	void fill_zero(double*& arr, const int& len) {
		for (int i = 0; i < len; i++) {
			arr[i] = 0.0;
		}
	}
	void copy(double*& dest, double* const& src, const int& len) {
		for (int i = 0; i < len; i++) {
			dest[i] = src[i];
		}
	}
	void add(double*& dest, double* const& src, const int& len) {
		for (int i = 0; i < len; i++) {
			dest[i] += src[i];
		}
	}
	double mean(double*& arr, const int& len) {
		double Mean = 0.0;
		for (int i = 0; i < len; i++) {
			Mean += arr[i];
		}
		return Mean / (double)len;
	}

	double variance(double*& arr, const double& mean, const int& len) {
		double var = 0.0;
		for (int i = 0; i < len; i++) {
			var += (arr[i] - mean) * (arr[i] - mean);
		}
		return var / (double)len;
	}
	double* alloc(const int& len) {
		double* arr = new double[len];
		return arr;
	}

	void free(double*& arr) {
		delete[] arr;
	}
	void guassian_noise(double*& arr, const double& mean, const double& variance, const int& len) {
		double stddev = sqrt(km::uni_gen_var / variance);
		for (int i = 0; i < len; i++) {
			arr[i] = (km::uni_gen(km::rEngine) / stddev) + mean;
		}
	}

	void guassian_norm(double*& arr, const double& mean, const double& variance, const int& len) {
		double m = km_1d::mean(arr, len);
		double v = km_1d::variance(arr, m, len);
		double stddev = sqrt(v / variance + epsilon);
		for (int i = 0; i < len; i++) {
			arr[i] = ((arr[i] - m) / stddev) + mean;
		}
	}
	void argmax(int& dest, double *const& src, const int& len) {
		int max_idx = 0;
		for (int i = 1; i < len; i++) {
			if (src[max_idx] < src[i]) {
				max_idx = i;
			}
		}
		dest = max_idx;
	}
}

namespace km_2d {

	void fill_random(double**& arr, const int& row, const int& col) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				arr[i][j] = km::uni_gen(km::rEngine);
			}
		}
	}
	void fill_zero(double**& arr, const int& row, const int& col) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				arr[i][j] = 0.0;
			}
		}
	}
	double mean(double**& arr, const int& row, const int& col) {
		double Mean = 0.0;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				Mean += arr[i][j];
			}
		}
		return Mean / (row*col);
	}

	double variance(double**& arr, const double& mean, const int& row, const int& col) {
		double var = 0.0;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				var += (arr[i][j] - mean) * (arr[i][j] - mean);
			}
		}
		return var / (row*col);
	}

	void add(double**& dest, double** const& src, const int& row, const int& col) {

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				dest[i][j] += src[i][j];
			}
		}
	}

	void copy(double**& dest, double** const& src, const int& row, const int& col) {

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				dest[i][j] = src[i][j];
			}
		}
	}

	double CEloss(double** &dLoss_container, double** const& output_port, int* const& target, const int& row, const int& col) {
		double mean_loss = 0.0;
		double base = 0.0;
		for (int m = 0; m < row; m++) {
			base = 0.0;
			for (int i = 0; i < col; i++) {
				dLoss_container[m][i] = exp(output_port[m][i]);
				base += dLoss_container[m][i];
			}
			for (int i = 0; i < col; i++) {
				dLoss_container[m][i] /= base; // dLoss_container[m] <- softmax(output_port[m])

				if (target[m] == i) { /* dLoss_conatainer[m] <- กำ CELoss(softmax(output_port[m]), taget[m]))
																__________________________________________
																		กำ output_port[m]					*/
					mean_loss -= log(dLoss_container[m][i]);
					dLoss_container[m][i] -= 1.0;
				}
			}
		}
		return mean_loss / row;
	}

	double MSEloss(double** &dLoss_container, double** const& output_port, double** const& target, const int& row, const int& col) {
		double mean_loss = 0.0;
		for (int m = 0; m < row; m++) {
			//cout << "********************************" << endl;
			for (int i = 0; i < col; i++) {
				//cout << output_port[m][i] << " vs " << target[m][i] << endl;

				mean_loss += (output_port[m][i] - target[m][i])*(output_port[m][i] - target[m][i]);
				dLoss_container[m][i] = 2.0 * (output_port[m][i] - target[m][i]);
			}
		}
		return mean_loss / row;
	}

	void guassian_noise(double**& arr, const double& mean, const double& variance, const int& row, const int& col) {
		double stddev = sqrt(km::uni_gen_var / variance);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				arr[i][j] = (km::uni_gen(km::rEngine) / stddev) + mean;
			}
		}
	}
	void free(double**& arr, const int& row) {
		for (int i = 0; i < row; i++) {
			delete[] arr[i];
		}
		delete[] arr;
	}
	double** alloc(const int& row, const int& col) {
		double** arr = new double*[row];
		for (int i = 0; i < row; i++) {
			arr[i] = new double[col];
		}
		return arr;
	}
	void guassian_norm(double**& arr, const double& mean, const double& variance, const int& row, const int& col) {
		double m = km_2d::mean(arr, row, col);
		double v = km_2d::variance(arr, m, row, col);
		double stddev = sqrt(v / variance + epsilon);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				arr[i][j] = ((arr[i][j] - m) / stddev) + mean;
			}
		}
	}

	void argmax(int*& dest, double **const& src, const int& row, const int& col) {
		int max_idx;
		for (int i = 0; i < row; i++) {
			max_idx = 0;
			for (int j = 1; j < col; j++) {
				if (src[i][max_idx] < src[i][j]) {
					max_idx = j;
				}
			}
			dest[i] = max_idx;
		}
	}
}