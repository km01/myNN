#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <random>
#include <cmath>
#define epsilon 0.00000001
using namespace std;
random_device rdev;
mt19937 rEngine(rdev());

const double rhalf = 1.0;
uniform_real_distribution<> uni_gen(-rhalf, rhalf);
uniform_real_distribution<> RandGen(0.0, rhalf);
const double uni_gen_var = 0.333333333;

namespace km_1d {

	void fill_random(double* arr, const int& len) {
		for (int i = 0; i < len; i++) {
			arr[i] = uni_gen(rEngine);
		}
	}

	void fill_zero(double* arr, const int& len) {
		for (int i = 0; i < len; i++) {
			arr[i] = 0.0;
		}
	}
	void copy(double* dest, double* const& src, const int& len) {
		for (int i = 0; i < len; i++) {
			dest[i] = src[i];
		}
	}
	double mean(double* arr, const int& len) {
		double Mean = 0.0;
		for (int i = 0; i < len; i++) {
			Mean += arr[i];
		}
		return Mean / (double)len;
	}

	double variance(double* arr, const double& mean, const int& len) {
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

	void free(double* arr) {
		delete[] arr;
	}
	void guassian_noise(double* arr, double mean, double variance, int len) {
		double stddev = sqrt(uni_gen_var / variance);
		for (int i = 0; i < len; i++) {
			arr[i] = (uni_gen(rEngine) / stddev) + mean;
		}
	}

	void guassian_norm(double* arr, const double& mean, const double& variance, const int& len) {
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

	void fill_random(double** arr, const int& row, const int& col) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				arr[i][j] = uni_gen(rEngine);
			}
		}
	}
	void fill_zero(double**arr, const int& row, const int& col) {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				arr[i][j] = 0.0;
			}
		}
	}
	double mean(double** arr, const int& row, const int& col) {
		double Mean = 0.0;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				Mean += arr[i][j];
			}
		}
		return Mean / (row*col);
	}

	double variance(double** arr, const double& mean, const int& row, const int& col) {
		double var = 0.0;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				var += (arr[i][j] - mean) * (arr[i][j] - mean);
			}
		}
		return var / (row*col);
	}

	void copy(double** dest, double** const& src, const int& row, const int& col) {

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				dest[i][j] = src[i][j];
			}
		}
	}

	void guassian_noise(double** arr, const double& mean, const double& variance, const int& row, const int& col) {
		double stddev = sqrt(uni_gen_var / variance);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				arr[i][j] = (uni_gen(rEngine) / stddev) + mean;
			}
		}
	}
	void free(double**arr, const int& row) {
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
	void guassian_norm(double** arr, const double& mean, const double& variance, const int& row, const int& col) {
		double m = km_2d::mean(arr, row, col);
		double v = km_2d::variance(arr, m, row, col);
		double stddev = sqrt(v / variance + epsilon);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				arr[i][j] = ((arr[i][j] - m) / stddev) + mean;
			}
		}
	}

	void argmax(int* dest, double **const& src, const int& row, const int& col) {
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