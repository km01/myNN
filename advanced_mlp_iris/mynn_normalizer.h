#pragma once
#include "mynn_core.h"

class batch_normalizer {//feature map 의 channel 마다 mean과 variance를 따로 구함
public:
	int nb_group;
	int element_size;
	int batch_size;
	bool batch_memory_allocated;
	double denumerator;
	double* learned_scale;
	double* learned_shift;

	double* gradients_scale;
	double* gradients_shift;

	double* mean;
	double* variance;

	double* dLossdmean;
	double* dLossdvariance;

	double* stddev;
	double** batch_normalized_container;
	double* single_normalized_container;
	int step;

	double exp_moving;
	double *TEST_MEAN;
	double *TEST_VARIANCE;
	double *TEST_STD_DEV;

	batch_normalizer(int element_sz, int nb_g) {
		create(element_sz, nb_g);
	}
	void store_unit(ofstream& fout) {
		fout << nb_group << endl;
		fout << element_size << endl;
		fout << exp_moving << endl;

		for (int i = 0; i < nb_group; i++) {

			fout << learned_scale[i] << endl;
			fout << learned_shift[i] << endl;
		}
		for (int i = 0; i < nb_group; i++) {
			fout << TEST_MEAN[i] << endl;
			fout << TEST_VARIANCE[i] << endl;
			fout << TEST_STD_DEV[i] << endl;
		}
	}
	void zero_grad() {
		for (int i = 0; i < nb_group; i++) {
			gradients_scale[i] = 0.0;
			gradients_shift[i] = 0.0;
		}
	}
	void feed_grad_itself(const double& learning_rate) {
		for (int i = 0; i < nb_group; i++) {
			learned_scale[i] -= gradients_scale[i] * learning_rate;
			learned_shift[i] -= gradients_shift[i] * learning_rate;
		}
	}

	void load(ifstream& fin) {
		string buffer;
		getline(fin, buffer);
		exp_moving = atof(buffer.c_str());
		for (int i = 0; i < nb_group; i++) {
			getline(fin, buffer);
			learned_scale[i] = atof(buffer.c_str());
			getline(fin, buffer);
			learned_shift[i] = atof(buffer.c_str());
		}
		for (int i = 0; i < nb_group; i++) {
			getline(fin, buffer);
			TEST_MEAN[i] = atof(buffer.c_str());
			getline(fin, buffer);
			TEST_VARIANCE[i] = atof(buffer.c_str());
			getline(fin, buffer);
			TEST_STD_DEV[i] = atof(buffer.c_str());
		}
		batch_memory_allocated = false;
	}

	void create(int element_sz, int nb_g) {
		nb_group = nb_g;
		batch_size = -1;
		element_size = element_sz;
		learned_scale = new double[nb_group];
		learned_shift = new double[nb_group];
		gradients_scale = new double[nb_group];
		gradients_shift = new double[nb_group];
		for (int i = 0; i < nb_group; i++) {
			learned_scale[i] = 1.0;
			learned_shift[i] = 0.0;
		}
		mean = new double[nb_group];
		variance = new double[nb_group];
		dLossdmean = new double[nb_group];
		dLossdvariance = new double[nb_group];
		stddev = new double[nb_group];
		single_normalized_container = new double[element_size * nb_g];
		TEST_MEAN = km_1d::alloc(nb_group);
		TEST_VARIANCE = km_1d::alloc(nb_group);
		TEST_STD_DEV = km_1d::alloc(nb_group);
		km_1d::fill_zero(TEST_MEAN, nb_group);
		km_1d::fill_zero(TEST_VARIANCE, nb_group);
		km_1d::fill_zero(TEST_STD_DEV, nb_group);
		exp_moving = 0.95;
		batch_memory_allocated = false;
	}

	void alloc_batch_memory(int new_batch_size) {
		if (batch_memory_allocated == true) {
			km_2d::free(batch_normalized_container, batch_size);
		}
		batch_size = new_batch_size;
		denumerator = (double)batch_size * (double)element_size;
		batch_normalized_container = km_2d::alloc(batch_size, element_size*nb_group);
		batch_memory_allocated = true;
	}
	
	~batch_normalizer() {
		if (batch_memory_allocated == true) {
			km_2d::free(batch_normalized_container, batch_size);
		}

		delete[] learned_scale;
		delete[] learned_shift;
		delete[] mean;
		delete[] variance;
		delete[] stddev;
	}

	void batch_back_prop(double** inner_deltaflow, double** un_normalized_container) { // dLoss/dztelda

		for (int g = 0; g < nb_group; g++) {
			dLossdmean[g] = 0.0;
			dLossdvariance[g] = 0.0;
			step = g * element_size;

			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					gradients_shift[g] += inner_deltaflow[m][step + i];
					gradients_scale[g] += inner_deltaflow[m][step + i] * batch_normalized_container[m][step + i];
					inner_deltaflow[m][step + i] *= learned_scale[g];
					dLossdmean[g] += -pow(variance[g], -0.5)*inner_deltaflow[m][g*element_size + i];
					dLossdvariance[g] += (0.5)*(mean[g] - un_normalized_container[m][step + i])*pow(variance[g], -1.5) * inner_deltaflow[m][step + i];
				}
			}

			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					inner_deltaflow[m][step + i] = (inner_deltaflow[m][step + i] / stddev[g]) + dLossdvariance[g] * (2.0 / denumerator)*(un_normalized_container[m][step + i] - mean[g]) + (dLossdmean[g] / denumerator);
				}
			}
		}
	}

	void update_for_testing() {
		for (int i = 0; i < nb_group; i++) {
			TEST_MEAN[i] = exp_moving * TEST_MEAN[i] + (1.0 - exp_moving) * mean[i];
			TEST_VARIANCE[i] = exp_moving * TEST_VARIANCE[i] + (1.0 - exp_moving) * variance[i];
			TEST_STD_DEV[i] = exp_moving * TEST_STD_DEV[i] + (1.0 - exp_moving) * stddev[i];
		}
	}
	void batch_leak_back_prop(double** inner_deltaflow, double** un_normalized_container) { // dLoss/dztelda

		for (int g = 0; g < nb_group; g++) {
			dLossdmean[g] = 0.0;
			dLossdvariance[g] = 0.0;
			step = g * element_size;

			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					inner_deltaflow[m][step + i] *= learned_scale[g];
					dLossdmean[g] += -pow(variance[g], -0.5)*inner_deltaflow[m][g*element_size + i];
					dLossdvariance[g] += (0.5)*(mean[g] - un_normalized_container[m][step + i])*pow(variance[g], -1.5) * inner_deltaflow[m][step + i];
				}
			}

			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					inner_deltaflow[m][step + i] = (inner_deltaflow[m][step + i] / stddev[g]) + dLossdvariance[g] * (2.0 / denumerator)*(un_normalized_container[m][step + i] - mean[g]) + (dLossdmean[g] / denumerator);
				}
			}
		}

	}
	void single_leak_back_prop(double* single_inner_deltaflow, double* single_un_normalized_container) { // dLoss/dztelda

		for (int g = 0; g < nb_group; g++) {
			dLossdmean[g] = 0.0;
			dLossdvariance[g] = 0.0;
			step = g * element_size;

			for (int i = 0; i < element_size; i++) {
				single_inner_deltaflow[step + i] *= learned_scale[g];
				single_inner_deltaflow[step + i] /= TEST_STD_DEV[g];
			}
		}
	}

	void single_normalize(double* single_destination, double* single_un_normalized_container) {
		for (int g = 0; g < nb_group; g++) {
			step = g * element_size;
			for (int i = 0; i < element_size; i++) {
				single_normalized_container[step + i] = (single_un_normalized_container[step + i] - TEST_MEAN[g]) / TEST_STD_DEV[g];
				single_destination[step + i] = learned_scale[g] * single_normalized_container[step + i] + learned_shift[g];
			}
		}
	}

	void batch_normalize(double** destination, double** un_normalized_container) {

		for (int g = 0; g < nb_group; g++) {
			step = g * element_size;
			mean[g] = 0.0;
			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					mean[g] += un_normalized_container[m][step + i];
				}
			}
			mean[g] = mean[g] / denumerator;
		}

		for (int g = 0; g < nb_group; g++) {
			variance[g] = 0.0;
			step = g * element_size;
			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {

					variance[g] += (un_normalized_container[m][step + i] - mean[g])*(un_normalized_container[m][step + i] - mean[g]);
				}
			}

			variance[g] = (variance[g] / (denumerator)) + epsilon;
			stddev[g] = sqrt(variance[g]);
		}
		update_for_testing();
		for (int g = 0; g < nb_group; g++) {
			step = g * element_size;
			for (int m = 0; m < batch_size; m++) {
				for (int i = 0; i < element_size; i++) {
					batch_normalized_container[m][step + i] = (un_normalized_container[m][step + i] - mean[g]) / stddev[g];
					destination[m][step + i] = learned_scale[g] * batch_normalized_container[m][step + i] + learned_shift[g];
				}
			}
		}
	}
};