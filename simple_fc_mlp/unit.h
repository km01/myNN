#pragma once
#include "mynn_core.h"
#include "activation.h"
enum UNIT { PERCEPTRONS };
class unit {
public:

	unit() {};
	UNIT unit_type;

	virtual void single_load_input(double* const& single_input) {
		km_1d::copy(single_input_container, single_input, input_size);
	}
	virtual void batch_load_inputs(double** const& mini_batch_x) {
		km_2d::copy(batch_input_container, mini_batch_x, batch_size, input_size);
	}
	virtual void sample_forward_prop(double* to_next) = 0;
	virtual void sample_leak_back_prop(double* from_next) = 0;

	virtual void batch_forward_prop(double** to_next) = 0;
	virtual void batch_back_prop(double** from_next) = 0;
	virtual void batch_leak_back_prop(double** from_next) = 0;

	virtual void grad_zero() = 0;
	virtual void feed_grad_itself(const double& learning_rate) = 0;
	virtual void alloc_batch_storage(const int& new_batch_size) = 0;

	virtual void save_unit(ofstream& fout) = 0;
	virtual void load_unit(ifstream& fin) = 0;
	Activation* activator;
	int input_size;
	int output_size;
	int batch_size;
	bool batch_memory_allocated;
	double* single_input_container;
	double* single_inner_deltaflow;
	double* single_pre_activated_container;
	double* single_port;

	double** batch_input_container;
	double** batch_inner_deltaflow;
	double** batch_pre_activated_container;
	double** batch_port;
	virtual ~unit() {}
};

class perceptrons : public unit {
public:
	double* bias;
	double* weight;
	double* gradients_bias;
	double* gradients_weight;
	int weight_len;
	bool use_bias;
	int bias_len;

	perceptrons() {}
	perceptrons(int input_sz, int output_sz, ACTIVATION f, bool use_Bias = true) {
		this->create(input_sz, output_sz, f, use_Bias);
	}

	void create(int input_sz, int output_sz, ACTIVATION f, bool use_Bias = true) {
		use_bias = use_Bias;
		unit_type = PERCEPTRONS;
		output_size = output_sz;
		input_size = input_sz;
		batch_size = NULL;
		bias_len = output_size;
		weight_len = input_size * output_size;

		Give_function(activator, f, output_size);

		weight = new double[weight_len];
		gradients_weight = new double[weight_len];

		km_1d::fill_random(weight, weight_len);
		
		if (activator->formula == RELU ) {
			km_1d::guassian_norm(weight, 0.0, 2.0 / (double)input_size , weight_len);
		}
		else {
			km_1d::guassian_norm(weight, 0.0, 1.0 / (double)input_size, weight_len);
		}

		if (use_bias == true) {
			bias = new double[bias_len];
			gradients_bias = new double[bias_len];
			km_1d::fill_zero(bias, bias_len);
		}

		single_inner_deltaflow = new double[output_size];
		single_pre_activated_container = new double[output_size];
		single_input_container = new double[input_size];
		single_port = new double[input_size];
		batch_memory_allocated = false;
	}
	void save_unit(ofstream& fout) {
		fout << unit_type << endl;
		fout << input_size << endl;
		fout << output_size << endl;
		fout << activator->formula << endl;
		fout << use_bias << endl;
		for (int i = 0; i < weight_len; i++) {
			fout << weight[i] << endl;
		}
		if (use_bias == true) {
			for (int i = 0; i < bias_len; i++) {
				fout << bias[i] << endl;
			}
		}
	}
	virtual void load_unit(ifstream& fin) {
		int input_sz;
		int output_sz;
		ACTIVATION f;
		bool ub;
		string buffer;
		getline(fin, buffer);
		input_sz = atoi(buffer.c_str());
		getline(fin, buffer);
		output_sz = atoi(buffer.c_str());
		getline(fin, buffer);
		f = (ACTIVATION)atoi(buffer.c_str());
		getline(fin, buffer);
		ub = (bool)atoi(buffer.c_str());
		create(input_sz, output_sz, f, ub);
		for (int i = 0; i < weight_len; i++) {
			getline(fin, buffer);
			weight[i] = atof(buffer.c_str());
		}
		if (use_bias == true) {
			for (int i = 0; i < bias_len; i++) {
				getline(fin, buffer);
				bias[i] = atof(buffer.c_str());
			}
		}
	}
	virtual void alloc_batch_storage(const int& new_batch_size) {
		if (batch_memory_allocated) {
			for (int i = 0; i < batch_size; i++) {
				delete[] batch_port[i];
				delete[] batch_pre_activated_container[i];
				delete[] batch_inner_deltaflow[i];
				delete[] batch_input_container[i];

			}
			delete[] batch_inner_deltaflow;
			delete[] batch_port;
			delete[] batch_input_container;
			delete[] batch_pre_activated_container;
		}
		
		batch_size = new_batch_size;
		activator->batch_size = batch_size;
		batch_port = new double*[batch_size];
		batch_pre_activated_container = new double*[batch_size];
		batch_inner_deltaflow = new double*[batch_size];
		batch_input_container = new double*[batch_size];

		for (int m = 0; m < batch_size; m++) {
			batch_port[m] = new double[input_size];
			batch_pre_activated_container[m] = new double[output_size];
			batch_inner_deltaflow[m] = new double[output_size];
			batch_input_container[m] = new double[input_size];
		}
		batch_memory_allocated = true;
	}

	virtual void sample_forward_prop(double* to_next) {
		if (use_bias == true) {
			for (int w = 0; w < output_size; w++) {
				single_pre_activated_container[w] = bias[w];
			}
		}
		else {
			km_1d::fill_zero(single_pre_activated_container, output_size);
		}

		for (int w = 0; w < output_size; w++) {
			for (int h = 0; h < input_size; h++) {
				single_pre_activated_container[w] += weight[w*input_size + h] * single_input_container[h];
			}
		}
		activator->function(to_next, single_pre_activated_container);
	}

	virtual void sample_leak_back_prop(double* from_next) {
		activator->function_prime(single_inner_deltaflow, single_pre_activated_container); //inner_df[m][w] <= d(activated)/d(pre_activated)
		for (int w = 0; w < output_size; w++) {
			single_inner_deltaflow[w] *= from_next[w]; //inner_df[m][w] <= d(Loss)/d(ztelda[m][w])
		}
		km_1d::fill_zero(single_port, input_size);
		for (int h = 0; h < input_size; h++) {

			for (int w = 0; w < output_size; w++) {
				single_port[h] += single_inner_deltaflow[w] * weight[w*input_size + h];
			}
		}
	}

	void input_X_weight(double** container) { /*pre_activated_container <- matmul(input, weight)*/
		if (use_bias == true) {
			for (int m = 0; m < batch_size; m++) {
				for (int w = 0; w < output_size; w++) {
					container[m][w] = bias[w];
				}
			}
		}
		else {

			km_2d::fill_zero(container, batch_size, output_size);
		}

		for (int m = 0; m < batch_size; m++) {
			for (int w = 0; w < output_size; w++) {
				for (int h = 0; h < input_size; h++) {
					container[m][w] += weight[w*input_size + h] * batch_input_container[m][h];
				}
			}
		}
	}

	virtual void batch_forward_prop(double** to_next) {
		input_X_weight(batch_pre_activated_container);
		activator->batch_function(to_next, batch_pre_activated_container);
	}

	void get_gradients() {
		if (use_bias == true) {
			for (int m = 0; m < batch_size; m++) {
				for (int w = 0; w < output_size; w++) {
					gradients_bias[w] += batch_inner_deltaflow[m][w];
				}
			}
		}
		for (int m = 0; m < batch_size; m++) {
			for (int w = 0; w < output_size; w++) {
				for (int h = 0; h < input_size; h++) {
					gradients_weight[w*input_size + h] += batch_inner_deltaflow[m][w] * batch_input_container[m][h];
				}
			}
		}
	}

	void recitfying_delta() {
		for (int m = 0; m < batch_size; m++) {
			for (int h = 0; h < input_size; h++) {
				batch_port[m][h] = 0.0;
				for (int w = 0; w < output_size; w++) {
					batch_port[m][h] += batch_inner_deltaflow[m][w] * weight[w*input_size + h];
				}
			}
		}
	}

	virtual void batch_back_prop(double** from_next) {
		activator->batch_function_prime(batch_inner_deltaflow, batch_pre_activated_container); //inner_df[m][w] <= d(activated)/d(pre_activated)
		for (int m = 0; m < batch_size; m++) {
			for (int w = 0; w < output_size; w++) {
				batch_inner_deltaflow[m][w] *= from_next[m][w]; //inner_df[m][w] <= d(Loss)/d(ztelda[m][w])
			}
		}
		get_gradients();// get d(Loss)/d(Weight)
		recitfying_delta(); //entrance_port[m][w] <= d(Loss)/d(input[m][w])
	}

	virtual void batch_leak_back_prop(double** from_next) {
		activator->batch_function_prime(batch_inner_deltaflow, batch_pre_activated_container); //inner_df[m][w] <= d(activated)/d(pre_activated)
		for (int m = 0; m < batch_size; m++) {
			for (int w = 0; w < output_size; w++) {
				batch_inner_deltaflow[m][w] *= from_next[m][w]; //inner_df[m][w] <= d(Loss)/d(ztelda[m][w])
			}
		}
		recitfying_delta(); //entrance_port[m][w] <= d(Loss)/d(input[m][w])
	}
	virtual void grad_zero() {
		if (use_bias == true) {
			km_1d::fill_zero(gradients_bias, bias_len);
		}
		km_1d::fill_zero(gradients_weight, weight_len);
	}

	virtual void feed_grad_itself(const double& learning_rate) {
		if (use_bias == true) {
			for (int i = 0; i < output_size; i++) {
				bias[i] -= gradients_bias[i] * learning_rate;
			}
		}
		for (int i = 0; i < weight_len; i++) {
			weight[i] -= gradients_weight[i] * learning_rate;
		}
	}
	virtual ~perceptrons() {
		if (batch_memory_allocated) {
			for (int i = 0; i < batch_size; i++) {
				delete[] batch_port[i];
				delete[] batch_pre_activated_container[i];
				delete[] batch_inner_deltaflow[i];
				delete[] batch_input_container[i];

			}
			delete[] batch_inner_deltaflow;
			delete[] batch_port;
			delete[] batch_input_container;
			delete[] batch_pre_activated_container;
		}
		delete activator;
		delete[] weight;
		delete[] gradients_weight;
		if (use_bias == true) {
			delete[] gradients_bias;
			delete[] bias;
		}
		delete[] single_input_container;
		delete[] single_pre_activated_container;
		delete[] single_inner_deltaflow;
		delete[] single_port;
	}
};