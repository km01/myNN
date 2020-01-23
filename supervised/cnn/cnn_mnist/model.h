#pragma once
#include "unit.h"
class N_Layer {
public:
	int n;
	N_Layer() {}
	N_Layer(int _n) {n = _n; }
};
class classifier : public unit {
public:
	unit** layer;
	int n_layer;
	bool content_alloc;
	bool model_alloc;
	double** output_port;
	double** softmax_output;
	int* argmax;
	classifier() { model_alloc = false; content_alloc = false; }
	classifier(N_Layer number_of_layers) {
		n_layer = number_of_layers.n;
		layer = new unit*[n_layer];
		content_alloc = false;
		model_alloc = false;
	}
	virtual void publish() {
		input_size = layer[0]->input_size;
		output_size = layer[n_layer - 1]->output_size;
		for (int i = 0; i < n_layer-1; i++) {
			assert(layer[i]->output_size == layer[i + 1]->input_size);
		}
		content_alloc = true;
	}

	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->delegate(p_bag, g_bag, len_bag);
		}
	}
	virtual void alloc(int b_size) {
		if (model_alloc) {
			km_2d::free(output_port, max_batch_size);
			km_2d::free(softmax_output, max_batch_size);
			delete[] argmax;
		}


		max_batch_size = b_size;
		for (int i = 0; i < n_layer; i++) {
			layer[i]->alloc(max_batch_size);
		}
		using_size = max_batch_size;

		output_port = km_2d::alloc(max_batch_size, output_size);
		softmax_output = km_2d::alloc(max_batch_size, output_size);
		argmax = new int[max_batch_size];
		model_alloc = true;

		input_port = layer[0]->input_port;
		downstream = layer[0]->downstream;
	}
	virtual void forward(double** &input) {
		set_input(input);
		for (int i = 0; i < n_layer - 1; i++) {
			layer[i]->forward(layer[i + 1]->input_port);
		}
		layer[n_layer - 1]->forward(output_port);
		softmax();
	}

	void softmax() {
		double max = 0.0;
		double base = 0.0;
		for (int m = 0; m < using_size; m++) {
			max = output_port[m][0];
			argmax[m] = 0;
			for (int i = 1; i < output_size; i++) {
				if (max < output_port[m][i]) {
					max = output_port[m][i];
					argmax[m] = i;
				}
			}
			base = 0.0;
			for (int i = 0; i < output_size; i++) {
				softmax_output[m][i] = exp(output_port[m][i] - max);
				base += softmax_output[m][i];
			}
			for (int i = 0; i < output_size; i++) {
				softmax_output[m][i] = softmax_output[m][i] / base;
			}
		}
	}
	double Bulit_in_CELoss(double** &dLoss_container, int* const& target) {
		double mean_loss = 0.0;
		for (int m = 0; m < using_size; m++) {
			for (int i = 0; i < output_size; i++) {
				if (target[m] == i) {
					dLoss_container[m][i] = softmax_output[m][i] - 1.0;
					mean_loss -= log(softmax_output[m][i]);
				}
				else {
					dLoss_container[m][i] = softmax_output[m][i];
				}
			}
		}
		return mean_loss / output_size;
	}
	virtual void backward(double** const& dLoss) {
		layer[n_layer - 1]->backward(dLoss);
		for (int i = n_layer - 2; i >= 0; i--) {
			layer[i]->backward(layer[i + 1]->downstream);
		}
	}

	~classifier() { 
		if (model_alloc) {
			km_2d::free(output_port, max_batch_size);
			km_2d::free(softmax_output, max_batch_size);
			delete[] argmax;
		}


		if (content_alloc) {
			for (int i = 0; i < n_layer; i++) {
				delete layer[i];
			}
			delete[] layer;
		}
	}
};
