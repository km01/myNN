#pragma once
#include "mynn_core.h"
#include "unit.h"
class number_of_units {
public:
	int nb_units;
	number_of_units(int Number_of_unit) {
		nb_units = Number_of_unit;
	}
};
class multi_layer_network {
public:
	unit** layer;
	int end_layer;
	int n_layer;
	int batch_size;
	int input_size;
	int output_size;
	double** batch_output;
	double** batch_dLoss;
	int* batch_argmax;
	int sample_argmax;
	double* sample_output;
	double* sample_dLoss;

	double cost;
	double accuracy;
	double dLossinput;
	bool allocated_batch_memory;
	bool allocated_sample_memory;
	multi_layer_network(){}
	multi_layer_network(int n) {
		allocated_batch_memory = false;
		allocated_sample_memory = false;

		n_layer = n;
		batch_size = -1;
		end_layer = n_layer - 1;
		layer = new unit*[n_layer];
	}
	
	multi_layer_network(number_of_units num) : multi_layer_network(num.nb_units) {}
	
	void zero_grad() {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->zero_grad();
		}
	}
	void alloc_sample_memory() {
		if (!allocated_sample_memory) {
			input_size = layer[0]->input_size;
			output_size = layer[end_layer]->output_size;
			sample_output = new double[output_size];
			allocated_sample_memory = true;
		}
	}
	void alloc_batch_memory(const int& new_batch_size) {
		if (allocated_batch_memory) {
			km_2d::free(batch_output, batch_size);
		}
		batch_size = new_batch_size;
		for (int i = 0; i < n_layer; i++) {
			layer[i]->alloc_batch_storage(batch_size);
		}
		input_size = layer[0]->input_size;
		output_size = layer[end_layer]->output_size;
		batch_output = km_2d::alloc(batch_size, layer[end_layer]->output_size);
		batch_argmax = new int[batch_size];
		allocated_batch_memory = true;
	}

	void feed_grad_itself(const double& learning_rate) {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->feed_grad_itself(learning_rate);
		}
	}

	void batch_feed_forward(double** const& mini_x) {
		layer[0]->batch_load_inputs(mini_x);
		for (int i = 0; i < n_layer-1; i++) {
			layer[i]->batch_forward_prop(layer[i + 1]->batch_input_container);
		}
		layer[end_layer]->batch_forward_prop(batch_output);
		km_2d::argmax(batch_argmax, batch_output, batch_size, output_size);
	}

	void batch_feed_backward(double** const& dLoss) {
		layer[end_layer]->batch_back_prop(dLoss);
		for (int i = end_layer - 1; i >= 0; i--) {
			layer[i]->batch_back_prop(layer[i + 1]->batch_port);
		}
	}

	void single_feed_forward(double* const& single_input) {
		layer[0]->single_load_input(single_input);
		for (int i = 0; i < end_layer - 1; i++) {
			layer[i]->sample_forward_prop(layer[i + 1]->single_input_container);
		}
		layer[end_layer]->sample_forward_prop(sample_output);
		km_1d::argmax(sample_argmax,sample_output, output_size);
	}
	~multi_layer_network() {
		if (allocated_batch_memory) {
			km_2d::free(batch_output, batch_size);
		}
		if (allocated_sample_memory) {
			km_1d::free(sample_output);
		}
		for (int i = 0; i < end_layer; i++) {
			delete layer[i];
		}
		delete[] layer;
	}
};
