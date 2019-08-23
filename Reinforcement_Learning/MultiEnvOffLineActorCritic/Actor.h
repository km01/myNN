#pragma once
#include "unit.h"

class Actor : public bundle {
public:

	Actor() { content_allocated = false; }
	double* prob;
	double** prob_ptr;
	Actor(n_Layer number_of_layers) {
		n_layer = number_of_layers.n;
		layer = new unit * [n_layer];
		content_allocated = false;
	}

	virtual void publish() { //set input, output
		for (int i = 0; i < n_layer - 1; i++) {
			assert(layer[i]->output_size == layer[i + 1]->input_size);
		}
		input_size = layer[0]->input_size;
		output_size = layer[n_layer - 1]->output_size;
		_input_ = layer[0]->_input_;
		prob = km_1d::alloc(output_size);
		prob_ptr = new double* [1];
		prob_ptr[0] = prob;
		content_allocated = true;
	}

	~Actor() { if (content_allocated) { km_1d::free(prob); delete[] prob_ptr; } }

	int DeterministicPolicy(double* const& input) {
		_charge_(input);
		_calculate_(prob);
		return km::argmax(prob, output_size);
	}

	int StochasticPolicy(double* const& input) {
		_charge_(input);
		_calculate_(prob);
		return km::pick(prob, output_size);
	}

	double WithGradPolicy(double* const& state, const int& target_port) {
		km_1d::copy(in_port[0], state, input_size);
		forward(prob_ptr);
		return prob_ptr[0][target_port];
	}
};