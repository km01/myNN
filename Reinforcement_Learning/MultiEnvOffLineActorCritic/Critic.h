#pragma once
#include "unit.h"

class Critic : public bundle {
public:
	double* v;
	double** v_ptr;
	Critic() { content_allocated = false; }
	Critic(n_Layer number_of_layers) {
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
		v = km_1d::alloc(1);
		v_ptr = new double* [1];	v_ptr[0] = v;
		content_allocated = true;
	}

	~Critic() { if (content_allocated) { km_1d::free(v); delete[] v_ptr; } }
	double Evaluate(double* const& input) {
		_charge_(input);
		_calculate_(v);
		return v[0];
	}

	double WithGradEvaluate(double* const& state) { //return V(s,a)
		km_1d::copy(in_port[0], state, input_size);
		forward(v_ptr);
		return v_ptr[0][0];
	}
};