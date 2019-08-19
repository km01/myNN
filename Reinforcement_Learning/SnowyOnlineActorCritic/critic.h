#pragma once
#include "unit.h"

class Critic : public bundle {
public:

	double** batch_v; /* suppose batch size = 1 */
	double** dLoss;
	double* one_v;

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

		one_v = km_1d::alloc(output_size);
		dLoss = km_2d::alloc(1, output_size);
		batch_v = km_2d::alloc(1, output_size);
		content_allocated = true;
	}

	~Critic() {
		if (content_allocated) {
			km_1d::free(one_v);
			km_2d::free(batch_v, 1);
			km_2d::free(dLoss, 1);
		}
	}
	double NoGradEvaluate_V(double* const& state) {
		_charge_(state);
		_calculate_(one_v);
		return one_v[0];
	}

	double WithGradEvaluate_V(double* const& state) { //return V(s,a)
		km_1d::copy(in_port[0], state, input_size);
		forward(batch_v);
		return batch_v[0][0];
	}

	void TDErorrBackprop(const double& TDTarget) {
		km_1d::fill_zero(dLoss[0], output_size);
		dLoss[0][0] = 2.0 * (batch_v[0][0] - TDTarget);
		backward(dLoss);
	}
};