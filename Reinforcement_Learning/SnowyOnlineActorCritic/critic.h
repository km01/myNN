#pragma once
#include "unit.h"

class Critic : public bundle {
public:

	double** batch_q; /* suppose batch size = 1 */
	double** dLoss;
	double* one_q;

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

		one_q = km_1d::alloc(output_size);
		dLoss = km_2d::alloc(1, output_size);
		batch_q = km_2d::alloc(1, output_size);
		content_allocated = true;
	}

	~Critic() {
		if (content_allocated) {
			km_1d::free(one_q);
			km_2d::free(batch_q, 1);
			km_2d::free(dLoss, 1);
		}
	}
	double NoGradEvaluate_V(double* const& state, double* const& prob) {
		_charge_(state);
		_calculate_(one_q);
		double v = 0.0;
		for (int i = 0; i < output_size; i++) {
			v += prob[i] * one_q[i];
		}
		return v;
	}

	double WithGradEvaluate_Q(double* const& state, const int& action) { //return Q(s,a)
		km_1d::copy(in_port[0], state, input_size);
		forward(batch_q);
		return batch_q[0][action];
	}


	void WithGradEvaluate_A(double& Q, double& V, double* const& state, double* const& prob, const int& action) { //return A(s,a)
		km_1d::copy(in_port[0], state, input_size);
		forward(batch_q);
		V = 0.0;
		Q = batch_q[0][action];
		for (int i = 0; i < output_size; i++) {
			V += prob[i] * batch_q[0][i];
		}
	}

	void TDErorrBackprop(const double& TDTarget, const int& port_id) {
		km_1d::fill_zero(dLoss[0], output_size);
		dLoss[0][port_id] = 2.0 * (batch_q[0][port_id] - TDTarget);
		backward(dLoss);
	}

};