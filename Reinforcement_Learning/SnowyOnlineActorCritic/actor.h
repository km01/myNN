#pragma once
#include "unit.h"

class Actor : public bundle {
public:

	double** batch_policy_prob; /* suppose batch size = 1 */
	double* policy_prob;
	double** PolicyGradient;
	Actor() { content_allocated = false; }
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
		policy_prob = km_1d::alloc(output_size);
		PolicyGradient = km_2d::alloc(1, output_size);
		batch_policy_prob = km_2d::alloc(1, output_size);
		content_allocated = true;
	}
	~Actor() {
		if (content_allocated) {
			km_1d::free(policy_prob);
			km_2d::free(PolicyGradient, 1);
			km_2d::free(batch_policy_prob, 1);
		}
	}
	int policy(double* const& state) {
		_charge_(state);
		_calculate_(policy_prob);
		return km::argmax(policy_prob, output_size);
	}
	
	int WithGradPolicy(double* const& state) {
		km_1d::copy(in_port[0], state, input_size);
		forward(batch_policy_prob);
		return km::pick(batch_policy_prob[0], output_size);
	}

	void OnLinePolicyGradientBackprop(const double& G, const int& target_port) {
		km_1d::fill_zero(PolicyGradient[0], output_size);
		PolicyGradient[0][target_port] = G / (-batch_policy_prob[0][target_port]);
		backward(PolicyGradient);
	}

};