#pragma once
#include "unit.h"

class Decoder : public bundle {
public:

	double* NoGradResult;
	double** WithGradResult;
	double** dLoss;
	Decoder() { content_allocated = false; }
	Decoder(n_Layer number_of_layers) {
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

		NoGradResult = km_1d::alloc(output_size);
		dLoss = km_2d::alloc(1, output_size);
		WithGradResult = km_2d::alloc(1, output_size);
		content_allocated = true;
	}

	~Decoder() {
		if (content_allocated) {
			km_1d::free(NoGradResult);
			km_2d::free(WithGradResult, 1);
			km_2d::free(dLoss, 1);
		}
	}
	void NoGradDecode(double* const& state) {
		_charge_(state);
		_calculate_(NoGradResult);
	}

	void WithGradDecode(double* const& state) { //return V(s,a)
		km_1d::copy(in_port[0], state, input_size);
		forward(WithGradResult);
	}

	void GetGradBackprop(double** const& _src1, double** const& _src2) {
		km_2d::add(dLoss, _src1, _src2, 1, output_size);
		backward(dLoss);
	}
};