#pragma once
#include "unit.h"

class Decoder : public bundle {
public:

	double** decode_container;
	Decoder() { content_allocated = false; }
	Decoder(n_Layer number_of_layers) {
		n_layer = number_of_layers.n;
		layer = new unit * [n_layer];

	}

	virtual void publish() { //set input, output
		for (int i = 0; i < n_layer - 1; i++) {
			assert(layer[i]->output_size == layer[i + 1]->input_size);
		}
		input_size = layer[0]->input_size;
		output_size = layer[n_layer - 1]->output_size;
		_input_ = layer[0]->_input_;
		decode_container = km_2d::alloc(1, output_size);
		content_allocated = true;
	}

	~Decoder() {
		if (content_allocated) { if (content_allocated) { km_2d::free(decode_container, 1); } }
	}

	void Decode(double* const& input, double*& output) {
		_charge_(input);
		_calculate_(output);
	}

	void WithGradDecode(double* const& state, double*& out_port) { //return V(s,a)
		km_1d::copy(in_port[0], state, input_size);
		forward(decode_container);
		km_1d::copy(out_port, decode_container[0], output_size);
	}

};