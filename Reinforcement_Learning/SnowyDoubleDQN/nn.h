#pragma once
#include "unit.h"


class nn : public unit {
public:
	unit** layer;
	int n_layer;
	bool content_alloc;
	bool model_alloc;
	double** output_port;
	double* out_port;
	int one_argmax;

	int* argmax;
	nn() { model_alloc = false; content_alloc = false; }

	nn(N_Layer number_of_layers) {
		n_layer = number_of_layers.n;
		layer = new unit*[n_layer];
		content_alloc = false;
		model_alloc = false;
	}
	void save_nn(string file) {
		ofstream fout(file);
		save(fout);
		fout.close();
	}
	nn(string file) {
		ifstream fin(file);
		string buffer;
		getline(fin, buffer);
		if (buffer == "nn") {
			load(fin);
		}
		fin.close();
	}
	virtual void save(ofstream& fout) {
		fout << "nn" << endl;
		fout << n_layer << endl;
		for (int i = 0; i < n_layer; i++) {
			layer[i]->save(fout);
		}
	}
	virtual void load(ifstream& fin) {
		string buffer;
		getline(fin, buffer);
		n_layer = atoi(buffer.c_str());
		layer = new unit*[n_layer];
		content_alloc = false;
		model_alloc = false;
		for (int i = 0; i < n_layer; i++) {
			getline(fin, buffer);
			cout << "layer[" << i << "]  " << buffer << endl;
			if (buffer == "fully_connected") {
				layer[i] = new fully_connected();
			}
			else if (buffer == "ReLU") {
				layer[i] = new ReLU();
			}
			else if (buffer == "Tanh") {
				layer[i] = new Tanh();
			}
			else if (buffer == "kernel3D") {
				layer[i] = new kernel3D();
			}
			else if (buffer == "BatchNormalizer") {
				layer[i] = new BatchNormalizer();
			}
			layer[i]->load(fin);
		}
	}
	virtual unit* clone() {
		nn* _clone = new nn(N_Layer(n_layer));
		for (int i = 0; i < n_layer; i++) {
			_clone->layer[i] = layer[i]->clone();
		}
		return _clone;
	}

	virtual void publish() { //set input, output
		input_size = layer[0]->input_size;
		output_size = layer[n_layer - 1]->output_size;
		for (int i = 0; i < n_layer - 1; i++) {
			assert(layer[i]->output_size == layer[i + 1]->input_size);
		}
		in_port = layer[0]->in_port;
		out_port = km_1d::alloc(output_size);
		content_alloc = true;
	}

	virtual void UseMemory(const int& _using_size) {
		assert(max_batch_size >= _using_size);
		using_size = _using_size;
		for (int i = 0; i < n_layer; i++) {
			layer[i]->UseMemory(using_size);
		}
	}
	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->delegate(p_bag, g_bag, len_bag);
		}
	}


	virtual void alloc(int b_size) {
		if (model_alloc) {
			km_2d::free(output_port, max_batch_size);
			delete[] argmax;
		}

		max_batch_size = b_size;
		for (int i = 0; i < n_layer; i++) {
			layer[i]->alloc(max_batch_size);
		}
		using_size = max_batch_size;
		output_port = km_2d::alloc(max_batch_size, output_size);
		argmax = new int[max_batch_size];
		model_alloc = true;
		input_port = layer[0]->input_port;
		downstream = layer[0]->downstream;
	}

	virtual void forward(double** &output) {
		for (int i = 0; i < n_layer - 1; i++) {
			layer[i]->forward(layer[i + 1]->input_port);
		}
		layer[n_layer - 1]->forward(output);
	}
	void calculate(double* &next) {
		for (int i = 0; i < n_layer - 1; i++) {
			layer[i]->calculate(layer[i + 1]->in_port);
		}
		layer[n_layer - 1]->calculate(next);
	}
	int one_predict(double*& in) {
		charge(in);
		calculate(out_port);
		return km::argmax(out_port, output_size);
	}
	void predict(double** & input) {
		set_input(input);
		forward(output_port);
		km_2d::argmax(argmax, output_port, using_size, output_size);
	}

	virtual void backward(double** const& dLoss) {
		layer[n_layer - 1]->backward(dLoss);
		for (int i = n_layer - 2; i >= 0; i--) {
			layer[i]->backward(layer[i + 1]->downstream);
		}
	}

	~nn() {
		if (model_alloc) {
			km_2d::free(output_port, max_batch_size);
			delete[] argmax;
		}


		if (content_alloc) {
			for (int i = 0; i < n_layer; i++) {
				delete layer[i];
			}
			delete[] layer;
			km_1d::free(out_port);
		}
	}
};
