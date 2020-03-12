#pragma once
#include "Core.h"

class Unit {
public:
	int n_in = 0;
	int n_out = 0;
	int n_cache = 0;
	darr::v2D in_cache = nullptr;
	darr::v2D in_grad_cache = nullptr;
	Unit() {}

	~Unit() {
	}

	virtual void setCache(const int& cache_size) { n_cache = cache_size; }

	virtual void charge(darr::v const& input, const int& id) {
		for (int i = 0; i < n_in; i++) { in_cache[id][i] = input[i]; }
	}

	virtual void forward(darr::v& output, const int& id) = 0;

	virtual void backward(darr::v& output_grad, const int& id) = 0;

	virtual void b_charge(darr::v2D const& inputs, const int& batch_size) {
		for (int n = 0; n < batch_size; n++) { charge(inputs[n], n); }
	}

	virtual void b_forward(darr::v2D const& outputs, const int& batch_size) {
		for (int n = 0; n < batch_size; n++) { forward(outputs[n], n); }
	}

	virtual void b_bacward(darr::v2D const& output_grads, const int& batch_size) {
		for (int n = 0; n < batch_size; n++) { backward(output_grads[n], n); }
	}

	virtual void takeParams(vector<darr::v>& param_bag, vector<darr::v>& grad_bag, vector<int>& len_bag) {}
};

class Dense : public Unit {
public:

	darr::v weight = nullptr;
	darr::v w_grad = nullptr;
	darr::v bias = nullptr;
	darr::v b_grad = nullptr;
	Dense() {}
	Dense(const int& input_sz, const int& output_sz) { create(input_sz, output_sz); }

	virtual void setCache(const int& new_cache_size) {
		darr::resize(in_cache, this->n_cache, new_cache_size, n_in);
		darr::resize(in_grad_cache, this->n_cache, new_cache_size, n_in);
		this->n_cache = new_cache_size;
	}

	void create(const int& input_sz, const int& output_sz) {
		n_in = input_sz;
		n_out = output_sz;
		weight = darr::alloc(n_in * n_out);
		w_grad = darr::alloc(n_in * n_out);
		bias = darr::alloc(n_out);
		b_grad = darr::alloc(n_out);
		//	km_1d::fill_guassian_noise(weight, 0.0, 2.0 / (double)n_in, n_in * n_out);

		km::normal_sampling(weight, n_in * n_out, 0.0, 2.0 / double(n_in));
		km::setZero(bias, n_out);
	}

	void forward(darr::v& output, const int& id) {
		for (int w = 0; w < n_out; w++) {
			output[w] = bias[w];
		}
		for (int h = 0; h < n_in; h++) {
			for (int w = 0; w < n_out; w++) {
				output[w] += weight[h * n_out + w] * in_cache[id][h];
			}
		}
	}

	void backward(darr::v& output_grad, const int& id) {
		for (int w = 0; w < n_out; w++) {
			b_grad[w] += output_grad[w];
		}
		for (int h = 0; h < n_in; h++) {
			in_grad_cache[id][h] = 0.0;
			for (int w = 0; w < n_out; w++) {
				w_grad[h * n_out + w] += output_grad[w] * in_cache[id][h];
				in_grad_cache[id][h] += output_grad[w] * weight[h * n_out + w];
			}
		}
	}

	virtual void takeParams(vector<darr::v>& param_bag, vector<darr::v>& grad_bag, vector<int>& len_bag) {
		param_bag.push_back(weight);
		grad_bag.push_back(w_grad);
		len_bag.push_back(n_in * n_out);
		param_bag.push_back(bias);
		grad_bag.push_back(b_grad);
		len_bag.push_back(n_out);
	}

	~Dense() {
		darr::free(weight);
		darr::free(w_grad);
		darr::free(bias);
		darr::free(b_grad);
	}
};


class ReLU : public Unit {
public:

	ReLU() {}
	ReLU(const int& size) { create(size); }

	void create(const int& size) {
		n_in = size;
		n_out = size;
	}

	virtual void setCache(const int& new_cache_size) {
		darr::resize(in_cache, this->n_cache, new_cache_size, n_in);
		darr::resize(in_grad_cache, this->n_cache, new_cache_size, n_in);
		this->n_cache = new_cache_size;
	}

	~ReLU() {

	}

	void forward(darr::v& output, const int& id) {
		for (int i = 0; i < n_in; i++) {
			if (in_cache[id][i] > 0) {
				output[i] = in_cache[id][i];
			}
			else {
				output[i] = 0.0;
			}
		}
	}

	void backward(darr::v& output_grad, const int& id) {
		for (int i = 0; i < n_in; i++) {
			if (in_cache[id][i] > 0) {
				in_grad_cache[id][i] = output_grad[i];
			}
			else {
				in_grad_cache[id][i] = 0.0;
			}
		}
	}
};

class Softmax : public Unit {
public:

	Softmax() {}
	Softmax(const int& size) { create(size); }

	void create(const int& size) {
		n_in = size;
		n_out = size;
	}

	virtual void setCache(const int& new_cache_size) {
		darr::resize(in_cache, this->n_cache, new_cache_size, n_in);
		darr::resize(in_grad_cache, this->n_cache, new_cache_size, n_in);
		this->n_cache = new_cache_size;
	}

	~Softmax() {

	}

	void forward(darr::v& output, const int& id) {
		double denom = 0.0;
		for (int i = 0; i < n_in; i++) {
			output[i] = exp(in_cache[id][i]);
			denom += output[i];
		}
		for (int i = 0; i < n_in; i++) {
			output[i] /= denom;
		}
	}

	void backward(darr::v& output_grad, const int& id) {
		double regularizer = 0.0;
		double regularizer_grad = 0.0;
		for (int i = 0; i < n_in; i++) {
			regularizer += exp(in_cache[id][i]);
			regularizer_grad += exp(in_cache[id][i]) * output_grad[i];
		}
		regularizer = 1.0 / regularizer;
		for (int i = 0; i < n_in; i++) {
			in_grad_cache[id][i] = (output_grad[i] - regularizer * regularizer_grad) * regularizer * exp(in_cache[id][i]);
		}
	}
};

class Sigmoid : public Unit {
public:

	Sigmoid() {}
	Sigmoid(const int& size) { create(size); }

	void create(const int& size) {
		n_in = size;
		n_out = size;
	}

	virtual void setCache(const int& new_cache_size) {
		darr::resize(in_cache, this->n_cache, new_cache_size, n_in);
		darr::resize(in_grad_cache, this->n_cache, new_cache_size, n_in);
		this->n_cache = new_cache_size;
	}

	~Sigmoid() {

	}

	void forward(darr::v& output, const int& id) {
		for (int i = 0; i < n_in; i++) {
			output[i] = 1.0 / (1.0 + exp(-in_cache[id][i]));
		}
	}

	void backward(darr::v& output_grad, const int& id) {
		double sig = 0.0;
		for (int i = 0; i < n_in; i++) {
			sig = 1.0 / (1.0 + exp(-in_cache[id][i]));
			in_grad_cache[id][i] = output_grad[i] * sig * (1.0 - sig);
		}
	}
};

class LeakyReLU : public Unit {
public:

	double slope = 0.0;
	LeakyReLU() {}
	LeakyReLU(const int& size, const double& slope) { create(size, slope); }

	void create(const int& size, const double& slope) {
		n_in = size;
		n_out = size;
		this->slope = slope;
	}

	virtual void setCache(const int& new_cache_size) {
		darr::resize(in_cache, this->n_cache, new_cache_size, n_in);
		darr::resize(in_grad_cache, this->n_cache, new_cache_size, n_in);
		this->n_cache = new_cache_size;
	}

	~LeakyReLU() {

	}

	void forward(darr::v& output, const int& id) {
		for (int i = 0; i < n_in; i++) {
			if (in_cache[id][i] > 0) {
				output[i] = in_cache[id][i];
			}
			else {
				output[i] = in_cache[id][i] * slope;
			}
		}
	}

	void backward(darr::v& output_grad, const int& id) {
		for (int i = 0; i < n_in; i++) {
			if (in_cache[id][i] > 0) {
				in_grad_cache[id][i] = output_grad[i];
			}
			else {
				in_grad_cache[id][i] = output_grad[i] * slope;
			}
		}
	}
};


class Sequential : public Unit {
public:

	vector<Unit*> units;
	int n_units = 0;
	Sequential() { units.clear(); }
	~Sequential() {}
	void append(Unit* new_unit) {
		if (n_units == 0) {
			n_in = new_unit->n_in;
		}
		else {
			if (units[n_units - 1]->n_out != new_unit->n_in) {
				cout << "invalid Sequential.push" << endl;
				assert(false);
			}
		}
		units.push_back(new_unit);
		n_out = new_unit->n_out;
		n_units += 1;
	}

	void forward(darr::v& output, const int& id) {
		for (int i = 0; i < n_units - 1; i++) {
			units[i]->forward(units[i + 1]->in_cache[id], id);
		}
		units[n_units - 1]->forward(output, id);
	}

	void backward(darr::v& output_grad, const int& id) {
		units[n_units - 1]->backward(output_grad, id);
		for (int i = n_units - 2; i >= 0; i--) {
			units[i]->backward(units[i + 1]->in_grad_cache[id], id);
		}
	}

	virtual void setCache(const int& cache_size) {
		for (int i = 0; i < n_units; i++) {
			units[i]->setCache(cache_size);
		}
		this->n_cache = cache_size;
		this->in_cache = units[0]->in_cache;
		this->in_grad_cache = units[0]->in_grad_cache;
	}

	virtual void takeParams(vector<darr::v>& param_bag, vector<darr::v>& grad_bag, vector<int>& len_bag) {
		for (int i = 0; i < n_units; i++) {
			units[i]->takeParams(param_bag, grad_bag, len_bag);
		}
	}
};
