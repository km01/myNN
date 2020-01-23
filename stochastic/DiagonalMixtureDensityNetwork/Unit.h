#pragma once
#include "Core.h"
using namespace km;

class Unit {
public:
	int n_in = 0;
	int n_out = 0;
	int cache_size = 0;
	Darr grad_port = nullptr;
	DarrList in_port = nullptr;
	void setIOsize(const int& n_input, const int& n_output) { n_in = n_input; n_out = n_output; }
	virtual void charge(Darr& input, const int& id) {
		for (int i = 0; i < n_in; i++) {
			in_port[id][i] = input[i];
		}
	}
	virtual void forward(Darr& rear, const int& id) = 0;
	virtual void backward(Darr& rear, const int& id) = 0;
	virtual void setCache(const int& cache_size) {}
	virtual void takeParams(vector<Darr>& param_bag, vector<Darr>& grad_bag, vector<int>& len_bag) {}
	virtual ~Unit() {}
};

class Dense : public Unit {
public:
	Darr weight = nullptr;
	Darr bias = nullptr;
	Darr w_grad = nullptr;
	Darr b_grad = nullptr;
	Dense() {}

	Dense(const int& n_input, const int& n_output) {
		create(n_input, n_output);
	}

	void create(const int& n_input, const int& n_output) {
		setIOsize(n_input, n_output);
		weight = alloc(n_input * n_output);
		w_grad = alloc(n_input * n_output);
		bias = alloc(n_output);
		b_grad = alloc(n_output);
		grad_port = alloc(n_input);
		setNormal(weight, n_in * n_out, 0.0, 2.0 / double(n_in + n_out));
		setZero(bias, n_out);
	}

	virtual void forward(Darr& next_input, const int& id) {
		for (int o = 0; o < n_out; o++) {
			next_input[o] = bias[o];
		}
		for (int i = 0; i < n_in; i++) {
			for (int o = 0; o < n_out; o++) {
				next_input[o] += weight[i * n_out + o] * in_port[id][i];
			}
		}
	}

	virtual void backward(Darr& output_grad, const int& id) {
		for (int o = 0; o < n_out; o++) {
			b_grad[o] += output_grad[o];
		}
		for (int i = 0; i < n_in; i++) {
			grad_port[i] = 0.0;
			for (int o = 0; o < n_out; o++) {
				w_grad[i * n_out + o] += output_grad[o] * in_port[id][i];
				grad_port[i] += weight[i * n_out + o] * output_grad[o];
			}
		}
	}

	virtual void setCache(const int& cache_size) {
		free(in_port, this->cache_size);
		this->cache_size = cache_size;
		in_port = alloc(this->cache_size, this->n_in);
	}

	virtual void takeParams(vector<Darr>& param_bag, vector<Darr>& grad_bag, vector<int>& len_bag) {
		param_bag.push_back(weight);
		grad_bag.push_back(w_grad);
		len_bag.push_back(n_in * n_out);
		param_bag.push_back(bias);
		grad_bag.push_back(b_grad);
		len_bag.push_back(n_out);
	}

	~Dense() {
		free(weight);
		free(w_grad);
		free(bias);
		free(b_grad);
		free(grad_port);
		free(in_port, this->cache_size);
	}
};

class ReLU : public Unit {
public:
	ReLU() {}
	ReLU(const int& n_input) {
		setIOsize(n_input, n_input);
		grad_port = alloc(n_input);
	}
	virtual void forward(Darr& next_input, const int& id) {
		for (int i = 0; i < n_in; i++) {
			if (in_port[id][i] < 0) {
				next_input[i] = 0.0;
			}
			else {
				next_input[i] = in_port[id][i];
			}
		}
	}

	virtual void backward(Darr& output_grad, const int& id) {
		for (int i = 0; i < n_in; i++) {
			if (in_port[id][i] < 0) {
				grad_port[i] = 0.0;
			}
			else {
				grad_port[i] = output_grad[i];
			}
		}
	}

	virtual void setCache(const int& cache_size) {
		free(in_port, this->cache_size);
		this->cache_size = cache_size;
		in_port = alloc(this->cache_size, this->n_in);
	}

	~ReLU() {
		free(in_port, this->cache_size);
		free(grad_port);
	}
};


class LeakyReLU : public Unit {
public:
	double slope = 0.2;
	LeakyReLU() {}
	LeakyReLU(const int& n_input) {
		setIOsize(n_input, n_input);
		grad_port = alloc(n_input);
	}

	LeakyReLU* setSlope(const double& slope) {
		this->slope = slope;
		return this;
	}

	virtual void forward(Darr& next_input, const int& id) {
		for (int i = 0; i < n_in; i++) {
			if (in_port[id][i] < 0) {
				next_input[i] = in_port[id][i] * slope;
			}
			else {
				next_input[i] = in_port[id][i];
			}
		}
	}

	virtual void backward(Darr& output_grad, const int& id) {
		for (int i = 0; i < n_in; i++) {
			if (in_port[id][i] < 0) {
				grad_port[i] = slope * output_grad[i];
			}
			else {
				grad_port[i] = output_grad[i];
			}
		}
	}

	virtual void setCache(const int& cache_size) {
		free(in_port, this->cache_size);
		this->cache_size = cache_size;
		in_port = alloc(this->cache_size, this->n_in);
	}

	~LeakyReLU() {
		free(in_port, this->cache_size);
		free(grad_port);
	}
};


class Sigmoid : public Unit {
public:
	Sigmoid() {}
	Sigmoid(const int& n_input) {
		setIOsize(n_input, n_input);
		grad_port = alloc(n_input);
	}

	virtual void forward(Darr& next_input, const int& id) {
		for (int i = 0; i < n_in; i++) {
			next_input[i] = 1.0 / (1.0 + exp(-in_port[id][i]));
		}
	}

	virtual void backward(Darr& output_grad, const int& id) {
		double sig = 0.0;
		for (int i = 0; i < n_in; i++) {
			sig = 1.0 / (1.0 + exp(-in_port[id][i]));
			grad_port[i] = output_grad[i] * sig * (1.0 - sig);
		}
	}

	virtual void setCache(const int& cache_size) {
		free(in_port, this->cache_size);
		this->cache_size = cache_size;
		in_port = alloc(this->cache_size, this->n_in);
	}

	~Sigmoid() {
		free(in_port, this->cache_size);
		free(grad_port);
	}
};

class Tanh : public Unit {
public:
	Tanh() {}
	Tanh(const int& n_input) {
		setIOsize(n_input, n_input);
		grad_port = alloc(n_input);
	}

	virtual void forward(Darr& next_input, const int& id) {
		for (int i = 0; i < n_in; i++) {
			next_input[i] = 2.0 / (1.0 + exp(-2.0 * in_port[id][i])) - 1.0;
		}
	}

	virtual void backward(Darr& output_grad, const int& id) {
		double x = 0.0;
		for (int i = 0; i < n_in; i++) {
			x = 2.0 / (1.0 + exp(-2.0 * in_port[id][i])) - 1.0;
			grad_port[i] = output_grad[i] * (1.0 - x * x);
		}
	}

	virtual void setCache(const int& cache_size) {
		free(in_port, this->cache_size);
		this->cache_size = cache_size;
		in_port = alloc(this->cache_size, this->n_in);
	}

	~Tanh() {
		free(in_port, this->cache_size);
		free(grad_port);
	}
};

class Sequential : public Unit {
public:

	vector<Unit*> list;
	int sequential_len = 0;
	int last_idx = -1;
	Sequential() { list.clear(); }

	virtual void forward(Darr& next_input, const int& id) {
		for (int i = 0; i < last_idx; i++) {
			list[i]->forward(list[i + 1]->in_port[id], id);
		}
		list[last_idx]->forward(next_input, id);
	}

	virtual void backward(Darr& output_grad, const int& id) {
		list[last_idx]->backward(output_grad, id);
		for (int i = last_idx - 1; i >= 0; i--) {
			list[i]->backward(list[i + 1]->grad_port, id);
		}
	}

	void push(Unit* unit) {
		list.push_back(unit);
		grad_port = list[0]->grad_port;
		sequential_len++;
		last_idx = sequential_len - 1;
		n_in = list[0]->n_in;
		n_out = list[last_idx]->n_out;
	}

	virtual void setCache(const int& cache_size) {
		for (int i = 0; i < sequential_len; i++) {
			list[i]->setCache(cache_size);
		}
		in_port = list[0]->in_port;
	}

	virtual void takeParams(vector<Darr>& param_bag, vector<Darr>& grad_bag, vector<int>& len_bag) {
		for (int i = 0; i < sequential_len; i++) {
			list[i]->takeParams(param_bag, grad_bag, len_bag);
		}
	}

	~Sequential() {
		for (int i = 0; i < sequential_len; i++) {
			delete list[i];
		}
	}
};