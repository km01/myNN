#pragma once
#include "Core.h"
using namespace km;

class Unit {
public:
	int n_in = 0;
	int n_out = 0;
	int cache_size = 0;
	Darr grad_port=nullptr;
	DarrList in_port=nullptr;
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
	virtual ~Unit(){}
};

class Dense : public Unit {
public:
	Darr weight = nullptr;
	Darr bias = nullptr;
	Darr w_grad = nullptr;
	Darr b_grad = nullptr;
	Dense(){}

	Dense(const int& n_input, const int& n_output) {
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
				grad_port[i] += weight[i* n_out + o] * output_grad[o];
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

class Shape {
public:
	union {
		struct { int c, h, w; };
		struct { int d1, d2, d3; };
	};
	Shape() {}
	Shape(const int& _c, const int& _h, const int& _w) :c(_c), h(_h), w(_w) {}
	void copy(const Shape& src) { c = src.c; h = src.h; w = src.w; }
	int len() { return c * h * w; }
};

bool shape_check(const Shape& in, const Shape& k_shape, const Shape& out, const int& h_stride, const int& w_stride) {
	if (in.c == k_shape.c) {
		if (h_stride * (out.h - 1) + k_shape.h == in.h) {
			if (w_stride * (out.w - 1) + k_shape.w == in.w) {
				return true;
			}
		}
	}
	return false;
}

class Conv2D : public Unit {
public:
	Shape in_dim; Shape out_dim; Shape mask_dim;
	DarrList mask = nullptr;
	DarrList mask_grad = nullptr;
	Darr bias = nullptr;
	Darr b_grad = nullptr;
	int w_strd = 0; int h_strd = 0;
	int in_plain_size = 0; int out_plain_size = 0; int mask_plain_size=0;
	Conv2D() {}
	Conv2D(const Shape& in_shape, const Shape& mask_shape, const Shape& out_shape, const int& h_strd, const int& w_strd) {
		assert(shape_check(in_shape, mask_shape, out_shape, h_strd, w_strd));
		in_dim.copy(in_shape);	mask_dim.copy(mask_shape);	out_dim.copy(out_shape);
		this->h_strd = h_strd;
		this->w_strd = w_strd;
		n_in = in_dim.c * in_dim.h * in_dim.w;
		n_out = out_dim.c * out_dim.h * out_dim.w;
		in_plain_size = in_dim.w * in_dim.h;
		out_plain_size = out_dim.w * out_dim.h;
		mask_plain_size = mask_dim.w * mask_dim.h;
		mask = alloc(out_dim.c, mask_dim.c * mask_dim.h * mask_dim.w);
		mask_grad = alloc(out_dim.c, mask_dim.c * mask_dim.h * mask_dim.w);
		bias = alloc(out_dim.c);
		b_grad = alloc(out_dim.c);
		grad_port = alloc(n_in);
		setNormal(mask, out_dim.c, mask_dim.c * mask_dim.h * mask_dim.w, 0.0, sqrt(2.0/n_in));
		setZero(bias, out_dim.c);
	}

	virtual void takeParams(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		for (int m = 0; m < out_dim.c; m++) {
			p_bag.push_back(mask[m]);
			g_bag.push_back(mask_grad[m]);
			len_bag.push_back(mask_dim.c * mask_dim.h * mask_dim.w);
		}
		p_bag.push_back(bias);
		g_bag.push_back(b_grad);
		len_bag.push_back(out_dim.c);
	}

	virtual void setCache(const int& cache_size) {
		free(in_port, this->cache_size);
		this->cache_size = cache_size;
		in_port = alloc(this->cache_size, this->n_in);
	}

	virtual void forward(Darr& next_input, const int& id) {
		for (int out_c = 0; out_c < out_dim.c; out_c++) {
			for (int k = 0; k < out_plain_size; k++) {
				next_input[(out_c * out_plain_size) + k] = bias[out_c];
			}
		}
		for (int out_c = 0; out_c < out_dim.c; out_c++) {
			for (int in_c = 0; in_c < in_dim.c; in_c++) {
				for (int in_h = 0, out_h = 0; out_h < out_dim.h; in_h += h_strd, out_h++) {
					for (int in_w = 0, out_w = 0; out_w < out_dim.w; in_w += w_strd, out_w++) {
						for (int kh = 0; kh < mask_dim.h; kh++) {
							for (int kw = 0; kw < mask_dim.w; kw++) {
								next_input[(out_c * out_plain_size) + (out_h * out_dim.w) + (out_w)] 
									+= mask[out_c][(in_c * mask_plain_size) + (kh * mask_dim.w) + (kw)]
									* in_port[id][(in_c * (in_plain_size)) + (in_dim.w * (in_h + kh)) + (in_w + kw)];
							}
						}
					}
				}
			}
		}
	}

	virtual void backward(Darr& output_grad, const int& id) {
		for (int i = 0; i < n_in; i++) {
			grad_port[i] = 0.0;
		}
		for (int out_c = 0; out_c < out_dim.c; out_c++) {
			for (int k = 0; k < out_plain_size; k++) {
				b_grad[out_c] += output_grad[(out_c * out_plain_size) + k];
			}
			for (int in_c = 0; in_c < in_dim.c; in_c++) {
				for (int in_h = 0, out_h = 0; out_h < out_dim.h; in_h += h_strd, out_h++) {
					for (int in_w = 0, out_w = 0; out_w < out_dim.w; in_w += w_strd, out_w++) {
						for (int kh = 0; kh < mask_dim.h; kh++) {
							for (int kw = 0; kw < mask_dim.w; kw++) {
								mask_grad[out_c][(in_c * mask_plain_size) + (kh * mask_dim.w) + (kw)] += output_grad[(out_c * out_plain_size) + (out_h * out_dim.w) + (out_w)]
									* in_port[id][(in_c * (in_plain_size)) + (in_dim.w * (in_h + kh)) + (in_w + kw)];
								grad_port[(in_c * (in_plain_size)) + (in_dim.w * (in_h + kh)) + (in_w + kw)] += output_grad[(out_c * out_plain_size) + (out_h * out_dim.w) + (out_w)] *
									mask[out_c][(in_c * mask_plain_size) + (kh * mask_dim.w) + (kw)];
							}
						}
					}
				}
			}
		}
	}
	~Conv2D() {
		free(mask);
		free(mask_grad, out_dim.c);
		free(bias);
		free(b_grad);
		free(in_port, cache_size);
		free(grad_port);
	}
};

class ReLU : public Unit {
public:
	ReLU(){}
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
			next_input[i] = 1.0/(1.0 + exp(-in_port[id][i]));
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
			grad_port[i] = output_grad[i] * (1.0 - x*x);
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
		for (int i = 0; i <last_idx; i++) {
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

class Conv2D_Transposed : public Unit {
public:
	Shape in_dim; Shape out_dim; Shape mask_dim;
	DarrList mask = nullptr;
	DarrList mask_grad = nullptr;
	Darr bias = nullptr;
	Darr b_grad = nullptr;
	int w_strd = 0; int h_strd = 0;
	int in_plain_size = 0; int out_plain_size = 0; int mask_plain_size = 0;
	Conv2D_Transposed() {}
	Conv2D_Transposed(const Shape& in_shape, const Shape& mask_shape, const Shape& out_shape, const int& h_strd, const int& w_strd) {
		in_dim.copy(in_shape);	mask_dim.copy(mask_shape);	out_dim.copy(out_shape);
		this->h_strd = h_strd;
		this->w_strd = w_strd;
		n_in = in_dim.c * in_dim.h * in_dim.w;
		n_out = out_dim.c * out_dim.h * out_dim.w;
		in_plain_size = in_dim.w * in_dim.h;
		out_plain_size = out_dim.w * out_dim.h;
		mask_plain_size = mask_dim.w * mask_dim.h;

		mask = alloc(in_dim.c, mask_dim.c * mask_dim.h * mask_dim.w);
		mask_grad = alloc(in_dim.c, mask_dim.c * mask_dim.h * mask_dim.w);
		grad_port = alloc(n_in);
		bias = alloc(in_dim.c);
		b_grad = alloc(in_dim.c);
		setNormal(mask, in_dim.c, mask_dim.c * mask_dim.h * mask_dim.w, 0.0, sqrt(2.0 / n_in));
		setZero(bias, in_dim.c);
	}

	virtual void takeParams(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		for (int m = 0; m < out_dim.c; m++) {
			p_bag.push_back(mask[m]);
			g_bag.push_back(mask_grad[m]);
			len_bag.push_back(mask_dim.c * mask_dim.h * mask_dim.w);
		}
		p_bag.push_back(bias);
		g_bag.push_back(b_grad);
		len_bag.push_back(in_dim.c);
	}

	virtual void setCache(const int& cache_size) {
		free(in_port, this->cache_size);
		this->cache_size = cache_size;
		in_port = alloc(this->cache_size, this->n_in);
	}

	virtual void forward(Darr& next_input, const int& id) {
		for (int i = 0; i < n_out; i++) {
			next_input[i] = 0.0;
		}

		for (int out_c = 0; out_c < out_dim.c; out_c++) {
			for (int in_c = 0; in_c < in_dim.c; in_c++) {
				for (int in_h = 0, out_h = 0; in_h < in_dim.h; in_h++, out_h += h_strd) {
					for (int in_w = 0, out_w = 0; in_w < in_dim.w; in_w++, out_w += w_strd) {

						for (int kh = 0; kh < mask_dim.h; kh++) {
							for (int kw = 0; kw < mask_dim.w; kw++) {
								next_input[(out_c * out_plain_size) + ((out_h + kh) * out_dim.w) + (out_w + kw)]
									+= mask[in_c][(out_c * mask_plain_size) + (kh * mask_dim.w) + (kw)]
									* (in_port[id][(in_c * (in_plain_size)) + ( in_h * in_dim.w ) + (in_w)] + bias[in_c]);
							}
						}
					}
				}
			}
		}
	}

	virtual void backward(Darr& output_grad, const int& id) {
		for (int i = 0; i < n_in; i++) {
			grad_port[i] = 0.0;
		}
		for (int out_c = 0; out_c < out_dim.c; out_c++) {
			for (int in_c = 0; in_c < in_dim.c; in_c++) {
				for (int in_h = 0, out_h = 0; in_h < in_dim.h; in_h++, out_h += h_strd) {
					for (int in_w = 0, out_w = 0; in_w < in_dim.w; in_w++, out_w += w_strd) {
						for (int kh = 0; kh < mask_dim.h; kh++) {
							for (int kw = 0; kw < mask_dim.w; kw++) {
								grad_port[(in_c * (in_plain_size)) + (in_h * in_dim.w) + (in_w)] += mask[in_c][(out_c * mask_plain_size) + (kh * mask_dim.w) + (kw)] *
									output_grad[(out_c * out_plain_size) + ((out_h + kh) * out_dim.w) + (out_w + kw)];

								mask_grad[in_c][(out_c * mask_plain_size) + (kh * mask_dim.w) + (kw)] += output_grad[(out_c * out_plain_size) + ((out_h + kh) * out_dim.w) + (out_w + kw)]
									* (in_port[id][(in_c * (in_plain_size)) + (in_h * in_dim.w) + (in_w)] + bias[in_c]);

								b_grad[in_c] += grad_port[(in_c * (in_plain_size)) + (in_h * in_dim.w) + (in_w)];
							}
						}
					}
				}
			}
		}
	}

	~Conv2D_Transposed() {
		free(mask);
		free(mask_grad, out_dim.c);
		free(bias);
		free(b_grad);
		free(in_port, cache_size);
		free(grad_port);
	}
};