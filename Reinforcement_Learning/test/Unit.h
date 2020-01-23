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
	virtual void charge(Darr const& input, const int& id) {
		for (int i = 0; i < n_in; i++) {
			in_port[id][i] = input[i];
		}
	}
	virtual void forward(Darr& rear, const int& id) = 0;
	virtual void backward(Darr const& rear, const int& id) = 0;
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

	virtual void backward(Darr const& output_grad, const int& id) {
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

class Conv2D : public Unit {
public:
	int in_h = 0, in_w = 0, in_c = 0;
	int out_h = 0, out_w = 0, out_c = 0;
	int mask_h = 0, mask_w = 0;
	int w_strd = 0, h_strd = 0;
	DarrList mask = nullptr;
	DarrList mask_grad = nullptr;
	Darr bias = nullptr;
	Darr b_grad = nullptr;
	int in_plain_size = 0; int out_plain_size = 0; int mask_plain_size = 0;

	Conv2D() {}

	bool valid_shape(const c_h_w& input_chw, const h_w& mask_hw, const c_h_w& output_chw, const h_w& stride_hw) {
		if (stride_hw.first * (output_chw.second.first - 1) + mask_hw.first == input_chw.second.first &&
			stride_hw.second * (output_chw.second.second - 1) + mask_hw.second == input_chw.second.second) {
			return true;
		}
		return false;
	}

	Conv2D(const c_h_w& input_chw, const h_w& mask_hw, const c_h_w& output_chw, const h_w& stride_hw) {
		if (!valid_shape(input_chw, mask_hw, output_chw, stride_hw)) {
			cout << " invalid Conv2D" << endl;
			assert(true);
		}
		in_c = input_chw.first; in_h = input_chw.second.first; in_w = input_chw.second.second;
		out_c = output_chw.first; out_h = output_chw.second.first; out_w = output_chw.second.second;
		setIOsize(in_c * in_h * in_w, out_c * out_h * out_w);
		mask_h = mask_hw.first; mask_w = mask_hw.second;
		h_strd = stride_hw.first; w_strd = stride_hw.second;
		in_plain_size = in_h * in_w;
		out_plain_size = out_h * out_w;
		mask_plain_size = mask_h * mask_w;
		mask = alloc(out_c, in_c * mask_plain_size);
		mask_grad = alloc(out_c, in_c * mask_plain_size);
		bias = alloc(out_c);
		b_grad = alloc(out_c);
		grad_port = alloc(n_in);
		setNormal(mask, out_c, in_c * mask_plain_size, 0.0, sqrt(2.0 / n_in));
		setZero(bias, out_c);
	}

	virtual void takeParams(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		for (int m = 0; m < out_c; m++) {
			p_bag.push_back(mask[m]);
			g_bag.push_back(mask_grad[m]);
			len_bag.push_back(in_c * mask_plain_size);
		}
		p_bag.push_back(bias);
		g_bag.push_back(b_grad);
		len_bag.push_back(out_c);
	}

	virtual void setCache(const int& cache_size) {
		free(in_port, this->cache_size);
		this->cache_size = cache_size;
		in_port = alloc(this->cache_size, this->n_in);
	}

	virtual void forward(Darr& next_input, const int& id) {

		for (int oc = 0; oc < out_c; oc++) {
			for (int k = 0; k < out_plain_size; k++) {
				next_input[(oc * out_plain_size) + k] = bias[oc];
			}
			for (int ic = 0; ic < in_c; ic++) {
				for (int ih = 0, oh = 0; oh < out_h; ih += h_strd, oh++) {
					for (int iw = 0, ow = 0; ow < out_w; iw += w_strd, ow++) {
						for (int mh = 0; mh < mask_h; mh++) {
							for (int mw = 0; mw < mask_w; mw++) {
								next_input[(oc * out_plain_size) + (oh * out_w) + (ow)]
									+= mask[oc][(ic * mask_plain_size) + (mh * mask_w) + (mw)]
									* in_port[id][(ic * (in_plain_size)) + (in_w * (ih + mh)) + (iw + mw)];
							}
						}
					}
				}
			}
		}
	}

	virtual void backward(Darr const& output_grad, const int& id) {
		for (int i = 0; i < n_in; i++) {
			grad_port[i] = 0.0;
		}

		for (int oc = 0; oc < out_c; oc++) {
			for (int k = 0; k < out_plain_size; k++) {
				b_grad[oc] += output_grad[(oc * out_plain_size) + k];
			}
			for (int ic = 0; ic < in_c; ic++) {
				for (int ih = 0, oh = 0; oh < out_h; ih += h_strd, oh++) {
					for (int iw = 0, ow = 0; ow < out_w; iw += w_strd, ow++) {
						for (int mh = 0; mh < mask_h; mh++) {
							for (int mw = 0; mw < mask_w; mw++) {
								mask_grad[oc][(ic * mask_plain_size) + (mh * mask_w) + (mw)]
									+= output_grad[(oc * out_plain_size) + (oh * out_w) + (ow)]
									* in_port[id][(ic * (in_plain_size)) + (in_w * (ih + mh)) + (iw + mw)];
								grad_port[(ic * (in_plain_size)) + (in_w * (ih + mh)) + (iw + mw)]
									+= output_grad[(oc * out_plain_size) + (oh * out_w) + (ow)]
									* mask[oc][(ic * mask_plain_size) + (mh * mask_w) + (mw)];
							}
						}
					}
				}
			}
		}
	}

	~Conv2D() {
		free(mask, out_c);
		free(mask_grad, out_c);
		free(bias);
		free(b_grad);
		free(in_port, cache_size);
		free(grad_port);
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

	virtual void backward(Darr const& output_grad, const int& id) {
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

	virtual void backward(Darr const& output_grad, const int& id) {
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

	virtual void backward(Darr const& output_grad, const int& id) {
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

	virtual void backward(Darr const& output_grad, const int& id) {
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

	virtual void backward(Darr const& output_grad, const int& id) {
		list[last_idx]->backward(output_grad, id);
		for (int i = last_idx - 1; i >= 0; i--) {
			list[i]->backward(list[i + 1]->grad_port, id);
		}
	}

	void push(Unit* unit) {
		if (last_idx == -1) {
			grad_port = unit->grad_port;
			n_in = unit->n_in;
		}
		else {
			if (list[last_idx]->n_out != unit->n_in) {
				cout << "invalid Sequential.push" << endl;
				assert(true);
			}
		}
		list.push_back(unit);
		sequential_len++;
		last_idx = sequential_len - 1;
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

class Conv2DTransposed : public Unit {
public:
	int in_h = 0, in_w = 0, in_c = 0;
	int out_h = 0, out_w = 0, out_c = 0;
	int mask_h = 0, mask_w = 0;
	int w_strd = 0, h_strd = 0;
	DarrList mask = nullptr;
	DarrList mask_grad = nullptr;
	Darr bias = nullptr;
	Darr b_grad = nullptr;
	int in_plain_size = 0; int out_plain_size = 0; int mask_plain_size = 0;
	Conv2DTransposed() {}

	bool valid_shape(const c_h_w& input_chw, const h_w& mask_hw, const c_h_w& output_chw, const h_w& stride_hw) {
		if (stride_hw.first * (input_chw.second.first - 1) + mask_hw.first == output_chw.second.first &&
			stride_hw.second * (input_chw.second.second - 1) + mask_hw.second == output_chw.second.second) {
			return true;
		}
		return false;
	}

	Conv2DTransposed(const c_h_w& input_chw, const h_w& mask_hw, const c_h_w& output_chw, const h_w& stride_hw) {
		if (!valid_shape(input_chw, mask_hw, output_chw, stride_hw)) {
			cout << " invalid Conv2DTransposed" << endl;
			assert(true);
		}
		in_c = input_chw.first; in_h = input_chw.second.first; in_w = input_chw.second.second;
		out_c = output_chw.first; out_h = output_chw.second.first; out_w = output_chw.second.second;
		setIOsize(in_c * in_h * in_w, out_c * out_h * out_w);
		mask_h = mask_hw.first; mask_w = mask_hw.second;
		h_strd = stride_hw.first; w_strd = stride_hw.second;
		in_plain_size = in_h * in_w;
		out_plain_size = out_h * out_w;
		mask_plain_size = mask_h * mask_w;
		mask = alloc(in_c, out_c * mask_plain_size);
		mask_grad = alloc(in_c, out_c * mask_plain_size);
		bias = alloc(in_c);
		b_grad = alloc(in_c);
		grad_port = alloc(n_in);
		setNormal(mask, in_c, out_c * mask_plain_size, 0.0, sqrt(2.0 / n_in));
		setZero(bias, in_c);
	}

	virtual void takeParams(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		for (int m = 0; m < in_c; m++) {
			p_bag.push_back(mask[m]);
			g_bag.push_back(mask_grad[m]);
			len_bag.push_back(out_c * mask_plain_size);
		}
		p_bag.push_back(bias);
		g_bag.push_back(b_grad);
		len_bag.push_back(in_c);
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
		for (int oc = 0; oc < out_c; oc++) {
			for (int ic = 0; ic < in_c; ic++) {
				for (int ih = 0, oh = 0; ih < in_h; ih++, oh += h_strd) {
					for (int iw = 0, ow = 0; iw < in_w; iw++, ow += w_strd) {
						for (int mh = 0; mh < mask_h; mh++) {
							for (int mw = 0; mw < mask_w; mw++) {
								next_input[(oc * out_plain_size) + ((oh + mh) * out_w) + (ow + mw)]
									+= mask[ic][(oc * mask_plain_size) + (mh * mask_w) + (mw)]
									* (in_port[id][(ic * (in_plain_size)) + (ih * in_w) + (iw)] + bias[ic]);
							}
						}
					}
				}
			}
		}
	}

	virtual void backward(Darr const& output_grad, const int& id) {
		for (int i = 0; i < n_in; i++) {
			grad_port[i] = 0.0;
		}
		for (int oc = 0; oc < out_c; oc++) {
			for (int ic = 0; ic < in_c; ic++) {
				for (int ih = 0, oh = 0; ih < in_h; ih++, oh += h_strd) {
					for (int iw = 0, ow = 0; iw < in_w; iw++, ow += w_strd) {
						for (int mh = 0; mh < mask_h; mh++) {
							for (int mw = 0; mw < mask_w; mw++) {
								grad_port[(ic * (in_plain_size)) + (ih * in_w) + (iw)] += mask[ic][(oc * mask_plain_size) + (mh * mask_w) + (mw)] *
									output_grad[(oc * out_plain_size) + ((oh + mh) * out_w) + (ow + mw)];
								mask_grad[ic][(oc * mask_plain_size) + (mh * mask_w) + (mw)] += output_grad[(oc * out_plain_size) + ((oh + mh) * out_w) + (ow + mw)]
									* (in_port[id][(ic * (in_plain_size)) + (ih * in_w) + (iw)] + bias[ic]);
								b_grad[ic] += mask[ic][(oc * mask_plain_size) + (mh * mask_w) + (mw)] *
									output_grad[(oc * out_plain_size) + ((oh + mh) * out_w) + (ow + mw)];
							}
						}
					}
				}
			}
		}
	}

	~Conv2DTransposed() {
		free(mask, in_c);
		free(mask_grad, in_c);
		free(bias);
		free(b_grad);
		free(in_port, cache_size);
		free(grad_port);
	}
};

class Softmax : public Unit {
public:
	Darr denominater = nullptr;
	DarrList exp_port = nullptr;
	Softmax() {}
	Softmax(const int& n_input) {
		setIOsize(n_input, n_input);
		grad_port = alloc(n_input);
	}

	virtual void forward(Darr& next_input, const int& id) {
		denominater[id] = 0.0;
		for (int i = 0; i < n_in; i++) {
			exp_port[id][i] = exp(in_port[id][i]);
			denominater[id] += exp_port[id][i];
		}
		for (int i = 0; i < n_in; i++) {
			next_input[i] = exp_port[id][i] / denominater[id];
		}
	}

	virtual void backward(Darr const& output_grad, const int& id) {
		double coLoss = 0.0;
		for (int i = 0; i < n_in; i++) {
			coLoss += output_grad[i] * exp_port[id][i];
		}
		coLoss = (-1.0 / (denominater[id] * denominater[id])) * coLoss;
		for (int i = 0; i < n_in; i++) {
			grad_port[i] = exp_port[id][i] * (coLoss + (output_grad[i] / denominater[id]));
		}
	}

	virtual void setCache(const int& cache_size) {
		free(in_port, this->cache_size);
		this->cache_size = cache_size;
		in_port = alloc(this->cache_size, this->n_in);
		denominater = alloc(this->cache_size);
		exp_port = alloc(this->cache_size, this->n_in);
	}

	~Softmax() {
		free(exp_port, this->cache_size);
		free(denominater);
		free(in_port, this->cache_size);
		free(grad_port);
	}
};

class softmax : public Unit {
public:
	Darr sfmx_scaler = nullptr;
	DarrList exp_port = nullptr;
	softmax() {}
	softmax(const int& n_input) {
		setIOsize(n_input, n_input);
		grad_port = alloc(n_input);
	}

	virtual void forward(Darr& next_input, const int& id) {
		sfmx_scaler[id] = 0.0;
		for (int i = 0; i < n_in; i++) {
			exp_port[id][i] = exp(in_port[id][i]);
			sfmx_scaler[id] += exp_port[id][i];
		}
		sfmx_scaler[id] = 1.0 / sfmx_scaler[id];
		for (int i = 0; i < n_in; i++) {
			next_input[i] = exp_port[id][i] * sfmx_scaler[id];
		}
	}

	virtual void backward(Darr const& output_grad, const int& id) {
		double sfmx_scaler_grad = 0.0;
		for (int i = 0; i < n_in; i++) {
			sfmx_scaler_grad += exp_port[id][i] * output_grad[i];
		}
		sfmx_scaler_grad = sfmx_scaler_grad * (-1) * sfmx_scaler[id] * sfmx_scaler[id];
		for (int i = 0; i < n_in; i++) {
			grad_port[i] = ((output_grad[i] * sfmx_scaler[id]) + (sfmx_scaler_grad)) * exp_port[id][i];
		}
	}

	virtual void setCache(const int& cache_size) {
		free(in_port, this->cache_size);
		this->cache_size = cache_size;
		in_port = alloc(this->cache_size, this->n_in);
		sfmx_scaler = alloc(this->cache_size);
		exp_port = alloc(this->cache_size, this->n_in);
	}

	~softmax() {
		free(exp_port, this->cache_size);
		free(sfmx_scaler);
		free(in_port, this->cache_size);
		free(grad_port);
	}
};