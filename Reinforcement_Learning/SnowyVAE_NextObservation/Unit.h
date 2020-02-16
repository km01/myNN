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
		//darr::free(in_cache, n_cache);
		//darr::free(in_grad_cache, n_cache);
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

class Conv2D : public Unit {
public:
	int in_h = 0, in_w = 0, in_c = 0;
	int out_h = 0, out_w = 0, out_c = 0;
	int mask_h = 0, mask_w = 0;
	int w_strd = 0, h_strd = 0;
	darr::v2D mask = nullptr;
	darr::v2D mask_grad = nullptr;
	darr::v bias = nullptr;
	darr::v b_grad = nullptr;
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
			assert(false);
		}
		in_c = input_chw.first; in_h = input_chw.second.first; in_w = input_chw.second.second;
		out_c = output_chw.first; out_h = output_chw.second.first; out_w = output_chw.second.second;

		n_in = in_c * in_h * in_w;
		n_out = out_c * out_h * out_w;

		mask_h = mask_hw.first; mask_w = mask_hw.second;
		h_strd = stride_hw.first; w_strd = stride_hw.second;
		in_plain_size = in_h * in_w;
		out_plain_size = out_h * out_w;
		mask_plain_size = mask_h * mask_w;
		mask = darr::alloc(out_c, in_c * mask_plain_size);
		mask_grad = darr::alloc(out_c, in_c * mask_plain_size);
		bias = darr::alloc(out_c);
		b_grad = darr::alloc(out_c);
		for (int c = 0; c < out_c; c++) {
			//km_1d::fill_guassian_noise(mask[c], 0.0, 2.0 / n_in, in_c * mask_plain_size);

			km::normal_sampling(mask[c], in_c * mask_plain_size, 0.0, 2.0 / n_in);
		}
		km::setZero(bias, out_c);
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

	virtual void setCache(const int& new_cache_size) {
		darr::resize(in_cache, this->n_cache, new_cache_size, n_in);
		darr::resize(in_grad_cache, this->n_cache, new_cache_size, n_in);
		this->n_cache = new_cache_size;
	}

	void forward(darr::v& output, const int& id) {
		for (int oc = 0; oc < out_c; oc++) {
			for (int k = 0; k < out_plain_size; k++) {
				output[(oc * out_plain_size) + k] = bias[oc];
			}
			for (int ic = 0; ic < in_c; ic++) {
				for (int ih = 0, oh = 0; oh < out_h; ih += h_strd, oh++) {
					for (int iw = 0, ow = 0; ow < out_w; iw += w_strd, ow++) {
						for (int mh = 0; mh < mask_h; mh++) {
							for (int mw = 0; mw < mask_w; mw++) {
								output[(oc * out_plain_size) + (oh * out_w) + (ow)]
									+= mask[oc][(ic * mask_plain_size) + (mh * mask_w) + (mw)]
									* in_cache[id][(ic * (in_plain_size)) + (in_w * (ih + mh)) + (iw + mw)];
							}
						}
					}
				}
			}
		}
	}

	void backward(darr::v& output_grad, const int& id) {
		for (int i = 0; i < n_in; i++) {
			in_grad_cache[id][i] = 0.0;
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
									* in_cache[id][(ic * (in_plain_size)) + (in_w * (ih + mh)) + (iw + mw)];
								in_grad_cache[id][(ic * (in_plain_size)) + (in_w * (ih + mh)) + (iw + mw)]
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
		darr::free(mask, out_c);
		darr::free(mask_grad, out_c);
		darr::free(bias);
		darr::free(b_grad);
	}
};

class Conv2DTransposed : public Unit {
public:
	int in_h = 0, in_w = 0, in_c = 0;
	int out_h = 0, out_w = 0, out_c = 0;
	int mask_h = 0, mask_w = 0;
	int w_strd = 0, h_strd = 0;
	darr::v2D mask = nullptr;
	darr::v2D mask_grad = nullptr;

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
			cout << " invalid Conv2D" << endl;
			assert(false);
		}
		in_c = input_chw.first; in_h = input_chw.second.first; in_w = input_chw.second.second;
		out_c = output_chw.first; out_h = output_chw.second.first; out_w = output_chw.second.second;
		n_in = in_c * in_h * in_w;
		n_out = out_c * out_h * out_w;
		mask_h = mask_hw.first; mask_w = mask_hw.second;
		h_strd = stride_hw.first; w_strd = stride_hw.second;
		in_plain_size = in_h * in_w;
		out_plain_size = out_h * out_w;
		mask_plain_size = mask_h * mask_w;
		mask = darr::alloc(in_c, out_c * mask_plain_size);
		mask_grad = darr::alloc(in_c, out_c * mask_plain_size);

		for (int c = 0; c < in_c; c++) {
			km::normal_sampling(mask[c], out_c * mask_plain_size, 0.0, 2.0 / n_in);
		}
	}

	virtual void takeParams(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		for (int m = 0; m < in_c; m++) {
			p_bag.push_back(mask[m]);
			g_bag.push_back(mask_grad[m]);
			len_bag.push_back(out_c * mask_plain_size);
		}
	}

	virtual void setCache(const int& new_cache_size) {
		darr::resize(in_cache, this->n_cache, new_cache_size, n_in);
		darr::resize(in_grad_cache, this->n_cache, new_cache_size, n_in);
		this->n_cache = new_cache_size;
	}

	void forward(darr::v& output, const int& id) {

		for (int i = 0; i < n_out; i++) {
			output[i] = 0.0;
		}
		for (int oc = 0; oc < out_c; oc++) {
			for (int ic = 0; ic < in_c; ic++) {
				for (int ih = 0, oh = 0; ih < in_h; ih++, oh += h_strd) {
					for (int iw = 0, ow = 0; iw < in_w; iw++, ow += w_strd) {
						for (int mh = 0; mh < mask_h; mh++) {
							for (int mw = 0; mw < mask_w; mw++) {
								output[(oc * out_plain_size) + ((oh + mh) * out_w) + (ow + mw)]
									+= mask[ic][(oc * mask_plain_size) + (mh * mask_w) + (mw)]
									* (in_cache[id][(ic * (in_plain_size)) + (ih * in_w) + (iw)]);
							}
						}
					}
				}
			}
		}
	}

	void backward(darr::v& output_grad, const int& id) {

		for (int i = 0; i < n_in; i++) {
			in_grad_cache[id][i] = 0.0;
		}

		for (int oc = 0; oc < out_c; oc++) {
			for (int ic = 0; ic < in_c; ic++) {
				for (int ih = 0, oh = 0; ih < in_h; ih++, oh += h_strd) {
					for (int iw = 0, ow = 0; iw < in_w; iw++, ow += w_strd) {
						for (int mh = 0; mh < mask_h; mh++) {
							for (int mw = 0; mw < mask_w; mw++) {
								in_grad_cache[id][(ic * (in_plain_size)) + (ih * in_w) + (iw)]
									+= mask[ic][(oc * mask_plain_size) + (mh * mask_w) + (mw)]
									* output_grad[(oc * out_plain_size) + ((oh + mh) * out_w) + (ow + mw)];
								mask_grad[ic][(oc * mask_plain_size) + (mh * mask_w) + (mw)]
									+= output_grad[(oc * out_plain_size) + ((oh + mh) * out_w) + (ow + mw)]
									* (in_cache[id][(ic * (in_plain_size)) + (ih * in_w) + (iw)]);
							}
						}
					}
				}
			}
		}
	}

	~Conv2DTransposed() {
		darr::free(mask, in_c);
		darr::free(mask_grad, in_c);
	}
};

class ReparametrizationBlock_StdNormal_KLD : public Unit {
public:


	Dense nn;

	darr::v2D rand_cache = nullptr;
	darr::v2D mean_cache = nullptr;
	darr::v2D stddev_cache = nullptr;

	darr::v nn_output = nullptr;
	darr::v nn_output_grad = nullptr;

	darr::v general_mean = nullptr;
	darr::v general_mean_grad = nullptr;
	darr::v pre_general_stddev = nullptr;
	darr::v pre_general_stddev_grad = nullptr;

	ReparametrizationBlock_StdNormal_KLD() {}
	ReparametrizationBlock_StdNormal_KLD(const int& n_input, const int& n_output) {
		create(n_input, n_output);
	}

	void create(const int& n_input, const int& n_output) {
		n_in = n_input;
		n_out = n_output;
		nn.create(n_in, n_out * 2);
		nn_output = darr::alloc(nn.n_out);
		nn_output_grad = darr::alloc(nn.n_out);


		general_mean = darr::alloc(n_out);
		general_mean_grad = darr::alloc(n_out);
		pre_general_stddev = darr::alloc(n_out);
		pre_general_stddev_grad = darr::alloc(n_out);
		for (int i = 0; i < n_out; i++) {
			general_mean[i] = 0.0;
			pre_general_stddev[i] = 0.0;
		}
	}

	virtual void setCache(const int& new_cache_size) {
		nn.setCache(new_cache_size);
		darr::resize(rand_cache, this->n_cache, new_cache_size, n_out);
		darr::resize(mean_cache, this->n_cache, new_cache_size, n_out);
		darr::resize(stddev_cache, this->n_cache, new_cache_size, n_out);

		in_cache = nn.in_cache;
		in_grad_cache = nn.in_grad_cache;
		this->n_cache = new_cache_size;
	}

	void forward(darr::v& output, const int& id) {	/* sampling */
		nn.forward(nn_output, id);
		for (int i = 0; i < n_out; i++) {
			rand_cache[id][i] = STD(rEngine);

			mean_cache[id][i] = nn_output[i];

			if (nn_output[i + n_out] > 0) {
				stddev_cache[id][i] = nn_output[i + n_out] + 1.0;
			}
			else {
				stddev_cache[id][i] = exp(nn_output[i + n_out]);
			}
			output[i] = stddev_cache[id][i] * rand_cache[id][i] + mean_cache[id][i];
		}
	}

	double get_KLD(const int& id) {
		double kld_loss = 0.0;

		for (int i = 0; i < n_out; i++) {
			if (pre_general_stddev[i] > 0.0) {
				kld_loss += log(pre_general_stddev[i] + 1.0) - log(stddev_cache[id][i]);
				kld_loss += -0.5 * rand_cache[id][i] * rand_cache[id][i];
				kld_loss += 0.5 * (stddev_cache[id][i] * rand_cache[id][i] + mean_cache[id][i] - general_mean[i])
					* (stddev_cache[id][i] * rand_cache[id][i] + mean_cache[id][i] - general_mean[i]) / ((pre_general_stddev[i] + 1.0) * (pre_general_stddev[i] + 1.0));
			}
			else {
				kld_loss += pre_general_stddev[i] - log(stddev_cache[id][i]);
				kld_loss += -0.5 * rand_cache[id][i] * rand_cache[id][i];
				kld_loss += 0.5 * (stddev_cache[id][i] * rand_cache[id][i] + mean_cache[id][i] - general_mean[i])
					* (stddev_cache[id][i] * rand_cache[id][i] + mean_cache[id][i] - general_mean[i]) / exp(pre_general_stddev[i] * 2.0);
			}
		}

		return kld_loss;
	}

	void backward(darr::v& output_grad, const int& id) {
		double mean_grad_of_kld = 0.0;
		double mean_grad_of_output = 0.0;
		double stddev_grad_of_output = 0.0;
		double stddev_grad_of_kld = 0.0;

		double general_stddev = 0.0;
		double general_stddev_grad = 0.0;

		for (int i = 0; i < n_out; i++) {

			if (pre_general_stddev[i] > 0.0) {
				general_stddev = pre_general_stddev[i] + 1.0;
			}
			else {
				general_stddev = exp(pre_general_stddev[i]);
			}

			mean_grad_of_output = output_grad[i];
			stddev_grad_of_output = rand_cache[id][i] * output_grad[i];
			mean_grad_of_kld = (rand_cache[id][i] * stddev_cache[id][i] + mean_cache[id][i]) / (general_stddev * general_stddev);
			stddev_grad_of_kld = mean_grad_of_kld * rand_cache[id][i] - 1.0 / stddev_cache[id][i];

			general_stddev_grad = (1.0 - mean_grad_of_kld * (stddev_cache[id][i] * rand_cache[id][i] + mean_cache[id][i])) / general_stddev;

			general_mean_grad[i] += -mean_grad_of_kld;
			if (pre_general_stddev[i] > 0.0) {
				pre_general_stddev_grad[i] += general_stddev_grad;
			}
			else {
				pre_general_stddev_grad[i] += general_stddev_grad * general_stddev;
			}

			nn_output_grad[i] = mean_grad_of_output + mean_grad_of_kld;
			if (stddev_cache[id][i] > 1) {
				nn_output_grad[i + n_out] = stddev_grad_of_output + stddev_grad_of_kld;
			}
			else {
				nn_output_grad[i + n_out] = stddev_cache[id][i] * (stddev_grad_of_output + stddev_grad_of_kld);
			}

		}
		nn.backward(nn_output_grad, id);
	}

	virtual void takeParams(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		nn.takeParams(p_bag, g_bag, len_bag);
		p_bag.push_back(pre_general_stddev);
		g_bag.push_back(pre_general_stddev_grad);
		len_bag.push_back(n_out);
		p_bag.push_back(general_mean);
		g_bag.push_back(general_mean_grad);
		len_bag.push_back(n_out);
	}
};


class Softmax :public Unit {
public:

	darr::v2D exp_cache = nullptr;
	darr::v denominator = nullptr;
	Softmax() {}
	Softmax(const int& size) { create(size); }
	~Softmax() {

	}
	void create(const int& input_size) {
		n_in = input_size;
		n_out = n_in;
	}

	virtual void setCache(const int& new_cache_size) {
		darr::resize(in_cache, this->n_cache, new_cache_size, n_in);
		darr::resize(in_grad_cache, this->n_cache, new_cache_size, n_in);
		darr::resize(exp_cache, this->n_cache, new_cache_size, n_in);
		darr::resize(denominator, this->n_cache);
		this->n_cache = new_cache_size;
	}

	void forward(darr::v& output, const int& id) {
		denominator[id] = 0.0;
		for (int i = 0; i < n_out; i++) {
			exp_cache[id][i] = exp(in_cache[id][i]);
			denominator[id] += exp_cache[id][i];
		}
		for (int i = 0; i < n_out; i++) {
			output[i] = exp_cache[id][i] / denominator[id];
		}
	}

	void backward(darr::v& output_grad, const int& id) {
		double coLoss = 0.0;

		coLoss = 0.0;
		for (int i = 0; i < n_out; i++) {
			coLoss += output_grad[i] * exp_cache[id][i];
		}
		coLoss = (-1.0 / (denominator[id] * denominator[id])) * coLoss;
		for (int i = 0; i < n_out; i++) {
			in_grad_cache[id][i] = exp_cache[id][i] * (coLoss + (output_grad[i] / (denominator[id])));
		}
	}
};