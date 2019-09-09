#pragma once
#include "core.h"
#include <fstream>


class unit {
public:

	int input_size, output_size;
	int max_batch_size, batch_size;
	double** in_port;
	double** downstream;
	double* _input_;
	unit() {}
	virtual void _charge_(double* const& _input) { km_1d::copy(_input_, _input, input_size); }
	virtual void charge(double** const& _input) { km_2d::copy(in_port, _input, batch_size, input_size); }
	virtual void UseMemory(const int& _using_size) {
		assert(max_batch_size >= _using_size);
		batch_size = _using_size;
	}
	~unit() {}
	virtual void AllocMemory(int b_size) = 0;
	virtual void save(ofstream& fout) = 0;
	virtual void load(ifstream& fin) = 0;
	virtual unit* clone() = 0;
	virtual void _calculate_(double*& next) = 0;
	virtual void forward(double**& next_port) = 0;
	virtual void backward(double** const& out_grad) = 0;
	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {}
};
class Softmax :public unit {
public:
	double _denominator_;
	double** exp_container;
	double* batch_denominator;
	bool content_allocated, batch_allocated;
	Softmax() { content_allocated = false; batch_allocated = false; }
	Softmax(int size) { create(size); }
	~Softmax() {
		if (content_allocated) {
			km_1d::free(_input_);
		}
		if (batch_allocated) {
			km_2d::free(in_port, max_batch_size);
			km_2d::free(downstream, max_batch_size);
			km_2d::free(exp_container, max_batch_size);
			km_1d::free(batch_denominator);
		}
	}
	void create(int input_sz) {
		input_size = input_sz;
		output_size = input_size;
		/**********************************************************/
		_input_ = km_1d::alloc(input_size);
		/**********************************************************/
		content_allocated = true;
	}

	void AllocMemory(int _max_batch_size) {
		if (batch_allocated) {
			km_2d::free(in_port, max_batch_size);
			km_2d::free(exp_container, max_batch_size);
			km_2d::free(downstream, max_batch_size);
			km_1d::free(batch_denominator);
		}
		max_batch_size = _max_batch_size;
		batch_denominator = km_1d::alloc(max_batch_size);
		in_port = km_2d::alloc(max_batch_size, input_size);
		exp_container = km_2d::alloc(max_batch_size, input_size);
		downstream = km_2d::alloc(max_batch_size, input_size);
		batch_size = _max_batch_size;
		batch_allocated = true;
	}

	void forward(double**& next_port) {
		for (int m = 0; m < batch_size; m++) {
			batch_denominator[m] = 0.0;
			for (int i = 0; i < output_size; i++) {
				exp_container[m][i] = exp(in_port[m][i]);
				batch_denominator[m] += exp_container[m][i];
			}
			for (int i = 0; i < output_size; i++) {
				next_port[m][i] = exp_container[m][i] / batch_denominator[m];
			}
		}
	}
	void _calculate_(double*& next) {
		_denominator_ = 0.0;
		for (int i = 0; i < output_size; i++) {
			next[i] = exp(_input_[i]);
			_denominator_ += next[i];
		}
		for (int i = 0; i < output_size; i++) {
			next[i] = next[i] / _denominator_;
		}
	}
	void backward(double** const& dLoss) {
		double coLoss = 0.0;
		for (int m = 0; m < batch_size; m++) {
			coLoss = 0.0;
			for (int i = 0; i < output_size; i++) {
				coLoss += dLoss[m][i] * exp_container[m][i];
			}
			coLoss = (-1.0 / (batch_denominator[m] * batch_denominator[m])) * coLoss;
			for (int i = 0; i < output_size; i++) {
				downstream[m][i] = exp_container[m][i] * (coLoss + (dLoss[m][i] / (batch_denominator[m])));
			}
		}
	}
	virtual unit* clone() {
		return new Softmax(input_size);
	}
	virtual void save(ofstream& fout) {
		fout << "Softmax" << endl; fout << input_size << endl;
	}
	virtual void load(ifstream& fin) {
		string buffer;
		getline(fin, buffer);
		int _input_size = atoi(buffer.c_str());
		create(_input_size);
	}
};

class fully_connected : public unit {
public:

	double* weight; double* w_grad;
	double* bias; double* b_grad;
	int w_len, b_len;
	bool content_allocated, batch_allocated;
	fully_connected() { content_allocated = false; batch_allocated = false; }
	fully_connected(int input_sz, int output_sz) { create(input_sz, output_sz); }
	virtual unit* clone() {
		fully_connected* _clone = new fully_connected(input_size, output_size);
		km_1d::copy(_clone->weight, weight, w_len);
		km_1d::copy(_clone->bias, bias, b_len);
		return _clone;
	}
	virtual void save(ofstream& fout) {
		fout << "fully_connected" << endl;
		fout << input_size << endl;
		fout << output_size << endl;
		for (int i = 0; i < w_len; i++) {
			fout << weight[i] << endl;
		}
		for (int i = 0; i < b_len; i++) {
			fout << bias[i] << endl;
		}
	}
	virtual void load(ifstream& fin) {
		string buffer;
		getline(fin, buffer);
		int _input_size = atoi(buffer.c_str());
		getline(fin, buffer);
		int _output_size = atoi(buffer.c_str());
		create(_input_size, _output_size);
		for (int i = 0; i < w_len; i++) {
			getline(fin, buffer);
			weight[i] = atof(buffer.c_str());
		}
		for (int i = 0; i < b_len; i++) {
			getline(fin, buffer);
			bias[i] = atof(buffer.c_str());
		}
	}
	void AllocMemory(int _max_batch_size) {
		if (batch_allocated) {
			km_2d::free(in_port, max_batch_size);
			km_2d::free(downstream, max_batch_size);
		}
		max_batch_size = _max_batch_size;
		in_port = km_2d::alloc(max_batch_size, input_size);
		downstream = km_2d::alloc(max_batch_size, input_size);
		batch_size = _max_batch_size;
		batch_allocated = true;
	}
	void create(int input_sz, int output_sz) {
		input_size = input_sz;
		output_size = output_sz;
		w_len = input_size * output_size;
		b_len = output_size;
		/**********************************************************/
		_input_ = km_1d::alloc(input_size);
		weight = km_1d::alloc(w_len); w_grad = km_1d::alloc(w_len);
		bias = km_1d::alloc(b_len); b_grad = km_1d::alloc(b_len);
		/**********************************************************/
		km_1d::fill_guassian_noise(weight, 0.0, 2.0 / (double)input_size, w_len);
		km_1d::fill_zero(bias, b_len);
		content_allocated = true;
	}
	void forward(double**& next_port) {
		for (int m = 0; m < batch_size; m++) {
			for (int w = 0; w < output_size; w++) {
				next_port[m][w] = bias[w];
			}
			for (int h = 0; h < input_size; h++) {
				for (int w = 0; w < output_size; w++) {
					next_port[m][w] += weight[h * output_size + w] * in_port[m][h];
				}
			}
		}
	}
	void _calculate_(double*& next) {
		for (int w = 0; w < output_size; w++) {
			next[w] = bias[w];
		}
		for (int h = 0; h < input_size; h++) {
			for (int w = 0; w < output_size; w++) {
				next[w] += weight[h * output_size + w] * _input_[h];
			}
		}
	}
	void backward(double** const& dLoss) {
		for (int m = 0; m < batch_size; m++) {
			for (int w = 0; w < output_size; w++) {
				b_grad[w] += dLoss[m][w];
			}
			for (int h = 0; h < input_size; h++) {
				for (int w = 0; w < output_size; w++) {
					w_grad[h * output_size + w] += dLoss[m][w] * in_port[m][h];
				}
			}
			for (int h = 0; h < input_size; h++) {
				downstream[m][h] = 0.0;
				for (int w = 0; w < output_size; w++) {
					downstream[m][h] += dLoss[m][w] * weight[h * output_size + w];
				}
			}
		}
	}
	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		p_bag.push_back(weight);
		g_bag.push_back(w_grad);
		len_bag.push_back(w_len);
		p_bag.push_back(bias);
		g_bag.push_back(b_grad);
		len_bag.push_back(b_len);
	}
	~fully_connected() {
		if (content_allocated) {
			km_1d::free(weight); km_1d::free(w_grad);
			km_1d::free(bias); km_1d::free(b_grad);
			km_1d::free(_input_);
		}
		if (batch_allocated) {
			km_2d::free(in_port, max_batch_size);
			km_2d::free(downstream, max_batch_size);
		}
	}
};

enum ACTIVATION { SIGMOID, TANH, RELU };
class activation : public unit {
public:
	ACTIVATION fn;
	bool content_allocated, batch_allocated;
	activation() { content_allocated = false; }
	void create(int size) {
		input_size = size;
		output_size = size;
		_input_ = km_1d::alloc(input_size);
		content_allocated = false;
	}
	virtual void AllocMemory(int _max_batch_size) {
		if (batch_allocated) {
			km_2d::free(in_port, max_batch_size);
			km_2d::free(downstream, max_batch_size);
		}
		max_batch_size = _max_batch_size;
		batch_size = max_batch_size;
		in_port = km_2d::alloc(max_batch_size, input_size);
		downstream = km_2d::alloc(max_batch_size, input_size);
		batch_allocated = true;
	}
	~activation() {
		if (batch_allocated) {
			km_2d::free(in_port, max_batch_size);
			km_2d::free(downstream, max_batch_size);
		}
		if (content_allocated) {
			km_1d::free(_input_);
		}
	}
};
class ReLU : public activation {
public:
	ReLU() { content_allocated = false; fn = RELU; }
	ReLU(int size) { create(size); fn = RELU; }
	virtual unit* clone() {
		return new ReLU(input_size);
	}
	virtual void save(ofstream& fout) {
		fout << "ReLU" << endl; fout << input_size << endl;
	}
	virtual void load(ifstream& fin) {
		string buffer;
		getline(fin, buffer);
		int _input_size = atoi(buffer.c_str());
		create(_input_size);
	}
	void _calculate_(double*& next) {
		for (int i = 0; i < input_size; i++) {
			if (_input_[i] >= 0) {
				next[i] = _input_[i];
			}
			else {
				next[i] = 0.0;
			}
		}
	}
	void forward(double**& next_port) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < input_size; i++) {
				if (in_port[m][i] >= 0) {
					next_port[m][i] = in_port[m][i];
				}
				else {
					next_port[m][i] = 0.0;
				}
			}
		}
	}

	void backward(double** const& dLoss) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < input_size; i++) {
				if (in_port[m][i] >= 0) {
					downstream[m][i] = dLoss[m][i];
				}
				else {
					downstream[m][i] = 0.0;
				}
			}
		}
	}
};
class Tanh : public activation {
public:
	Tanh() { content_allocated = false; fn = TANH; }
	Tanh(int size) { create(size); fn = TANH; }
	virtual unit* clone() {
		return new Tanh(input_size);
	}
	virtual void save(ofstream& fout) {
		fout << "Tanh" << endl; fout << input_size << endl;
	}
	virtual void load(ifstream& fin) {
		string buffer;
		getline(fin, buffer);
		int _input_size = atoi(buffer.c_str());
		create(_input_size);
	}
	void _calculate_(double*& next) {
		double enx = 0.0, ex = 0.0;
		for (int i = 0; i < input_size; i++) {
			if (in_port[i] >= 0) { /*https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/  */
				enx = exp(-_input_[i]);
				ex = 1.0 / enx;
			}
			else {
				ex = exp(_input_[i]);
				enx = 1.0 / ex;
			}
			next[i] = (ex - enx) / (ex + enx);
		}
	}
	void forward(double**& next_port) {
		double enx = 0.0, ex = 0.0;
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < input_size; i++) {
				if (in_port[m][i] >= 0) {
					enx = exp(-in_port[m][i]);
					ex = 1.0 / enx;
				}
				else {
					ex = exp(in_port[m][i]);
					enx = 1.0 / ex;
				}
				next_port[m][i] = (ex - enx) / (ex + enx);
			}
		}
	}

	void backward(double** const& dLoss) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < input_size; i++) {
				if (in_port[m][i] >= 0) {
					downstream[m][i] = dLoss[m][i];
				}
				else {
					downstream[m][i] = 0.0;
				}
			}
		}
	}
};
class Sigmoid : public activation {
public:
	Sigmoid() { content_allocated = false; fn = SIGMOID; }
	Sigmoid(int size) { create(size); fn = SIGMOID; }
	virtual unit* clone() {
		return new Sigmoid(input_size);
	}
	virtual void save(ofstream& fout) {
		fout << "Sigmoid" << endl; fout << input_size << endl;
	}
	virtual void load(ifstream& fin) {
		string buffer;
		getline(fin, buffer);
		int _input_size = atoi(buffer.c_str());
		create(_input_size);
	}
	void _calculate_(double*& next) {
		double enx = 0.0, ex = 0.0;
		for (int i = 0; i < input_size; i++) {
			if (in_port[i] >= 0) { /*https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/  */
				enx = exp(-_input_[i]);
				ex = 1.0 / enx;
			}
			else {
				ex = exp(_input_[i]);
				enx = 1.0 / ex;
			}
			next[i] = (ex - enx) / (ex + enx);
		}
	}
	void forward(double**& next_port) {
		double enx = 0.0, ex = 0.0;
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < input_size; i++) {
				if (in_port[m][i] >= 0) {
					enx = exp(-in_port[m][i]);
					ex = 1.0 / enx;
				}
				else {
					ex = exp(in_port[m][i]);
					enx = 1.0 / ex;
				}
				next_port[m][i] = (ex - enx) / (ex + enx);
			}
		}
	}

	void backward(double** const& dLoss) {
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < input_size; i++) {
				if (in_port[m][i] >= 0) {
					downstream[m][i] = dLoss[m][i];
				}
				else {
					downstream[m][i] = 0.0;
				}
			}
		}
	}
};

class shape {
public:
	union {
		struct { int c, h, w; };
		struct { int d1, d2, d3; };
	};
	shape() {}
	shape(const int& _c, const int& _h, const int& _w) :c(_c), h(_h), w(_w) {}
	void copy(const shape& src) { c = src.c; h = src.h; w = src.w; }
	int len() { return c * h * w; }
};
bool shape_check(shape in, shape k_shape, shape out, int h_stride, int w_stride) {
	if (in.c == k_shape.c) {
		if (h_stride * (out.h - 1) + k_shape.h == in.h) {
			if (w_stride * (out.w - 1) + k_shape.w == in.w) {
				return true;
			}
		}
	}
	return false;
}
class Kernel3D : public unit {
public:
	shape in_dim, out_dim, kernel_dim;
	int h_strd, w_strd;
	bool content_allocated, batch_allocated;
	double** kernel; double** k_grad;
	double* bias; double* b_grad;
	int b_len, k_len;
	int in_plain_size, out_plain_size, kernel_plain_size;
	Kernel3D() { content_allocated = false; batch_allocated = false; }
	virtual unit* clone() {
		Kernel3D* _clone = new Kernel3D(in_dim, kernel_dim, out_dim, h_strd, w_strd);
		km_2d::copy(_clone->kernel, kernel, out_dim.c, k_len);
		km_1d::copy(_clone->bias, bias, out_dim.c);
		return _clone;
	}

	Kernel3D(shape in, shape k_shape, shape out, int h_stride, int w_stride) {
		create(in, k_shape, out, h_stride, w_stride);
	}

	virtual void save(ofstream& fout) {
		fout << "Kernel3D" << endl;
		fout << in_dim.c << endl; fout << in_dim.h << endl; fout << in_dim.w << endl;
		fout << kernel_dim.c << endl; fout << kernel_dim.h << endl; fout << kernel_dim.w << endl;
		fout << out_dim.c << endl; fout << out_dim.h << endl; fout << out_dim.w << endl;
		fout << h_strd << endl; fout << w_strd << endl;
		for (int k = 0; k < out_dim.c; k++) {
			for (int i = 0; i < k_len; i++) {
				fout << kernel[k][i] << endl;
			}
			fout << bias[k] << endl;
		}
	}

	virtual void load(ifstream& fin) {
		string buffer;
		int c, h, w;
		getline(fin, buffer); c = atoi(buffer.c_str());
		getline(fin, buffer); h = atoi(buffer.c_str());
		getline(fin, buffer); w = atoi(buffer.c_str());

		shape _in_dim(c, h, w);
		getline(fin, buffer); c = atoi(buffer.c_str());
		getline(fin, buffer); h = atoi(buffer.c_str());
		getline(fin, buffer); w = atoi(buffer.c_str());

		shape _kernel_dim(c, h, w);
		getline(fin, buffer); c = atoi(buffer.c_str());
		getline(fin, buffer); h = atoi(buffer.c_str());
		getline(fin, buffer); w = atoi(buffer.c_str());

		shape _out_dim(c, h, w);
		getline(fin, buffer); int _hstrd = atoi(buffer.c_str());
		getline(fin, buffer); int _wstrd = atoi(buffer.c_str());

		create(_in_dim, _kernel_dim, _out_dim, _hstrd, _wstrd);
		for (int k = 0; k < out_dim.c; k++) {
			for (int i = 0; i < k_len; i++) {
				getline(fin, buffer);
				kernel[k][i] = atof(buffer.c_str());
			}
			getline(fin, buffer);
			bias[k] = atof(buffer.c_str());
		}
	}

	void create(shape in, shape k_shape, shape out, int h_stride, int w_stride) {
		assert(shape_check(in, k_shape, out, h_stride, w_stride));
		input_size = in.c * in.h * in.w; output_size = out.c * out.h * out.w;
		b_len = out.c; k_len = k_shape.c * k_shape.h * k_shape.w;
		in_dim.copy(in); out_dim.copy(out);	kernel_dim.copy(k_shape);
		in_plain_size = in_dim.w * in_dim.h;
		out_plain_size = out_dim.w * out_dim.h;
		kernel_plain_size = kernel_dim.w * kernel_dim.h;
		w_strd = w_stride; h_strd = h_stride;
		kernel = km_2d::alloc(out.c, k_shape.c * k_shape.h * k_shape.w);
		k_grad = km_2d::alloc(out.c, k_shape.c * k_shape.h * k_shape.w);
		bias = km_1d::alloc(b_len); b_grad = km_1d::alloc(b_len);
		km_1d::fill_zero(bias, b_len);
		km_2d::fill_guassian_noise(kernel, 0.0, (2.0 / input_size), out.c, k_len);
		_input_ = km_1d::alloc(input_size);
		content_allocated = true;
		batch_allocated = false;
	}

	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		for (int k = 0; k < out_dim.c; k++) {
			p_bag.push_back(kernel[k]);
			g_bag.push_back(k_grad[k]);
			len_bag.push_back(k_len);
		}
		p_bag.push_back(bias);
		g_bag.push_back(b_grad);
		len_bag.push_back(b_len);
	}

	virtual void AllocMemory(int _max_batch_size) {
		if (batch_allocated) {
			km_2d::free(in_port, max_batch_size);
			km_2d::free(downstream, max_batch_size);
		}
		max_batch_size = _max_batch_size;
		batch_size = max_batch_size;
		in_port = km_2d::alloc(max_batch_size, input_size);
		downstream = km_2d::alloc(max_batch_size, input_size);
		batch_allocated = true;
	}

	~Kernel3D() {
		if (content_allocated) {
			km_2d::free(kernel, out_dim.c);
			km_2d::free(k_grad, out_dim.c);
			km_1d::free(bias);
			km_1d::free(b_grad);
			km_1d::free(_input_);
		}
		if (batch_allocated) {
			km_2d::free(in_port, max_batch_size);
			km_2d::free(downstream, max_batch_size);
		}
	}

	void _calculate_(double*& next) {
		km_1d::fill_zero(next, output_size);
		for (int out_c = 0; out_c < out_dim.c; out_c++) {
			for (int k = 0; k < out_plain_size; k++) {
				next[(out_c * out_plain_size) + k] = bias[out_c];
			}
			for (int in_c = 0; in_c < in_dim.c; in_c++) {
				for (int in_h = 0, out_h = 0; out_h < out_dim.h; in_h += h_strd, out_h++) {
					for (int in_w = 0, out_w = 0; out_w < out_dim.w; in_w += w_strd, out_w++) {
						for (int kh = 0; kh < kernel_dim.h; kh++) {
							for (int kw = 0; kw < kernel_dim.w; kw++) { //out[out_c][out_h][out_w] += kernel[out_c][in_c][kernel_h][kernel_w] * in[in_c][in_h + kernel_h][in_w + kernel_w]
								next[(out_c * out_plain_size) + (out_h * out_dim.w) + (out_w)] += kernel[out_c][(in_c * kernel_plain_size) + (kh * kernel_dim.w) + (kw)]
									* _input_[(in_c * (in_plain_size)) + (in_dim.w * (in_h + kh)) + (in_w + kw)];
							}
						}
					}
				}
			}
		}
	}

	void forward(double**& next_port) {
		km_2d::fill_zero(next_port, batch_size, output_size);
		for (int m = 0; m < batch_size; m++) {
			for (int out_c = 0; out_c < out_dim.c; out_c++) {
				for (int k = 0; k < out_plain_size; k++) {
					next_port[m][(out_c * out_plain_size) + k] = bias[out_c];
				}
				for (int in_c = 0; in_c < in_dim.c; in_c++) {
					for (int in_h = 0, out_h = 0; out_h < out_dim.h; in_h += h_strd, out_h++) {
						for (int in_w = 0, out_w = 0; out_w < out_dim.w; in_w += w_strd, out_w++) {
							for (int kh = 0; kh < kernel_dim.h; kh++) {
								for (int kw = 0; kw < kernel_dim.w; kw++) { //out[out_c][out_h][out_w] += kernel[out_c][in_c][kernel_h][kernel_w] * in[in_c][in_h + kernel_h][in_w + kernel_w]
									next_port[m][(out_c * out_plain_size) + (out_h * out_dim.w) + (out_w)] += kernel[out_c][(in_c * kernel_plain_size) + (kh * kernel_dim.w) + (kw)]
										* in_port[m][(in_c * (in_plain_size)) + (in_dim.w * (in_h + kh)) + (in_w + kw)];
								}
							}
						}
					}
				}
			}
		}
	}

	void backward(double** const& dLoss) {
		km_2d::fill_zero(downstream, batch_size, input_size);
		for (int m = 0; m < batch_size; m++) {
			for (int out_c = 0; out_c < out_dim.c; out_c++) {
				for (int k = 0; k < out_plain_size; k++) {
					b_grad[out_c] += dLoss[m][(out_c * out_plain_size) + k];
				}
				for (int in_c = 0; in_c < in_dim.c; in_c++) {
					for (int in_h = 0, out_h = 0; out_h < out_dim.h; in_h += h_strd, out_h++) {
						for (int in_w = 0, out_w = 0; out_w < out_dim.w; in_w += w_strd, out_w++) {
							for (int kh = 0; kh < kernel_dim.h; kh++) {
								for (int kw = 0; kw < kernel_dim.w; kw++) { //out[out_c][out_h][out_w] += kernel[out_c][in_c][kernel_h][kernel_w] * in[in_c][in_h + kernel_h][in_w + kernel_w]
									k_grad[out_c][(in_c * kernel_plain_size) + (kh * kernel_dim.w) + (kw)] += dLoss[m][(out_c * out_plain_size) + (out_h * out_dim.w) + (out_w)]
										* in_port[m][(in_c * (in_plain_size)) + (in_dim.w * (in_h + kh)) + (in_w + kw)];
									downstream[m][(in_c * (in_plain_size)) + (in_dim.w * (in_h + kh)) + (in_w + kw)] += dLoss[m][(out_c * out_plain_size) + (out_h * out_dim.w) + (out_w)] *
										kernel[out_c][(in_c * kernel_plain_size) + (kh * kernel_dim.w) + (kw)];
								}
							}
						}
					}
				}
			}
		}
	}
};

class n_Layer {
public:
	int n;
	n_Layer() { n = 0; }
	n_Layer(int _n) { n = _n; }
};

class bundle : public unit {
public:
	unit** layer;
	int n_layer;
	bool content_allocated;
	bundle() { content_allocated = false; }
	~bundle() {
		if (content_allocated) {
			for (int i = 0; i < n_layer; i++) {
				delete layer[i];
			}
			delete[] layer;
		}
	}
	bundle(n_Layer number_of_layers) {
		n_layer = number_of_layers.n;
		layer = new unit * [n_layer];
		content_allocated = false;
	}
	virtual void save(ofstream& fout) {
		fout << "nn" << endl;
		fout << n_layer << endl;
		for (int i = 0; i < n_layer; i++) {
			layer[i]->save(fout);
		}
	}

	virtual void publish() { //set input, output
		for (int i = 0; i < n_layer - 1; i++) {
			assert(layer[i]->output_size == layer[i + 1]->input_size);
		}
		input_size = layer[0]->input_size;
		output_size = layer[n_layer - 1]->output_size;
		_input_ = layer[0]->_input_;
		content_allocated = true;
	}

	virtual void load(ifstream& fin) {
		string buffer;
		getline(fin, buffer);
		n_layer = atoi(buffer.c_str());
		layer = new unit * [n_layer];
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
				layer[i] = new Kernel3D();
			}
			//else if (buffer == "BatchNormalizer") {
			//	layer[i] = new BatchNormalizer();
			//}
			layer[i]->load(fin);
		}
	}

	virtual unit* clone() {
		bundle* _clone = new bundle(n_Layer(n_layer));
		for (int i = 0; i < n_layer; i++) {
			_clone->layer[i] = layer[i]->clone();
		}
		return _clone;
	}

	virtual void AllocMemory(int _max_batch_size) {
		max_batch_size = _max_batch_size;
		batch_size = max_batch_size;
		for (int i = 0; i < n_layer; i++) {
			layer[i]->AllocMemory(max_batch_size);
		}
		in_port = layer[0]->in_port;
		downstream = layer[0]->downstream;
	}

	virtual void UseMemory(const int& _using_size) {
		assert(max_batch_size >= _using_size);
		batch_size = _using_size;
		for (int i = 0; i < n_layer; i++) {
			layer[i]->UseMemory(batch_size);
		}
	}

	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		for (int i = 0; i < n_layer; i++) {
			layer[i]->delegate(p_bag, g_bag, len_bag);
		}
	}

	virtual void forward(double**& output) {
		for (int i = 0; i < n_layer - 1; i++) {
			layer[i]->forward(layer[i + 1]->in_port);
		}
		layer[n_layer - 1]->forward(output);
	}

	void _calculate_(double*& next) {
		for (int i = 0; i < n_layer - 1; i++) {
			layer[i]->_calculate_(layer[i + 1]->_input_);
		}
		layer[n_layer - 1]->_calculate_(next);
	}

	virtual void backward(double** const& dLoss) {
		layer[n_layer - 1]->backward(dLoss);
		for (int i = n_layer - 2; i >= 0; i--) {
			layer[i]->backward(layer[i + 1]->downstream);
		}
	}
};