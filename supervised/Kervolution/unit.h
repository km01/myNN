#pragma once
#include "core.h"

#define _square_(x) ((x)*(x))
#define _cube_(x) ((x)*((x)*(x)))
class unit {
public:

	int input_size;
	int output_size;
	int max_batch_size;
	int using_size;

	double** input_port;
	double** downstream;

	double* in_port;

	unit() {}

	virtual void alloc(int b_size) = 0; // 배치사이즈만큼 메모리를 잡아줍니다
	virtual void set_input(double** const& input) {
		for (int i = 0; i < using_size; i++) {
			for (int j = 0; j < input_size; j++) {
				input_port[i][j] = input[i][j];
			}
		}
	}
	virtual void charge(double* const& input) {
		for (int i = 0; i < input_size; i++) {
			in_port[i] = input[i];
		}
	}
	virtual void UseMemory(const int& _using_size) {
		assert(max_batch_size >= _using_size);
		using_size = _using_size;
	}
	~unit() {

	}
	virtual unit* clone() = 0;
	virtual void calculate(double* &next) = 0;
	virtual void forward(double** &next_port) = 0; // input으로부터 output을 계산해 next_port에 담습니다.
	virtual void backward(double** const& dLoss) = 0; //  Loss에 대한 유닛의 아웃풋의 편미분이 들어옵니다. 파라미터마다 gradient를 계산하고, input의 편미분을 계산해서 downstream에 담습니다.  
	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {} //learnable parameter, gradient pointer 를 optimizer에게 넘겨줌
};

class learnable : public unit {
public:
	bool p_alloc;
	bool c_alloc;
	double** input_port_container;
	double** downstream_container;
	learnable() { p_alloc = false; c_alloc = false; }
};

class fully_connected : public learnable {
public:

	double* weight; double* w_grad;
	double* bias; double* b_grad;
	int w_len, b_len;
	fully_connected() { p_alloc = false; c_alloc = false; }
	virtual unit* clone() {
		fully_connected* _clone = new fully_connected(input_size, output_size);
		for (int i = 0; i < w_len; i++) {
			_clone->weight[i] = weight[i];
		}
		for (int i = 0; i < b_len; i++) {
			_clone->bias[i] = bias[i];
		}
		return _clone;
	}
	fully_connected(int input_sz, int output_sz) { create(input_sz, output_sz); }
	void alloc(int b_size) {
		if (c_alloc) {
			km_2d::free(input_port_container, max_batch_size);
			km_2d::free(downstream_container, max_batch_size);
		}
		max_batch_size = b_size;
		input_port_container = km_2d::alloc(max_batch_size, input_size);
		downstream_container = km_2d::alloc(max_batch_size, input_size);
		input_port = input_port_container;
		downstream = downstream_container;
		using_size = max_batch_size;
		c_alloc = true;
	}
	void create(int input_sz, int output_sz) {
		c_alloc = false;
		input_size = input_sz;
		output_size = output_sz;
		w_len = input_size * output_size;
		b_len = output_size;
		weight = km_1d::alloc(w_len); w_grad = km_1d::alloc(w_len);
		bias = km_1d::alloc(b_len); b_grad = km_1d::alloc(b_len);
		km_1d::guassian_noise(weight, 0.0, 2.0 / (double)input_size, w_len);
		km_1d::fill_zero(bias, b_len);
		p_alloc = true;

		in_port = km_1d::alloc(input_size);
	}
	void forward(double** &next_port) {
		for (int m = 0; m < using_size; m++) {
			for (int w = 0; w < output_size; w++) {
				next_port[m][w] = bias[w];
			}
			for (int h = 0; h < input_size; h++) {
				for (int w = 0; w < output_size; w++) {
					next_port[m][w] += weight[h*output_size + w] * input_port[m][h];
				}
			}
		}
	}
	void calculate(double* &next) {
		for (int w = 0; w < output_size; w++) {
			next[w] = bias[w];
		}
		for (int h = 0; h < input_size; h++) {
			for (int w = 0; w < output_size; w++) {
				next[w] += weight[h*output_size + w] * in_port[h];
			}
		}
	}
	void backward(double** const& dLoss) {
		for (int m = 0; m < using_size; m++) {
			for (int w = 0; w < output_size; w++) {
				b_grad[w] += dLoss[m][w];
			}
			for (int h = 0; h < input_size; h++) {
				for (int w = 0; w < output_size; w++) {
					w_grad[h*output_size + w] += dLoss[m][w] * input_port[m][h];
				}
			}
			for (int h = 0; h < input_size; h++) {
				downstream[m][h] = 0.0;
				for (int w = 0; w < output_size; w++) {
					downstream[m][h] += dLoss[m][w] * weight[h*output_size + w];
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
		if (p_alloc) {
			km_1d::free(weight); km_1d::free(w_grad);
			km_1d::free(bias); km_1d::free(b_grad);
		}
		if (c_alloc) {
			km_2d::free(input_port_container, max_batch_size);
			km_2d::free(downstream_container, max_batch_size);
		}
		in_port = km_1d::alloc(input_size);
	}
};
enum ACTIVATION { SIGMOID, TANH, RELU };
class activation : public unit {
public:
	ACTIVATION fn;
	bool c_alloc;
	double** input_port_container;
	double** downstream_container;
	activation() { c_alloc = false; }
	void create(int size) {
		input_size = size;
		output_size = size;
		c_alloc = false;

		in_port = km_1d::alloc(input_size);
	}
	virtual void alloc(int b_size) {
		if (c_alloc) {
			km_2d::free(input_port_container, max_batch_size);
			km_2d::free(downstream_container, max_batch_size);
		}
		max_batch_size = b_size;
		input_port_container = km_2d::alloc(max_batch_size, input_size);
		downstream_container = km_2d::alloc(max_batch_size, input_size);

		input_port = input_port_container;
		downstream = downstream_container;
		using_size = max_batch_size;
		c_alloc = true;
	}
	~activation() {
		if (c_alloc) {
			km_2d::free(input_port_container, max_batch_size);
			km_2d::free(downstream_container, max_batch_size);
		}
		in_port = km_1d::alloc(input_size);
	}
};

activation* define(ACTIVATION fn, int size);
class sigmoid : public activation {
public:
	sigmoid() { c_alloc = false; fn = SIGMOID; }
	sigmoid(int size) { create(size); fn = SIGMOID; }
	virtual unit* clone() {
		return new sigmoid(input_size);
	}
	void calculate(double* &next) {
		for (int i = 0; i < input_size; i++) {
			if (in_port[i] >= 0) { // inf를 방지하고자 x가 클경우 e^x를 1.0/(e^-x)로 계산함. 참고 : https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
				next[i] = 1.0 / (1.0 + exp(-in_port[i]));
			}
			else {
				next[i] = exp(in_port[i]) / (1.0 + exp(in_port[i]));
			}
		}
	}
	void forward(double** &next_port) {
		for (int m = 0; m < using_size; m++) {
			for (int i = 0; i < input_size; i++) {
				if (input_port_container[m][i] >= 0) { // inf를 방지하고자 x가 클경우 e^x를 1.0/(e^-x)로 계산함. 참고 : https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
					next_port[m][i] = 1.0 / (1.0 + exp(-input_port_container[m][i]));
				}
				else {
					next_port[m][i] = exp(input_port_container[m][i]) / (1.0 + exp(input_port_container[m][i]));
				}
			}
		}
	}

	void backward(double** const& dLoss) {
		double f = 0.0;
		for (int m = 0; m < using_size; m++) {
			for (int i = 0; i < input_size; i++) {
				if (input_port_container[m][i] >= 0) {
					f = 1.0 / (1.0 + exp(-input_port_container[m][i]));
				}
				else {
					f = exp(input_port_container[m][i]) / (1.0 + exp(input_port_container[m][i]));
				}
				downstream[m][i] = ((1.0 - f)*f)*dLoss[m][i];
			}
		}
	}
};
class ReLU : public activation {
public:
	ReLU() { c_alloc = false; fn = RELU; }
	ReLU(int size) { create(size); fn = RELU; }
	virtual unit* clone() {
		return new ReLU(input_size);
	}
	void calculate(double* &next) {
		for (int i = 0; i < input_size; i++) {
			if (in_port[i] >= 0) {
				next[i] = in_port[i];
			}
			else {
				next[i] = 0.0;
			}
		}
	}
	void forward(double** &next_port) {
		for (int m = 0; m < using_size; m++) {
			for (int i = 0; i < input_size; i++) {
				if (input_port[m][i] >= 0) {
					next_port[m][i] = input_port[m][i];
				}
				else {
					next_port[m][i] = 0.0;
				}
			}
		}
	}

	void backward(double** const& dLoss) {
		for (int m = 0; m < using_size; m++) {
			for (int i = 0; i < input_size; i++) {
				if (input_port_container[m][i] >= 0) {
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
	Tanh() { c_alloc = false; fn = TANH; }
	Tanh(int size) { create(size); fn = TANH; }
	virtual unit* clone() {
		return new Tanh(input_size);
	}
	void forward(double** &next_port) {
		double enx = 0.0, ex = 0.0;
		for (int m = 0; m < using_size; m++) {
			for (int i = 0; i < input_size; i++) {
				if (input_port[m][i] >= 0) {
					enx = exp(-input_port[m][i]);
					ex = 1.0 / enx;
				}
				else {
					ex = exp(input_port[m][i]);
					enx = 1.0 / ex;
				}
				next_port[m][i] = (ex - enx) / (ex + enx);
			}
		}
	}
	void calculate(double* &next) {
		double enx = 0.0, ex = 0.0;
		for (int i = 0; i < input_size; i++) {
			if (in_port[i] >= 0) { // inf를 방지하고자 x가 클경우 e^x를 1.0/(e^-x)로 계산함. 참고 : https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
				enx = exp(-in_port[i]);
				ex = 1.0 / enx;
			}
			else {
				ex = exp(in_port[i]);
				enx = 1.0 / ex;
			}
			next[i] = (ex - enx) / (ex + enx);
		}
	}
	void backward(double** const& dLoss) {
		double enx = 0.0, ex = 0.0;
		for (int m = 0; m < using_size; m++) {
			for (int i = 0; i < input_size; i++) {
				if (input_port_container[m][i] >= 0) {
					enx = exp(-input_port_container[m][i]);
					ex = 1.0 / enx;
				}
				else {
					ex = exp(input_port_container[m][i]);
					enx = 1.0 / ex;
				}
				downstream[m][i] = (1.0 - ((ex - enx) / (ex + enx)) * ((ex - enx) / (ex + enx)))*dLoss[m][i];
			}
		}
	}
};
class bundle :public unit {
public:
	bool content_alloc;
	bundle() { content_alloc = false; }
};

activation* define(ACTIVATION fn, int size) {
	if (fn == SIGMOID) {
		return new sigmoid(size);
	}
	else if (fn == TANH) {
		return new Tanh(size);
	}
	else {
		return new ReLU(size);
	}
}
class shape
{
public:
	union {
		struct { int c, h, w; };
		struct { int d1, d2, d3; };
	};
	shape() {}
	shape(const int& _c, const int& _h, const int& _w) :c(_c), h(_h), w(_w) {}
	void copy(const shape& src) {
		c = src.c; h = src.h; w = src.w;
	}
	int len() {
		return c * h*w;
	}
};

bool shape_check(shape in, shape k_shape, shape out, int h_stride, int w_stride) {
	if (in.c == k_shape.c) {

		if (h_stride*(out.h - 1) + k_shape.h == in.h) {

			if (w_stride*(out.w - 1) + k_shape.w == in.w) {
				return true;
			}
		}
	}
	return false;
}

class kernel3D : public learnable {
public:
	shape in_dim;
	shape out_dim;
	shape kernel_dim;
	int h_strd;
	int w_strd;
	bool w_alloc;
	double** kernel;
	int b_len;
	int k_len;
	double** k_grad;
	double* bias;
	double* b_grad;
	int stride;
	int in_plain_size, out_plain_size, kernel_plain_size;
	kernel3D() { w_alloc = false; c_alloc = false; }
	virtual unit* clone() {
		kernel3D* _clone = new kernel3D(in_dim, kernel_dim, out_dim, h_strd, w_strd);
		for (int k = 0; k < out_dim.c; k++) {
			for (int i = 0; i < k_len; i++) {
				_clone->kernel[k][i] = kernel[k][i];
			}
			_clone->bias[k] = bias[k];
		}
		return _clone;
	}

	kernel3D(shape in, shape k_shape, shape out, int h_stride, int w_stride) {
		create(in, k_shape, out, h_stride, w_stride);
	}

	void create(shape in, shape k_shape, shape out, int h_stride, int w_stride) {
		assert(shape_check(in, k_shape, out, h_stride, w_stride));
		c_alloc = false;
		input_size = in.c*in.h*in.w;
		output_size = out.c*out.h*out.w;
		b_len = out.c;
		k_len = k_shape.c*k_shape.h*k_shape.w;
		in_dim.copy(in);
		out_dim.copy(out);
		kernel_dim.copy(k_shape);
		in_plain_size = in_dim.w* in_dim.h;
		out_plain_size = out_dim.w* out_dim.h;
		kernel_plain_size = kernel_dim.w* kernel_dim.h;
		w_strd = w_stride; h_strd = h_stride;
		kernel = km_2d::alloc(out.c, k_shape.c*k_shape.h*k_shape.w);
		k_grad = km_2d::alloc(out.c, k_shape.c*k_shape.h*k_shape.w);
		bias = km_1d::alloc(b_len);
		b_grad = km_1d::alloc(b_len);
		km_1d::fill_zero(bias, b_len);
		km_2d::guassian_noise(kernel, 0.0, (2.0 / input_size), out.c, k_len);
		w_alloc = true;
		in_port = km_1d::alloc(input_size);
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

	void alloc(int b_size) {
		if (c_alloc) {
			km_2d::free(input_port_container, max_batch_size);
			km_2d::free(downstream_container, max_batch_size);
		}
		max_batch_size = b_size;
		using_size = max_batch_size;
		input_port_container = km_2d::alloc(max_batch_size, input_size);
		downstream_container = km_2d::alloc(max_batch_size, input_size);
		input_port = input_port_container;
		downstream = downstream_container;
		c_alloc = true;
	}

	~kernel3D() {
		if (w_alloc) {
			km_2d::free(kernel, out_dim.c);
			km_2d::free(k_grad, out_dim.c);
			km_1d::free(bias);
			km_1d::free(b_grad);
		}
		if (c_alloc) {
			km_2d::free(input_port_container, max_batch_size);
			km_2d::free(downstream_container, max_batch_size);
		}
		km_1d::free(in_port);
	}

	void calculate(double* &next) {
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
								next[(out_c * out_plain_size) + (out_h*out_dim.w) + (out_w)] += kernel[out_c][(in_c*kernel_plain_size) + (kh * kernel_dim.w) + (kw)]
									* in_port[(in_c * (in_plain_size)) + (in_dim.w *(in_h + kh)) + (in_w + kw)];
							}
						}
					}
				}
			}
		}
	}
	void forward(double** &next_port) {
		km_2d::fill_zero(next_port, using_size, output_size);
		for (int m = 0; m < using_size; m++) {
			for (int out_c = 0; out_c < out_dim.c; out_c++) {
				for (int k = 0; k < out_plain_size; k++) {
					next_port[m][(out_c * out_plain_size) + k] = bias[out_c];
				}
				for (int in_c = 0; in_c < in_dim.c; in_c++) {
					for (int in_h = 0, out_h = 0; out_h < out_dim.h; in_h += h_strd, out_h++) {
						for (int in_w = 0, out_w = 0; out_w < out_dim.w; in_w += w_strd, out_w++) {
							for (int kh = 0; kh < kernel_dim.h; kh++) {
								for (int kw = 0; kw < kernel_dim.w; kw++) { //out[out_c][out_h][out_w] += kernel[out_c][in_c][kernel_h][kernel_w] * in[in_c][in_h + kernel_h][in_w + kernel_w]
									next_port[m][(out_c * out_plain_size) + (out_h*out_dim.w) + (out_w)] += kernel[out_c][(in_c*kernel_plain_size) + (kh * kernel_dim.w) + (kw)]
										* input_port[m][(in_c * (in_plain_size)) + (in_dim.w *(in_h + kh)) + (in_w + kw)];
								}
							}
						}
					}
				}
			} 
		}
	}

	void backward(double** const& dLoss) {
		km_2d::fill_zero(downstream, using_size, input_size);
		for (int m = 0; m < using_size; m++) {
			for (int out_c = 0; out_c < out_dim.c; out_c++) {
				for (int k = 0; k < out_plain_size; k++) {
					b_grad[out_c] += dLoss[m][(out_c * out_plain_size) + k];
				}
				for (int in_c = 0; in_c < in_dim.c; in_c++) {
					for (int in_h = 0, out_h = 0; out_h < out_dim.h; in_h += h_strd, out_h++) {
						for (int in_w = 0, out_w = 0; out_w < out_dim.w; in_w += w_strd, out_w++) {
							//b_grad[out_c] += dLoss[m][(out_c * out_plain_size) + (out_h*out_dim.w) + (out_w)];
							for (int kh = 0; kh < kernel_dim.h; kh++) {
								for (int kw = 0; kw < kernel_dim.w; kw++) { //out[out_c][out_h][out_w] += kernel[out_c][in_c][kernel_h][kernel_w] * in[in_c][in_h + kernel_h][in_w + kernel_w]
									k_grad[out_c][(in_c*kernel_plain_size) + (kh * kernel_dim.w) + (kw)] += dLoss[m][(out_c * out_plain_size) + (out_h*out_dim.w) + (out_w)]
										* input_port[m][(in_c * (in_plain_size)) + (in_dim.w *(in_h + kh)) + (in_w + kw)];
									downstream[m][(in_c * (in_plain_size)) + (in_dim.w *(in_h + kh)) + (in_w + kw)] += dLoss[m][(out_c * out_plain_size) + (out_h*out_dim.w) + (out_w)] *
										kernel[out_c][(in_c*kernel_plain_size) + (kh * kernel_dim.w) + (kw)];
								}
							}
						}
					}
				}
			}
		}
	}
};

class L2NormKernel3D : public learnable {
public:
	shape in_dim;
	shape out_dim;
	shape kernel_dim;
	int h_strd;
	int w_strd;
	bool w_alloc;
	double** kernel;
	int k_len;
	double** k_grad;
	int stride;
	int in_plain_size, out_plain_size, kernel_plain_size;
	L2NormKernel3D() { w_alloc = false; c_alloc = false; }
	virtual unit* clone() {
		L2NormKernel3D* _clone = new L2NormKernel3D(in_dim, kernel_dim, out_dim, h_strd, w_strd);
		for (int k = 0; k < out_dim.c; k++) {
			for (int i = 0; i < k_len; i++) {
				_clone->kernel[k][i] = kernel[k][i];
			}
		}
		return _clone;
	}

	L2NormKernel3D(shape in, shape k_shape, shape out, int h_stride, int w_stride) {
		create(in, k_shape, out, h_stride, w_stride);
	}

	void create(shape in, shape k_shape, shape out, int h_stride, int w_stride) {
		assert(shape_check(in, k_shape, out, h_stride, w_stride));
		c_alloc = false;
		input_size = in.c*in.h*in.w;
		output_size = out.c*out.h*out.w;
		k_len = k_shape.c*k_shape.h*k_shape.w;
		in_dim.copy(in);
		out_dim.copy(out);
		kernel_dim.copy(k_shape);
		in_plain_size = in_dim.w* in_dim.h;
		out_plain_size = out_dim.w* out_dim.h;
		kernel_plain_size = kernel_dim.w* kernel_dim.h;
		w_strd = w_stride; h_strd = h_stride;
		kernel = km_2d::alloc(out.c, k_shape.c*k_shape.h*k_shape.w);
		k_grad = km_2d::alloc(out.c, k_shape.c*k_shape.h*k_shape.w);
		km_2d::guassian_noise(kernel, 0.0, (2.0 / input_size), out.c, k_len);
		w_alloc = true;
		in_port = km_1d::alloc(input_size);
	}
	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		for (int k = 0; k < out_dim.c; k++) {
			p_bag.push_back(kernel[k]);
			g_bag.push_back(k_grad[k]);
			len_bag.push_back(k_len);
		}
	}

	void alloc(int b_size) {
		if (c_alloc) {
			km_2d::free(input_port_container, max_batch_size);
			km_2d::free(downstream_container, max_batch_size);
		}
		max_batch_size = b_size;
		using_size = max_batch_size;
		input_port_container = km_2d::alloc(max_batch_size, input_size);
		downstream_container = km_2d::alloc(max_batch_size, input_size);
		input_port = input_port_container;
		downstream = downstream_container;
		c_alloc = true;
	}

	~L2NormKernel3D() {
		if (w_alloc) {
			km_2d::free(kernel, out_dim.c);
			km_2d::free(k_grad, out_dim.c);
		}
		if (c_alloc) {
			km_2d::free(input_port_container, max_batch_size);
			km_2d::free(downstream_container, max_batch_size);
		}
		km_1d::free(in_port);
	}

	void calculate(double* &next) {
		km_1d::fill_zero(next, output_size);
		for (int out_c = 0; out_c < out_dim.c; out_c++) {
			for (int in_c = 0; in_c < in_dim.c; in_c++) {
				for (int in_h = 0, out_h = 0; out_h < out_dim.h; in_h += h_strd, out_h++) {
					for (int in_w = 0, out_w = 0; out_w < out_dim.w; in_w += w_strd, out_w++) {
						for (int kh = 0; kh < kernel_dim.h; kh++) {
							for (int kw = 0; kw < kernel_dim.w; kw++) { //out[out_c][out_h][out_w] += kernel[out_c][in_c][kernel_h][kernel_w] * in[in_c][in_h + kernel_h][in_w + kernel_w]
								next[(out_c * out_plain_size) + (out_h*out_dim.w) + (out_w)] += _square_(kernel[out_c][(in_c*kernel_plain_size) + (kh * kernel_dim.w) + (kw)]
									- in_port[(in_c * (in_plain_size)) + (in_dim.w *(in_h + kh)) + (in_w + kw)]);
							}
						}
					}
				}
			}
		}
	}
	void forward(double** &next_port) {
		km_2d::fill_zero(next_port, using_size, output_size);
		for (int m = 0; m < using_size; m++) {
			for (int out_c = 0; out_c < out_dim.c; out_c++) {
				for (int in_c = 0; in_c < in_dim.c; in_c++) {
					for (int in_h = 0, out_h = 0; out_h < out_dim.h; in_h += h_strd, out_h++) {
						for (int in_w = 0, out_w = 0; out_w < out_dim.w; in_w += w_strd, out_w++) {
							for (int kh = 0; kh < kernel_dim.h; kh++) {
								for (int kw = 0; kw < kernel_dim.w; kw++) { //out[out_c][out_h][out_w] += kernel[out_c][in_c][kernel_h][kernel_w] * in[in_c][in_h + kernel_h][in_w + kernel_w]
									next_port[m][(out_c * out_plain_size) + (out_h*out_dim.w) + (out_w)] += _square_(kernel[out_c][(in_c*kernel_plain_size) + (kh * kernel_dim.w) + (kw)]
										- input_port[m][(in_c * (in_plain_size)) + (in_dim.w *(in_h + kh)) + (in_w + kw)]);
								}
							}
						}
					}
				}
			}
		}
	}

	void backward(double** const& dLoss) {
		km_2d::fill_zero(downstream, using_size, input_size);
		for (int m = 0; m < using_size; m++) {
			for (int out_c = 0; out_c < out_dim.c; out_c++) {
				for (int in_c = 0; in_c < in_dim.c; in_c++) {
					for (int in_h = 0, out_h = 0; out_h < out_dim.h; in_h += h_strd, out_h++) {
						for (int in_w = 0, out_w = 0; out_w < out_dim.w; in_w += w_strd, out_w++) {
							//b_grad[out_c] += dLoss[m][(out_c * out_plain_size) + (out_h*out_dim.w) + (out_w)];
							for (int kh = 0; kh < kernel_dim.h; kh++) {
								for (int kw = 0; kw < kernel_dim.w; kw++) { //out[out_c][out_h][out_w] += kernel[out_c][in_c][kernel_h][kernel_w] * in[in_c][in_h + kernel_h][in_w + kernel_w]
									k_grad[out_c][(in_c*kernel_plain_size) + (kh * kernel_dim.w) + (kw)] += dLoss[m][(out_c * out_plain_size) + (out_h*out_dim.w) + (out_w)]
										* 2.0 * (kernel[out_c][(in_c*kernel_plain_size) + (kh * kernel_dim.w) + (kw)] - input_port[m][(in_c * (in_plain_size)) + (in_dim.w *(in_h + kh)) + (in_w + kw)]);
									downstream[m][(in_c * (in_plain_size)) + (in_dim.w *(in_h + kh)) + (in_w + kw)] += dLoss[m][(out_c * out_plain_size) + (out_h*out_dim.w) + (out_w)]
										* 2.0 * (input_port[m][(in_c * (in_plain_size)) + (in_dim.w *(in_h + kh)) + (in_w + kw)] - kernel[out_c][(in_c*kernel_plain_size) + (kh * kernel_dim.w) + (kw)]);
								}
							}
						}
					}
				}
			}
		}
	}
};
class GaussianRBF : public activation {
public:
	double* gamma;
	double* gamma_grad;
	GaussianRBF() { c_alloc = false;}
	GaussianRBF(int size) { 
		create(size); 
		gamma_grad = new double[1];
		gamma = new double[1];
		gamma[0] = 0.5;
		c_alloc = true;
	}
	virtual unit* clone() {
		GaussianRBF* _clone = new GaussianRBF(input_size);
		_clone->gamma[0] = gamma[0];
		return _clone;
	}
	void forward(double** &next_port) {
		for (int m = 0; m < using_size; m++) {
			for (int i = 0; i < input_size; i++) {
				next_port[m][i] = exp(-gamma[0] * input_port_container[m][i]);
			}
		}
	}
	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		p_bag.push_back(gamma);
		g_bag.push_back(gamma_grad);
		len_bag.push_back(1);
	}

	void calculate(double* &next) {
		double enx = 0.0, ex = 0.0;
		for (int i = 0; i < input_size; i++) {
			next[i] = exp(-gamma[0] * in_port[i]);
		}
	}
	void backward(double** const& dLoss) {
		double enx = 0.0, ex = 0.0;
		for (int m = 0; m < using_size; m++) {
			for (int i = 0; i < input_size; i++) {
				gamma_grad[0] += -input_port_container[m][i] * exp(-gamma[0] * input_port_container[m][i]) * dLoss[m][i];
				downstream[m][i] = -gamma[0] * exp(-gamma[0] * input_port_container[m][i]) * dLoss[m][i];
			}
		}
	}
	~GaussianRBF() {
		if (c_alloc) {
			delete[] gamma;
			delete[] gamma_grad;
		}
	}

};
class BatchNormalizer : public learnable {
public:
	int n_group;
	int n_elems;
	double divider;
	double* scale;
	double* shift;
	double* scale_grad;
	double* shift_grad;
	double* batch_mean;
	double* batch_variance;
	double* test_mean;
	double* test_variance;
	double* dLoss_mean;
	double* dLoss_var;
	double** norm_container;
	bool b_alloc;

	double exp_moving;
	BatchNormalizer() { c_alloc = false; b_alloc = false; }
	BatchNormalizer(int size) {
		create(1, size);
	}
	BatchNormalizer(shape _shape) {
		create(_shape.c, _shape.h*_shape.w);
	}
	void create(int group, int elements) {
		n_group = group;
		n_elems = elements;
		input_size = n_group * n_elems;
		output_size = input_size;
		divider = 1.0 / n_elems;
		scale = km_1d::alloc(n_group);
		shift = km_1d::alloc(n_group);
		scale_grad = km_1d::alloc(n_group);
		shift_grad = km_1d::alloc(n_group);
		batch_mean = km_1d::alloc(n_group);
		batch_variance = km_1d::alloc(n_group);
		test_mean = km_1d::alloc(n_group);
		test_variance = km_1d::alloc(n_group);
		km_1d::fill_zero(test_mean, n_group);
		km_1d::fill_zero(test_variance, n_group);

		dLoss_mean = km_1d::alloc(n_group);
		dLoss_var = km_1d::alloc(n_group);
		for (int i = 0; i < n_group; i++) {
			scale[i] = 1.0;
		}
		exp_moving = 0.95;
		km_1d::fill_zero(shift, n_group);
		in_port = km_1d::alloc(input_size);
		c_alloc = true;
		b_alloc = false;
	}
	virtual unit* clone() {
		BatchNormalizer* _clone = new BatchNormalizer();
		_clone->create(n_group, n_elems);
		for (int i = 0; i < n_group; i++) {
			_clone->scale[i] = scale[i];
			_clone->shift[i] = shift[i];
			_clone->test_mean[i] = test_mean[i];
			_clone->test_variance[i] = test_variance[i];
		}


		return _clone;
	}
	void alloc(int b_size) {
		if (b_alloc) {
			km_2d::free(norm_container, max_batch_size);
			km_2d::free(input_port_container, max_batch_size);
		}
		max_batch_size = b_size;
		using_size = max_batch_size;
		norm_container = km_2d::alloc(max_batch_size, input_size);
		downstream_container = km_2d::alloc(max_batch_size, input_size);
		input_port_container = km_2d::alloc(max_batch_size, input_size);
		divider = 1.0 / (max_batch_size*n_elems);
		input_port = input_port_container;
		downstream = downstream_container;
		b_alloc = true;
	}
	virtual void UseMemory(const int& _using_size) {
		assert(max_batch_size >= _using_size);
		using_size = _using_size;
		divider = 1.0 / (using_size*n_elems);
	}

	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		p_bag.push_back(scale);
		g_bag.push_back(scale_grad);
		len_bag.push_back(n_group);
		p_bag.push_back(shift);
		g_bag.push_back(shift_grad);
		len_bag.push_back(n_group);
	}
	
	void calculate(double* &next) {
		for (int g = 0; g < n_group; g++) {
			for (int i = 0; i < n_elems; i++) {
				next[g*n_elems + i] = scale[g] *((in_port[g*n_elems + i] - test_mean[g])/sqrt(test_variance[g])) + shift[g];
			}
		}
	}
	void forward(double** &next_port) {
		for (int g = 0; g < n_group; g++) {
			batch_mean[g] = 0.0;
			for (int m = 0; m < using_size; m++) {
				for (int i = 0; i < n_elems; i++) {
					batch_mean[g] += input_port_container[m][g*n_elems + i];
				}
			}
			batch_mean[g] = batch_mean[g] * divider;
			batch_variance[g] = 0.0;
			for (int m = 0; m < using_size; m++) {
				for (int i = 0; i < n_elems; i++) {
					batch_variance[g] += _square_(input_port_container[m][g*n_elems + i] - batch_mean[g]);
				}
			}
			batch_variance[g] = epsilon + batch_variance[g] * divider;
			test_mean[g] = exp_moving * test_mean[g] + (1.0 - exp_moving)*batch_mean[g];
			test_variance[g] = exp_moving * test_variance[g] + (1.0 - exp_moving)*batch_variance[g];
		}
		for (int m = 0; m < using_size; m++) {
			for (int g = 0; g < n_group; g++) {
				for (int i = 0; i < n_elems; i++) {
					norm_container[m][g*n_elems + i] = (input_port_container[m][g*n_elems + i] - batch_mean[g]) / sqrt(batch_variance[g]);
					next_port[m][g*n_elems + i] = scale[g] * norm_container[m][g*n_elems + i] + shift[g];
				}
			}
		}
	}
	void backward(double** const& dLoss) {
		for (int g = 0; g < n_group; g++) {
			dLoss_mean[g] = 0.0;
			dLoss_var[g] = 0.0;
		}
		for (int m = 0; m < using_size; m++) {
			for (int g = 0; g < n_group; g++) {
				for (int i = 0; i < n_elems; i++) {
					shift_grad[g] += dLoss[m][g*n_elems + i];
					scale_grad[g] += dLoss[m][g*n_elems + i] * norm_container[m][g*n_elems + i];
					downstream_container[m][g*n_elems + i] = dLoss[m][g*n_elems + i] * scale[g];
					dLoss_mean[g] += -pow(batch_variance[g], -0.5)*downstream_container[m][g*n_elems + i];
					dLoss_var[g] += (0.5)*(batch_mean[g] - input_port_container[m][g*n_elems + i])*pow(batch_variance[g], -1.5) * downstream_container[m][g*n_elems + i];
				}
			}
		}
		for (int m = 0; m < using_size; m++) {
			for (int g = 0; g < n_group; g++) {
				for (int i = 0; i < n_elems; i++) {
					downstream_container[m][g*n_elems + i] = 
						(downstream_container[m][g*n_elems + i]/sqrt(batch_variance[g])) 
						+ dLoss_var[g] 
						* (2.0 *divider)
						* (input_port_container[m][g*n_elems + i] - batch_mean[g]) 
						+ (dLoss_mean[g] * divider);
				}
			}
		}
	}
	~BatchNormalizer(){

		if (c_alloc) {
			delete[] scale;
			delete[] shift;
			delete[] scale_grad;
			delete[] shift_grad;
			delete[] batch_mean;
			delete[] batch_variance;
			delete[] test_mean;
			delete[] test_variance;
			delete[] dLoss_mean;
			delete[] dLoss_var;
		}
		if (b_alloc) {
			km_2d::free(norm_container, max_batch_size);
		}
	}
};
class N_Layer {
public:
	int n;
	N_Layer() {}
	N_Layer(int _n) { n = _n; }
};
class SkipBlock : public unit {
public:
	unit** main_stream;
	int n_unit;
	bool model_alloc;
	SkipBlock() { model_alloc = false; }
	SkipBlock(N_Layer n) {
		n_unit = n.n;
		main_stream = new unit*[n_unit];
		model_alloc = false;
	}
	~SkipBlock() {
		if (model_alloc) {
			for (int i = 0; i < n_unit; i++) {
				delete main_stream[i];
			}
			delete[] main_stream;
		}
	}

	virtual void UseMemory(const int& _using_size) {
		assert(max_batch_size >= _using_size);
		using_size = _using_size;
		for (int i = 0; i < n_unit; i++) {
			main_stream[i]->UseMemory(using_size);
		}
	}

	virtual void alloc(int b_size) {
		max_batch_size = b_size;
		for (int i = 0; i < n_unit; i++) {
			main_stream[i]->alloc(max_batch_size);
		}
		using_size = max_batch_size;
		publish();
	}

	void publish() {
		input_size = main_stream[0]->input_size;
		input_port = main_stream[0]->input_port;
		downstream = main_stream[0]->downstream;
		in_port = main_stream[0]->in_port;
		output_size = main_stream[n_unit - 1]->output_size;
		model_alloc = true;
	}

	virtual unit* clone() {
		SkipBlock* _clone = new SkipBlock(N_Layer(n_unit));
		for (int i = 0; i < n_unit; i++) {
			_clone->main_stream[i] = main_stream[i]->clone();
		} _clone->publish();
		return _clone;
	}

	void forward(double** &next_port) {
		for (int i = 0; i < n_unit - 1; i++) { /* main stream */
			main_stream[i]->forward(main_stream[i + 1]->input_port);
		}	main_stream[n_unit - 1]->forward(next_port);
		km_2d::add(next_port, input_port, using_size, output_size);
	}

	void backward(double** const& dLoss) {
		main_stream[n_unit - 1]->backward(dLoss);
		for (int i = n_unit - 2; i >= 0; i--) {
			main_stream[i]->backward(main_stream[i + 1]->downstream);
		}
		km_2d::add(downstream, dLoss, using_size, input_size);
	}

	void calculate(double* &next) {
		for (int i = 0; i < n_unit - 1; i++) { /* main stream */
			main_stream[i]->calculate(main_stream[i + 1]->in_port);
		}main_stream[n_unit - 1]->calculate(next);
		km_1d::add(next, in_port, output_size);
	}

	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		for (int i = 0; i < n_unit; i++) {
			main_stream[i]->delegate(p_bag, g_bag, len_bag);
		}
	}
};