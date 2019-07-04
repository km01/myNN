#pragma once
#include "core.h"

class unit {
public:
	
	int input_size;
	int output_size;
	int max_batch_size;
	int using_size;

	double** input_port;
	double** downstream;


	unit() {}

	virtual void alloc(int b_size) = 0; // 배치사이즈만큼 메모리를 잡아줍니다
	virtual void set_input(double** const& input) {
		for (int i = 0; i < using_size; i++) {
			for (int j = 0; j < input_size; j++) {
				input_port[i][j] = input[i][j];
			}
		}
	}
	virtual void UseMemory(const int& _using_size) {
		assert(max_batch_size>=_using_size);
		using_size = _using_size;
	}
	~unit() {

	}
	virtual unit* clone() = 0;
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
	}
};
enum ACTIVATION {SIGMOID, TANH, RELU};
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
	}
};

activation* define(ACTIVATION fn, int size);
class sigmoid : public activation{
public:
	sigmoid() { c_alloc = false; fn = SIGMOID; }
	sigmoid(int size) { create(size); fn = SIGMOID; }
	virtual unit* clone() {
		return new sigmoid(input_size);
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
	ReLU() { c_alloc = false; fn = RELU;}
	ReLU(int size) { create(size); fn = RELU; }
	virtual unit* clone() {
		return new ReLU(input_size);
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
		struct { int d1, d2, d3;  };
	};
	shape(){}
	shape(const int& _c, const int& _h, const int& _w):c(_c),h(_h),w(_w){}
	void copy(const shape& src) {
		c = src.c; h = src.h; w = src.w;
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
	kernel3D(shape in, shape k_shape, shape out, int h_stride, int w_stride){
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
		km_2d::guassian_noise(kernel, 0.0, (2.0 / input_size),out.c, k_len);
		w_alloc = true;
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
	}


	void forward(double** &next_port) {
		km_2d::fill_zero(next_port, using_size, output_size);
		for (int m = 0; m < using_size; m++) {
			for (int out_c = 0; out_c < out_dim.c; out_c++) {
				for (int k = 0; k < out_plain_size; k++) {
					next_port[m][(out_c * out_plain_size) + k ] = bias[out_c];
				}
				for (int in_c = 0; in_c < in_dim.c; in_c++) {
					for (int in_h = 0, out_h = 0; out_h < out_dim.h; in_h += h_strd, out_h++) {
						for (int in_w = 0, out_w = 0; out_w < out_dim.w; in_w += w_strd, out_w++) {
							for (int kh = 0; kh < kernel_dim.h; kh++) {
								for (int kw = 0; kw < kernel_dim.w; kw++) { //out[out_c][out_h][out_w] += kernel[out_c][in_c][kernel_h][kernel_w] * in[in_c][in_h + kernel_h][in_w + kernel_w]
									next_port[m][(out_c * out_plain_size)+ (out_h*out_dim.w) + (out_w)] += kernel[out_c][(in_c*kernel_plain_size) + (kh * kernel_dim.w) + (kw)]
										* input_port[m][(in_c * (in_plain_size))+ (in_dim.w *(in_h + kh)) + (in_w + kw)];
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
					for (int k = 0; k < out_plain_size; k++) {
						b_grad[out_c] += dLoss[m][(out_c * out_plain_size) + k];
					}
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