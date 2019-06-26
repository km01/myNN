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

	virtual void alloc(int b_size) = 0;
	virtual void set_input(double** const& input) {
		for (int i = 0; i < using_size; i++) {
			for (int j = 0; j < input_size; j++) {
				input_port[i][j] = input[i][j];
			}
		}
	}
	~unit() {

	}

	virtual void forward(double** &next_port) = 0;
	virtual void backward(double** const& dLoss) = 0;
	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {}
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
	void forward(double** &next_port) {
		for (int m = 0; m < using_size; m++) {
			for (int i = 0; i < input_size; i++) {
				if (input_port_container[m][i] >= 0) {
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
class FcBlock : public bundle {
public:
	
	fully_connected* fc;
	activation* function;
	FcBlock() {content_alloc = false;}
	FcBlock(int input_sz, int output_sz, ACTIVATION fn) {
		create(input_sz, output_sz, fn);
	}
	void create(int input_sz, int output_sz, ACTIVATION fn) {
		input_size = input_sz;
		output_size = output_sz;
		fc = new fully_connected(input_size, output_size);
		function = define(fn, output_size);
		input_port = fc->input_port;
		downstream = fc->downstream;
		content_alloc = true;
	}

	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		fc->delegate(p_bag, g_bag, len_bag);
		function->delegate(p_bag, g_bag, len_bag);
	}
	virtual void alloc(int b_size) {
		max_batch_size = b_size;
		fc->alloc(max_batch_size);
		function->alloc(max_batch_size);
		input_port = fc->input_port;
		downstream = fc->downstream;
		using_size = max_batch_size;
	}

	void forward(double** &next_port) {
		fc->forward(function->input_port);
		function->forward(next_port);
	}
	void backward(double** const& dLoss) {
		function->backward(dLoss);
		fc->backward(function->downstream);
	}
	~FcBlock() {
		if (content_alloc) {
			delete fc;
			delete function;
		}
	}
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
class padder : public unit {
public:
	int east, west;
	int north, south;
	shape in_dim;
	shape out_dim;
	int in_plain_size, out_plain_size;
	bool c_alloc;
	double** input_port_container;
	double** downstream_container;
	padder(shape in, shape out) {
		in_dim.copy(in);
		out_dim.copy(out);
		input_size = in_dim.c*in_dim.h*in_dim.w;
		output_size = out_dim.c*out_dim.h*out_dim.w;
		in_plain_size = in_dim.w* in_dim.h;
		out_plain_size = out_dim.w* out_dim.h;
		if ((out_dim.h - in_dim.h) % 2 == 1) {
			north = (out_dim.h - in_dim.h) / 2;
			south = (out_dim.h - north)-1;
		}
		else {
			north = (out_dim.h - in_dim.h) / 2;
			south = out_dim.h - north;
		}
		if ((out_dim.w - in_dim.w) % 2 == 1) {
			west = (out_dim.w - in_dim.w) / 2;
			east = (out_dim.w - west) - 1;
		}
		else {
			west = (out_dim.w - in_dim.w) / 2;
			east = out_dim.w - west;
		}
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
		downstream_container = downstream;
		using_size = max_batch_size;
		c_alloc = true;
	}

	void forward(double** &next_port) {
		km_2d::fill_zero(next_port, using_size, output_size); // margin filling. 
		for (int m = 0; m < using_size; m++) {
			for (int c = 0; c < in_dim.c; c++) {
				for (int h = 0; h < in_dim.h; h++) {
					for (int w = 0; w < in_dim.w; w++) {
						next_port[m][(c*out_plain_size) + ((h + north)*out_dim.w) + (w + west)] = input_port_container[m][(c*in_plain_size) + (h * in_dim.w) + w];
					}	
				}
			}

		}
	}

	void backward(double** const& dLoss) {
		km_2d::fill_zero(downstream_container, using_size, input_size); // margin discarding. 
		for (int m = 0; m < using_size; m++) {
			for (int c = 0; c < in_dim.c; c++) {
				for (int h = 0; h < in_dim.h; h++) {
					for (int w = 0; w < in_dim.w; w++) {
						downstream_container[m][(c*in_plain_size) + (h * in_dim.w) + w] = dLoss[m][(c*out_plain_size) + ((h + north)*out_dim.w) + (w + west)];
					}
				}
			}

		}
	}
	~padder() {
		if (c_alloc) {
			km_2d::free(input_port_container, max_batch_size);
			km_2d::free(downstream_container, max_batch_size);
		}
	}
};

class ConvBlock : public bundle {
public:
	kernel3D* convolution;
	activation* function;
	ConvBlock() { content_alloc = false; }

	ConvBlock(shape in, shape k_shape, shape out, int h_stride, int w_stride, ACTIVATION fn) {
		create(in, k_shape, out,h_stride, w_stride, fn);
	}
	void create(shape in, shape k_shape, shape out, int h_stride, int w_stride, ACTIVATION fn) {

		convolution = new kernel3D(in, k_shape, out, h_stride, w_stride);
		function = define(fn, convolution->output_size);
		content_alloc = false;
		input_size = convolution->input_size;
		output_size = function->output_size;
	}

	virtual void delegate(vector<double*>& p_bag, vector<double*>& g_bag, vector<int>& len_bag) {
		convolution->delegate(p_bag, g_bag, len_bag);
		function->delegate(p_bag, g_bag, len_bag);
	}

	virtual void alloc(int b_size) {

		max_batch_size = b_size;
		convolution->alloc(max_batch_size);
		function->alloc(max_batch_size);

		input_port = convolution->input_port;
		downstream = convolution->downstream;
		using_size = max_batch_size;
	}

	void forward(double** &next_port) {

		convolution->forward(function->input_port);
		function->forward(next_port);
	}

	void backward(double** const& dLoss) {
		function->backward(dLoss);
		convolution->backward(function->downstream);
	}

	~ConvBlock() {
		if (content_alloc) {
			delete convolution;
			delete function;
		}
	}
};