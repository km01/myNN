#pragma once
#include "Unit.h"

class MaskedDense : public Unit {
public:
	Darr weight = nullptr;
	Darr bias = nullptr;
	Darr w_grad = nullptr;
	Darr b_grad = nullptr;
	bool* mask = nullptr;
	ReLU* relu = nullptr;
	MaskedDense() {}
	MaskedDense(const int& n_input, const int& n_output, const bool& addReLU) {
		create(n_input, n_output, addReLU);
	}

	void create(const int& n_input, const int& n_output, const bool& addReLU) {
		setIOsize(n_input, n_output);
		weight = alloc(n_input * n_output);
		w_grad = alloc(n_input * n_output);
		bias = alloc(n_output);
		b_grad = alloc(n_output);
		grad_port = alloc(n_input);
		mask = new bool[n_in * n_out];
		setNormal(weight, n_in * n_out, 0.0, 2.0 / (double(n_in) + n_out));
		setZero(bias, n_out);
		if (addReLU) {
			relu = new ReLU(n_out);
		}
	}

	void setMask(int* input_order, int* output_order, bool isLastLayer) {
		if (!isLastLayer) {
			for (int i = 0; i < n_in; i++) {
				for (int j = 0; j < n_out; j++) {
					if (input_order[i] <= output_order[j]) {
						mask[i * n_out + j] = true;
					}
					else {
						mask[i * n_out + j] = false;
					}
				}
			}
		}
		else {
			for (int i = 0; i < n_in; i++) {
				for (int j = 0; j < n_out; j++) {
					if (input_order[i] < output_order[j]) {
						mask[i * n_out + j] = true;
					}
					else {
						mask[i * n_out + j] = false;
					}
				}
			}
		}
	}

	virtual void forward(Darr& next_input, const int& id) {
		if (relu == nullptr) {
			for (int o = 0; o < n_out; o++) {
				next_input[o] = bias[o];
			}
			for (int i = 0; i < n_in; i++) {
				for (int o = 0; o < n_out; o++) {
					if (mask[i * n_out + o]) {
						next_input[o] += weight[i * n_out + o] * in_port[id][i];
					}
				}
			}
		}
		else { // if relu
			for (int o = 0; o < n_out; o++) {
				relu->in_port[id][o] = bias[o];
			}
			for (int i = 0; i < n_in; i++) {
				for (int o = 0; o < n_out; o++) {
					if (mask[i * n_out + o]) {
						relu->in_port[id][o] += weight[i * n_out + o] * in_port[id][i];
					}
				}
			}
			relu->forward(next_input, id);
		}
	}

	virtual void backward(Darr& output_grad, const int& id) {
		if (relu == nullptr) {
			for (int o = 0; o < n_out; o++) {
				b_grad[o] += output_grad[o];
			}
			for (int i = 0; i < n_in; i++) {
				grad_port[i] = 0.0;
				for (int o = 0; o < n_out; o++) {
					if (mask[i * n_out + o]) {
						w_grad[i * n_out + o] += output_grad[o] * in_port[id][i];
						grad_port[i] += weight[i * n_out + o] * output_grad[o];
					}
				}
			}
		}
		else { // if relu
			relu->backward(output_grad, id);
			for (int o = 0; o < n_out; o++) {
				b_grad[o] += relu->grad_port[o];
			}
			for (int i = 0; i < n_in; i++) {
				grad_port[i] = 0.0;
				for (int o = 0; o < n_out; o++) {
					if (mask[i * n_out + o]) {
						w_grad[i * n_out + o] += relu->grad_port[o] * in_port[id][i];
						grad_port[i] += weight[i * n_out + o] * relu->grad_port[o];
					}
				}
			}
		}
	}

	virtual void setCache(const int& cache_size) {
		free(in_port, this->cache_size);
		this->cache_size = cache_size;
		in_port = alloc(this->cache_size, this->n_in);
		if (relu != nullptr) {
			relu->setCache(cache_size);
		}
	}

	virtual void takeParams(vector<Darr>& param_bag, vector<Darr>& grad_bag, vector<int>& len_bag) {
		param_bag.push_back(weight);
		grad_bag.push_back(w_grad);
		len_bag.push_back(n_in * n_out);
		param_bag.push_back(bias);
		grad_bag.push_back(b_grad);
		len_bag.push_back(n_out);
	}

	void show_mask() {
		for (int i = 0; i < n_in; i++) {
			for (int o = 0; o < n_out; o++) {
				cout << mask[i * n_out + o] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	~MaskedDense() {
		free(weight);
		free(w_grad);
		free(bias);
		free(b_grad);
		free(grad_port);
		free(in_port, this->cache_size);
		if (mask != nullptr) {
			delete[] mask;
		}
		if (relu != nullptr) {
			delete relu;
		}
	}
};

void shuffle(int* arr, const int& len) {
	for (int r = 0; r < 5; r++) {
		for (int i = 0; i < len; i++) {
			int idx = rand() % len;
			int val = arr[i];
			arr[i] = arr[idx];
			arr[idx] = val;
		}
	}
}

int* assign_first(const int& n_in) {
	int* list = new int[n_in];
	for (int i = 0; i < n_in; i++) {
		list[i] = i;
	}
	shuffle(list, n_in);
	return list;
}

int* assign_not_first(const int& len, const int& n_in) {
	int* list = new int[len];
	for (int i = 0; i < n_in - 1; i++) {
		list[i] = i;
	}
	for (int i = n_in - 1; i < len; i++) {
		list[i] = (rand() % (n_in - 1));
	}
	return list;

}

class AutoRegressiveTransformer : public Unit {
public:
	vector<MaskedDense*> masked_autoencoder;
	int n_block = 0;
	int last_idx = -1;
	int** dependency_order = nullptr;
	AutoRegressiveTransformer() { masked_autoencoder.clear(); }
	void addLayer(const int& layer_n_in, const int& layer_n_out, const bool& addReLU, const bool& isLast) {
		masked_autoencoder.push_back(new MaskedDense(layer_n_in, layer_n_out, addReLU));
		n_block++;
		last_idx = n_block - 1;
		if (n_block == 1) {
			grad_port = masked_autoencoder[0]->grad_port;
			n_in = masked_autoencoder[0]->n_in;
		}
		if (isLast) {
			n_out = n_in;
		}
	}

	virtual void setMask(int* dependency) {
		dependency_order = new int* [n_block];
		dependency_order[0] = new int[n_in];
		for (int i = 0; i < n_in; i++) {
			dependency_order[0][i] = dependency[i];
		}
		for (int i = 1; i < n_block; i++) {
			dependency_order[i] = assign_not_first(masked_autoencoder[i]->n_in, n_in);
			masked_autoencoder[i - 1]->setMask(dependency_order[i - 1], dependency_order[i], false);
		}
		masked_autoencoder[n_block - 1]->setMask(dependency_order[n_block - 1], dependency_order[0], true);
	}

	virtual void setCache(const int& cache_size) {
		for (int i = 0; i < n_block; i++) {
			masked_autoencoder[i]->setCache(cache_size);
		}
		in_port = masked_autoencoder[0]->in_port;
	}

	virtual void forward(Darr& next_input, const int& id) {
		for (int i = 0; i < last_idx; i++) {
			masked_autoencoder[i]->forward(masked_autoencoder[i + 1]->in_port[id], id);
		}
		masked_autoencoder[last_idx]->forward(next_input, id);
	}

	virtual void backward(Darr& output_grad, const int& id) {
		masked_autoencoder[last_idx]->backward(output_grad, id);
		for (int i = last_idx - 1; i >= 0; i--) {
			masked_autoencoder[i]->backward(masked_autoencoder[i + 1]->grad_port, id);
		}
	}

	virtual void takeParams(vector<Darr>& param_bag, vector<Darr>& grad_bag, vector<int>& len_bag) {
		for (int i = 0; i < n_block; i++) {
			masked_autoencoder[i]->takeParams(param_bag, grad_bag, len_bag);
		}
	}
};

class GaussianAutoRegressiveBlock : public Unit {
public:
	AutoRegressiveTransformer mean_net;
	AutoRegressiveTransformer log_stddev_net;
	DarrList mean_port = nullptr;
	DarrList log_stddev_port = nullptr;
	DarrList stddev_port = nullptr;
	int* dependency_order = nullptr;
	Darr local_cache = nullptr;
	Darr determinant = nullptr;
	GaussianAutoRegressiveBlock() {}
	void addLayer(const int& layer_n_in, const int& layer_n_out, const bool& addReLU, const bool& isLast) {
		mean_net.addLayer(layer_n_in, layer_n_out, addReLU, isLast);
		log_stddev_net.addLayer(layer_n_in, layer_n_out, addReLU, isLast);
		n_in = mean_net.n_in;
		n_out = mean_net.n_out;
	}

	virtual void setMask() {
		dependency_order = assign_first(n_in);
		mean_net.setMask(dependency_order);
		log_stddev_net.setMask(dependency_order);
	}

	virtual void setCache(const int& cache_size) {
		mean_net.setCache(cache_size);
		log_stddev_net.setCache(cache_size);
		mean_port = alloc(cache_size, n_in);
		log_stddev_port = alloc(cache_size, n_in);
		stddev_port = alloc(cache_size, n_in);
		in_port = mean_net.in_port;
		local_cache = alloc(n_in);
		grad_port = alloc(n_in);
		determinant = alloc(cache_size);
	}

	virtual void forward(Darr& next_input, const int& id) {
		log_stddev_net.charge(in_port[id], id);
		mean_net.forward(mean_port[id], id);
		log_stddev_net.forward(log_stddev_port[id], id);
		determinant[id] = 1.0;
		for (int i = 0; i < n_in; i++) {
			if (log_stddev_port[id][i] > 0) {
				stddev_port[id][i] = log_stddev_port[id][i] + 1.0;
			}
			else {
				stddev_port[id][i] = exp(log_stddev_port[id][i]);
			}
			determinant[id] *= stddev_port[id][i];
			next_input[i] = stddev_port[id][i] * in_port[id][i] + mean_port[id][i];
		}
	}

	virtual void backward(Darr& output_grad, const int& id) {
		for (int i = 0; i < n_in; i++) {
			if (log_stddev_port[id][i] > 0) {
				local_cache[i] = in_port[id][i] * output_grad[i] - 1.0/(stddev_port[id][i]);
			}
			else {
				local_cache[i] = in_port[id][i] * stddev_port[id][i] * output_grad[i] - 1.0;
			}
		}

		log_stddev_net.backward(local_cache, id);
		mean_net.backward(output_grad, id);
		for (int i = 0; i < n_in; i++) {
			grad_port[i] = mean_net.grad_port[i] + log_stddev_net.grad_port[i] + output_grad[i] * stddev_port[id][i];
		}
	}

	virtual void takeParams(vector<Darr>& param_bag, vector<Darr>& grad_bag, vector<int>& len_bag) {
		mean_net.takeParams(param_bag, grad_bag, len_bag);
		log_stddev_net.takeParams(param_bag, grad_bag, len_bag);
	}
};

class MADE : public Unit {
public:
	vector<GaussianAutoRegressiveBlock*> layer;
	int n_block = 0;
	int last_idx = -1;
	Darr out = nullptr;
	Darr grad = nullptr;

	void addBlock(GaussianAutoRegressiveBlock* block) {
		layer.push_back(block);
		n_block++;
		last_idx = n_block - 1;
		if (n_block == 1) {
			grad_port = layer[0]->grad_port;
			n_in = layer[0]->n_in;
			n_out = n_in;
			out = alloc(n_in);
			grad = alloc(n_in);

		}
	}

	virtual void setMask() {
		for (int i = 0; i < n_block; i++) {
			layer[i]->setMask();
		}
	}

	virtual void setCache(const int& cache_size) {
		for (int i = 0; i < n_block; i++) {
			layer[i]->setCache(cache_size);
		}
	}

	virtual double fit(Darr& x, const int& id) {
		double likelihood = 1.0;
		layer[0]->charge(x, id);
		for (int i = 0; i < last_idx; i++) {
			layer[i]->forward(layer[i + 1]->in_port[id], id);
			likelihood *= layer[i]->determinant[id];
		}
		layer[last_idx]->forward(out, id);
		
		for (int d = 0; d < n_in; d++) {
			likelihood *= pow(2.0 * 3.141592, -0.5) * exp(-0.5* out[d] *out[d]);
		}

		layer[last_idx]->backward(out, id);
		for (int i = last_idx - 1; i >= 0; i--) {
			layer[i]->backward(layer[i + 1]->grad_port, id);
		}
		return likelihood;
	}

	virtual void forward(Darr& next_input, const int& id) {

	}

	virtual void backward(Darr& output_grad, const int& id) {

	}

	virtual void takeParams(vector<Darr>& param_bag, vector<Darr>& grad_bag, vector<int>& len_bag) {
		for (int i = 0; i < n_block; i++) {
			layer[i]->takeParams(param_bag, grad_bag, len_bag);
		}
	}
};