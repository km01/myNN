#pragma once
#include "Unit.h"
class GaussianDensityNetwork : public Unit {
public:

	vector<Unit*> units;
	int n_units = 0;
	int random_vector_size = 0;
	darr::v net_output_grad = nullptr;
	darr::v2D net_output_cache = nullptr;


	GaussianDensityNetwork() { units.clear(); }
	~GaussianDensityNetwork() {}
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
		random_vector_size = n_out / 2;
		n_units += 1;
	}

	void forward(darr::v& output, const int& id) {	assert(false);	}
	void backward(darr::v& output_grad, const int& id) {	assert(false);	}

	void gdn_forward(darr::v& mean, darr::v& stddev, const int& id) {
		for (int i = 0; i < n_units - 1; i++) {
			units[i]->forward(units[i + 1]->in_cache[id], id);
		}
		units[n_units - 1]->forward(net_output_cache[id], id);
		for (int i = 0; i < random_vector_size; i++) {
			mean[i] = net_output_cache[id][i];
			if (net_output_cache[id][random_vector_size + i] > 0.0) {
				stddev[i] = net_output_cache[id][random_vector_size + i] + 1.0;
			}
			else {
				stddev[i] = exp(net_output_cache[id][random_vector_size + i]);
			}
		}
	}

	void gdn_backward(darr::v& mean_grad, darr::v& stddev_grad, const int& id) {
		for (int i = 0; i < random_vector_size; i++) {
			net_output_grad[i] = mean_grad[i];
			if (net_output_cache[id][random_vector_size + i] > 0.0) {
				net_output_grad[random_vector_size + i] = stddev_grad[i];
			}
			else {
				net_output_grad[random_vector_size + i] = stddev_grad[i] * exp(net_output_cache[id][random_vector_size + i]);
			}
		}
		units[n_units - 1]->backward(net_output_grad, id);
		for (int i = n_units - 2; i >= 0; i--) {
			units[i]->backward(units[i + 1]->in_grad_cache[id], id);
		}
	}

	virtual void setCache(const int& cache_size) {
		darr::resize(net_output_grad, n_out);
		darr::resize(net_output_cache, this->n_cache, cache_size, n_out);

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

class ConditionalGaussianDensityNetwork : public Unit {
public:

	vector<Unit*> units;
	int n_units = 0;
	int random_vector_size = 0;
	darr::v net_output_grad = nullptr;
	darr::v2D net_output_cache = nullptr;
	int condition_vector_size = 0;
	ConditionalGaussianDensityNetwork(const int& condition_vector_size) { 
		units.clear(); 
		this->condition_vector_size = condition_vector_size;
	}
	~ConditionalGaussianDensityNetwork() {}

	virtual void cgdn_charge(darr::v const& input, const int& condition, const int& id) {
		for (int i = 0; i < n_in - condition_vector_size; i++) { 
			in_cache[id][i] = input[i]; 
		}
		for (int i = n_in - condition_vector_size; i < n_in; i++) {
			in_cache[id][i] = 0.0;
		}
		in_cache[id][n_in - condition_vector_size + condition] = 1.0;
	}


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
		random_vector_size = n_out / 2;
		n_units += 1;
	}

	void forward(darr::v& output, const int& id) { assert(false); }
	void backward(darr::v& output_grad, const int& id) { assert(false); }

	void cgdn_forward(darr::v& mean, darr::v& stddev, const int& id) {
		for (int i = 0; i < n_units - 1; i++) {
			units[i]->forward(units[i + 1]->in_cache[id], id);
		}
		units[n_units - 1]->forward(net_output_cache[id], id);
		for (int i = 0; i < random_vector_size; i++) {
			mean[i] = net_output_cache[id][i];
			if (net_output_cache[id][random_vector_size + i] > 0.0) {
				stddev[i] = net_output_cache[id][random_vector_size + i] + 1.0;
			}
			else {
				stddev[i] = exp(net_output_cache[id][random_vector_size + i]);
			}
		}
	}

	void cgdn_backward(darr::v& mean_grad, darr::v& stddev_grad, const int& id) {
		for (int i = 0; i < random_vector_size; i++) {
			net_output_grad[i] = mean_grad[i];
			if (net_output_cache[id][random_vector_size + i] > 0.0) {
				net_output_grad[random_vector_size + i] = stddev_grad[i];
			}
			else {
				net_output_grad[random_vector_size + i] = stddev_grad[i] * exp(net_output_cache[id][random_vector_size + i]);
			}
		}
		units[n_units - 1]->backward(net_output_grad, id);
		for (int i = n_units - 2; i >= 0; i--) {
			units[i]->backward(units[i + 1]->in_grad_cache[id], id);
		}
	}

	virtual void setCache(const int& cache_size) {
		darr::resize(net_output_grad, n_out);
		darr::resize(net_output_cache, this->n_cache, cache_size, n_out);

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
