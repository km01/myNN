#pragma once
#include "unit.h"

class DeulNet {
public:

	bundle* root;
	bundle* val_branch;
	bundle* adv_branch;
	double** in_port;
	double** out_port;
	int* argmax;
	double* in_put;
	int one_argmax;
	int batch_size;
	int max_batch_size;
	int input_size, encode_size, output_size;
	bool batch_allocated, published;
	double* root_out; double* adv_out; double* val_out; double* one_q;
	double* policy_probability;
	double** root_out_port; double** adv_out_port; double** val_out_port;
	double** rootLoss;
	double e, gamma;
	DeulNet() { batch_allocated = false; published = false; }
	DeulNet(n_Layer num_root_layer, n_Layer num_val_layer, n_Layer num_adv_layer) {
		root = new bundle(num_root_layer);
		val_branch = new bundle(num_val_layer);
		adv_branch = new bundle(num_adv_layer);
		batch_allocated = false; published = false;
	}

	void batch_predict(double** const& input) {
		root->charge(input);
		root->forward(root_out_port);
		km_2d::copy(adv_branch->in_port, root_out_port, batch_size, encode_size);
		km_2d::copy(val_branch->in_port, root_out_port, batch_size, encode_size);
		adv_branch->forward(adv_out_port);
		val_branch->forward(val_out_port);
	}
	int select(double* const& input) {
		one_predict(input);
		return km::argmax(adv_out, output_size);
	}
	int stochastic_select(double* const& state) {
		if (e < km::rand0to1(km::rEngine)) {
			return km::randint(output_size);
		}
		else {
			return select(state);
		}
	}
	void SetGamma(const double& _gamma) {
		gamma = _gamma;
	}
	void SetEpsilon(const double& _e) {
		e = _e;
	}
	void StartLearning() {
		UseMemory(max_batch_size);
	}
	void EndLearning() {
		UseMemory(1);
	}
	void one_predict(double* const& input) {
		root->set_input(input);
		root->calculate(root_out);
		km_1d::copy(adv_branch->in_put, root_out, encode_size);
		km_1d::copy(val_branch->in_put, root_out, encode_size);
		adv_branch->calculate(adv_out);
		val_branch->calculate(val_out); 
		double max_adv = km::max(adv_out, output_size);
		for (int a = 0; a < output_size; a++) {
			one_q[a] = val_out[0] + adv_out[a] - max_adv; /* Q(s,a) := V(s) + (A(s,a) */
		}
	}
	void backward(double** const& ValLoss, double** const& AdvLoss) {
		adv_branch->backward(AdvLoss);
		val_branch->backward(ValLoss);
		km_2d::add(rootLoss, adv_branch->downstream, val_branch->downstream, batch_size, encode_size);
		root->backward(rootLoss);
	}

	void max_evaluate(double*& virtual_Q, double** const& virtual_S, double* const& virtual_R, bool* const& virtual_T) {
		double value_from_q;
		for (int i = 0; i < output_size; i++) {
			if (virtual_T[i] == TERMINAL) {
				virtual_Q[i] = virtual_R[i];
			}
			else {
				one_predict(virtual_S[i]);
				virtual_Q[i] = virtual_R[i] + gamma * km::max(one_q, output_size);
			}
		}
	}

	void publish() {
		root->publish();
		adv_branch->publish();
		val_branch->publish();
		input_size = root->input_size;
		output_size = adv_branch->output_size;
		encode_size = root->output_size;
		in_put = root->in_put;
		root_out = km_1d::alloc(encode_size);
		adv_out = km_1d::alloc(output_size);
		val_out = km_1d::alloc(1);
		one_q = km_1d::alloc(output_size);
		policy_probability = km_1d::alloc(output_size);
		
		published = true;
	}
	~DeulNet() {
		if (published) {
			delete root;
			delete adv_branch;
			delete val_branch;
			delete root_out;
			delete adv_out;
			delete val_out;
			delete one_q;
		}
		if (batch_allocated) {
			delete[] argmax;
			km_2d::free(out_port, max_batch_size);
			km_2d::free(root_out_port, max_batch_size);
			km_2d::free(adv_out_port, max_batch_size);
			km_2d::free(val_out_port, max_batch_size);
			km_2d::free(rootLoss, max_batch_size);
		}
	}
	double max_Loss_fn(double**& dLoss_val, double**& dLoss_adv, double** const& target) {
		double mean_loss = 0.0;
		double target_v = 0.0;
		for (int m = 0; m < batch_size; m++) {
			target_v = km::max(target[m], output_size);
			dLoss_val[m][0] = 0.0;
			for (int i = 0; i < output_size; i++) {

				dLoss_adv[m][i] = 2.0 * ((adv_out_port[m][i]) - (target[m][i] - target_v));
				dLoss_val[m][0] += 2.0 * (val_out_port[m][0] - (target_v));
				mean_loss += ((adv_out_port[m][i]) - (target[m][i] - target_v)) * ((adv_out_port[m][i]) - (target[m][i] - target_v));
			}
		}
		return mean_loss / batch_size;
	}
	void AllocMemory(int _max_batch_size) {
		if (batch_allocated) {
			delete[] argmax;
			km_2d::free(out_port, max_batch_size);
			km_2d::free(root_out_port, max_batch_size);
			km_2d::free(adv_out_port, max_batch_size);
			km_2d::free(val_out_port, max_batch_size);
			km_2d::free(rootLoss, max_batch_size);
		}

		max_batch_size = _max_batch_size;
		batch_size = max_batch_size;
		root->AllocMemory(max_batch_size);
		adv_branch->AllocMemory(max_batch_size);
		val_branch->AllocMemory(max_batch_size);

		in_port = root->in_port;
		out_port = km_2d::alloc(max_batch_size, output_size);
		root_out_port = km_2d::alloc(max_batch_size, encode_size);
		adv_out_port = km_2d::alloc(max_batch_size, output_size);
		val_out_port = km_2d::alloc(max_batch_size, 1);
		rootLoss = km_2d::alloc(max_batch_size, encode_size);
		batch_allocated = true;
	}

	void UseMemory(const int& _using_size) {
		batch_size = _using_size;
		root->UseMemory(batch_size);
		val_branch->UseMemory(batch_size);
		adv_branch->UseMemory(batch_size);
	}
};