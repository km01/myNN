#pragma once
#include "core.h"

class ReplayMemory {
public:
	int size;

	int curr_block;
	int curr_data;
	int* OrderList;
	double** state;
	double* td_target;
	int* action;
	double* prob;
	int state_len;
	int n_env;
	int sample_size;
	int batch_size;
	double** batch_s;
	double* batch_target;
	double* batch_prob;
	int* batch_a;
	bool allocated;
	ReplayMemory() { allocated = false; }
	ReplayMemory(int _state_len, int _n_env, int _sample_size, int _batch_size) {
		state_len = _state_len;
		n_env = _n_env;
		sample_size = _sample_size;
		batch_size = _batch_size;
		size = n_env * sample_size;
		state = km_2d::alloc(size, state_len);
		td_target = km_1d::alloc(size);
		prob = km_1d::alloc(size);
		action = new int[size];
		OrderList = new int[size];
		for (int i = 0; i < size; i++) {
			OrderList[i] = i;
		}
		batch_s = km_2d::alloc(batch_size, state_len);
		batch_target = km_1d::alloc(batch_size);
		batch_prob = km_1d::alloc(batch_size);
		batch_a = new int[batch_size];
		allocated = true;
		clear();
	}


	void FetchBatchData() {
		for (int m = 0; m < batch_size; m++) {
			km_1d::copy(batch_s[m], state[OrderList[curr_data]], state_len);
			batch_target[m] = td_target[OrderList[curr_data]];
			batch_prob[m] = prob[OrderList[curr_data]];
			batch_a[m] = action[OrderList[curr_data]];
			curr_data++;
		}
	}

	void push(double** const& _state, int* const& _action, double* const& _td_target, double* const& _prob) {
		int idx = 0;
		for (int m = 0; m < sample_size; m++) {
			idx = curr_block * sample_size + m;
			km_1d::copy(state[idx], _state[m], state_len);
			action[idx] = _action[m];
			td_target[idx] = _td_target[m];
			prob[idx] = _prob[m];
		}
		curr_block++;
	}


	void clear() {
		curr_block = 0;
		curr_data = 0;
	}
	void shuffle() {
		km::shuffle(OrderList, size);
	}
	~ReplayMemory() {
		if (allocated) {
			km_2d::free(state, size);
			km_2d::free(batch_s, batch_size);
			km_1d::free(td_target);
			km_1d::free(batch_target);
			km_1d::free(prob);
			km_1d::free(batch_prob);
			delete[] action;
			delete[] batch_a;
			delete[] OrderList;
		}
	}
};