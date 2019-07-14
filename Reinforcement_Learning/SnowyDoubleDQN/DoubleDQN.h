#pragma once

#include "nn.h"
class ReplayMemory {
public:
	int size;
	int head;
	int tail;
	int* OrderList;
	bool allocated;
	double** state;
	double** Qvalue;
	int state_len;
	int qeval_len;
	ReplayMemory() { allocated = false; }
	ReplayMemory(int _state_len, int _qval_len) {
		state_len = _state_len;
		qeval_len = _qval_len;
		allocated = false;
	}
	void allocate(int _size) {
		size = _size;
		state = km_2d::alloc(size, state_len);
		Qvalue = km_2d::alloc(size, qeval_len);
		OrderList = new int[size];
		allocated = true;
		for (int i = 0; i < size; i++) {
			OrderList[i] = i;
		}
		InitSession();
	}

	void pop(double*& state_container, double*& target_container) {
		km_1d::copy(state_container, state[OrderList[head]], state_len);
		km_1d::copy(target_container, Qvalue[OrderList[head]], qeval_len);
		head++;
	}
	void push(double* const& _state, double* const& _Qeval) {

		km_1d::copy(state[tail], _state, state_len);
		km_1d::copy(Qvalue[tail], _Qeval, qeval_len);
		tail++;
	}
	bool IsFull() {
		if (tail >= size) {
			return true;
		}
		else {
			return false;
		}
	}
	void fetch(double**& batch_x, double**& batch_y, const int& batch_size) {
		for (int i = 0; i < batch_size; i++) {
			pop(batch_x[i], batch_y[i]);
		}
	}
	void InitSession() {
		km::shuffle(OrderList, size);
		head = 0;
		tail = 0;
	}
	~ReplayMemory() {
		if (allocated) {
			km_2d::free(state, size);
			km_2d::free(Qvalue, size);
			delete[] OrderList;
		}
	}

};
class DDQnet {
public:
	nn* DQNNetwork; // selecter
	nn* TargetNetwork; // evaluater
	double e, gamma;
	int input_size;
	int output_size;
	DDQnet(nn* dqnet, nn* targetnet, const double& _gamma) {
		DQNNetwork = dqnet;
		TargetNetwork = targetnet;
		input_size = DQNNetwork->input_size;
		output_size = DQNNetwork->output_size;
		gamma = _gamma;
	}

	void SetEpsilon(const double& _e) {
		e = _e;
	}

	int stochastic_select(double* const&state) {
		if (e < km::rand0to1(km::rEngine)) {
			return km::randint(output_size);
		}
		else {
			return select(state);
		}
	}

	void StartLearning() {
		DQNNetwork->UseMemory(DQNNetwork->max_batch_size);
	}

	void EndLearning() {
		DQNNetwork->UseMemory(1);
	}

	void predict(double** const& batch_x) {
		DQNNetwork->set_input(batch_x);
		DQNNetwork->forward(DQNNetwork->output_port);
	}

	void backward(double** const& batch_dLoss) {
		DQNNetwork->backward(batch_dLoss);
	}

	int select(double* const& input) {
		DQNNetwork->charge(input);
		DQNNetwork->calculate(DQNNetwork->out_port);
		return km::argmax(DQNNetwork->out_port, DQNNetwork->output_size);
	}

	void evaluate(double*& virtual_Q, double** const& virtual_S, double* const& virtual_R, bool* const& virtual_T) {
		for (int i = 0; i < output_size; i++) {
			if (virtual_T[i] == TERMINAL) {
				virtual_Q[i] = virtual_R[i];
			}
			else {
				TargetNetwork->charge(virtual_S[i]);
				TargetNetwork->calculate(TargetNetwork->out_port);
				virtual_Q[i] = virtual_R[i] + gamma * TargetNetwork->out_port[select(virtual_S[i])];
			}
		}
	}
};