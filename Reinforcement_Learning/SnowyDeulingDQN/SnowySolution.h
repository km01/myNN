#pragma once
#include "SnowyGraphicLauncher.h"
#include "DeulNet.h"
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
class SnowySolution {
public:
	DeulNet* agent;
	optimizer* root_optim;
	optimizer* val_optim;
	optimizer* adv_optim;
	ReplayMemory* memory;
	Snowy::envirnoment* world;
	int batch_size;
	int memory_size;
	int n_batch;
	bool allocated;
	double** batch_x;
	double** batch_y;
	double** batch_adv_dLoss;
	double** batch_val_dLoss;
	double** virtual_S;
	double* virtual_R;
	bool* virtual_T;
	double* virtual_Q;

	SnowySolution(Snowy::envirnoment* snowy, DeulNet* _net, optimizer* _r_optim, optimizer* _val_optim, optimizer* _adv_optim) {
		allocated = false;
		world = snowy;
		agent = _net;
		memory = new ReplayMemory(agent->input_size, agent->output_size);
		root_optim = _r_optim;
		val_optim = _val_optim;
		adv_optim = _adv_optim;
	}

	void allocate(int _memory_size, int _batch_size) {
		memory_size = _memory_size;
		batch_size = _batch_size;
		n_batch = memory_size / batch_size;
		batch_x = km_2d::alloc(batch_size, agent->input_size);
		batch_y = km_2d::alloc(batch_size, agent->output_size);
		batch_adv_dLoss = km_2d::alloc(batch_size, agent->output_size);
		batch_val_dLoss = km_2d::alloc(batch_size, 1);
		virtual_S = km_2d::alloc(agent->output_size, agent->input_size);
		virtual_R = km_1d::alloc(agent->output_size);
		virtual_T = new bool[agent->output_size];
		virtual_Q = km_1d::alloc(agent->output_size);
		agent->AllocMemory(batch_size);
		agent->UseMemory(1);
		memory->allocate(memory_size);
		allocated = true;
	}

	double fit() {
		double loss = 0.0;
		agent->StartLearning();
		for (int i = 0; i < n_batch; i++) {
			root_optim->zero_grad();
			val_optim->zero_grad();
			adv_optim->zero_grad();
			memory->fetch(batch_x, batch_y, batch_size);
			agent->batch_predict(batch_x);
			loss += agent->max_Loss_fn(batch_val_dLoss, batch_adv_dLoss, batch_y);

			agent->backward(batch_val_dLoss, batch_adv_dLoss);
			root_optim->step();
			val_optim->step();
			adv_optim->step();
		}
		cout << "loss : " << loss / (n_batch) << endl;
		agent->EndLearning();
		return loss;
	}

	void InitSession() {
		world->InitSession();
		world->interact(Snowy::HOLD);
		world->current_T = NON_TERMINAL;
	}

	int AgentTraining() {
		int Count = 0;
		InitSession();
		Snowy::ACTION a;
		while (true) {

			if (world->current_T == TERMINAL) {
				break;
			}
			a = (Snowy::ACTION)agent->stochastic_select(world->current_S);
			world->SupposeVirtualSituation(virtual_S, virtual_R, virtual_T);
			agent->max_evaluate(virtual_Q, virtual_S, virtual_R, virtual_T);
			if (memory->IsFull()) {
				fit();
				memory->InitSession();
			}
			memory->push(world->current_S, virtual_Q);
			world->interact(a);
			Count++;
		}
		return Count;
	}

	~SnowySolution() {
		if (allocated) {
			km_2d::free(batch_x, batch_size);
			km_2d::free(batch_y, batch_size);
			km_2d::free(batch_val_dLoss, batch_size);
			km_2d::free(batch_adv_dLoss, batch_size);

			km_2d::free(virtual_S, agent->output_size);
			km_1d::free(virtual_R);
			delete[] virtual_T;
			km_1d::free(virtual_Q);
		}
	}
};