#pragma once
#include "environment.h"
#include "model.h"
#include "optimizer.h"
#include "loss_function.h"
#include <vector>

MSELoss ddqnloss_fn;

class ddqn {
public:
	nn* selecter;
	nn* net;
	optimizer* optim;
	ReplaysManager* memory;
	enviroment* env;
	double gamma;
	double e;
	int n_state;
	int n_action;
	bool is_allocated;
	double* state_vector;
	double** dLoss;
	double** batch_x;
	double** batch_y;
	int batch_size;
	int mem_size;
	int nb_batch;
	vector<ACTION> actionList;
	vector<STATE> stateList;
	ddqn() { is_allocated = false; }
	ddqn(enviroment& _env, double _gamma, double _e) {
		env = &_env;
		n_state = env->n_state;
		n_action = env->n_action;
		gamma = _gamma;
		e = _e;
		is_allocated = false;
	}
	~ddqn() {
		if (is_allocated) {

			km_2d::free(batch_x, batch_size);
			km_2d::free(batch_y, batch_size);
			km_2d::free(dLoss, batch_size);
			km_1d::free(state_vector);
			delete memory;
		}
	}

	void alloc_memorys(int n_memory, int _n_batch) {
		batch_size = _n_batch;
		mem_size = n_memory;
		nb_batch = mem_size / batch_size;
		memory = new ReplaysManager(n_memory);
		net->alloc_single_memory();
		net->alloc_batch_memory(_n_batch);
		state_vector = km_1d::alloc(n_state);
		batch_x = km_2d::alloc(_n_batch, n_state);
		batch_y = km_2d::alloc(_n_batch, n_action);
		dLoss = km_2d::alloc(_n_batch, n_action);
		memory->init_session();
		is_allocated = true;
		selecter = new nn(number_of_units(net->n_layer));
		for (int i = 0; i < net->n_layer; i++) {
			selecter->layer[i] = net->layer[i]->clone();
		}
		selecter->alloc_single_memory();
		selecter->alloc_batch_memory(1);
	}

	void setSvector(const STATE& s) {
		km_1d::fill_zero(state_vector, n_state);
		state_vector[s.first*env->width + s.second] = 1.0;
	}

	void set_batch() {
		STATE s;
		ACTION a;
		STATE next_s;
		double r;
		bool isterminal;
		km_2d::fill_zero(batch_x, net->batch_size, n_state);
		km_2d::fill_zero(batch_y, net->batch_size, n_action);
		for (int i = 0; i < net->batch_size; i++) {
			memory->recall(s, a, next_s, r, isterminal);
			batch_x[i][s.first*env->width + s.second] = 1.0;

			if (isterminal) {
				batch_y[i][a] = r;
			}
			else {
				batch_y[i][a] = r + gamma * selecter->single_output[select(next_s)];
			}
		}
	}

	ACTION select(const STATE& s) {
		setSvector(s);
		selecter->single_feed_forward(state_vector);
		return (ACTION)selecter->single_argmax;
	}

	ACTION unstable_select(const STATE& s) {
		if (e < rand0to1(rEngine)) {
			return (ACTION)randint(n_action);
		}
		else {
			return select(s);
		}
	}

	void take(const STATE& cur_state, const ACTION& cur_action, const STATE& next_state, const double& cur_reward, const bool& _TERMINAL) {
		if (memory->isfull()) {
			this->fit();
			memory->init_session();
		}
		memory->memorize(cur_state, cur_action, next_state, cur_reward, _TERMINAL);

	}
	double training_run(STATE start) {

		STATE s;
		ACTION a;
		STATE next_s = start;
		ACTION next_a = unstable_select(s);
		double reward = 0.0;
		double eval = 0.0;

		while (true) {

			s = next_s;
			a = next_a;
			reward += env->reward(s);
			if (env->isTerminal(s)) {
				break;
			}
			next_s = env->next(s, a);

			take(s, a, next_s, env->reward(next_s), env->isTerminal(next_s));
			next_a = unstable_select(next_s);
		}
		return reward;
	}
	void printPath() {
		for (int i = 0; i < stateList.size(); i++) {
			cout << stateList[i].first << ", " << stateList[i].second << endl;
		}
	}
	double testing_run(STATE start) {
		stateList.clear();
		actionList.clear();
		STATE s;
		ACTION a;
		STATE next_s = start;
		ACTION next_a = select(s);
		double reward = 0.0;
		double eval = 0.0;

		while (true) {

			s = next_s;
			a = next_a;
			stateList.push_back(s);
			actionList.push_back(a);
			reward += env->reward(s);
			if (env->isTerminal(s)) {
				break;
			}
			next_s = env->next(s, a);
			next_a = select(next_s);
		}
		return reward;
	}

	double fit() {
		double loss = 0.0;
		for (int i = 0; i < nb_batch; i++) {
			net->zero_grad();
			set_batch();
			net->batch_feed_forward(batch_x);
			loss += ddqnloss_fn.batch_loss_prime(dLoss, net->batch_output, batch_y, net->batch_size, n_action);
			net->batch_feed_backward(dLoss);
			optim->step();
		}
		for (int i = 0; i < net->n_layer; i++) {
			selecter->layer[i]->ddqn_selecter_update(net->layer[i], 0.05);
		}
		return loss;
	}
};