#pragma once
#include "environment.h"
#include "model.h"
#include "optimizer.h"
#include "loss_function.h"
#include <vector>

MSELoss deuling_dqnloss_fn;


class deul_net {
public:
	nn* root_net;
	nn* value_net;
	nn* action_net;
	bool is_allocated;
	double* single_output;
	double** batch_output;
	double** batch_root_intersection_port;
	double** batch_value_dLoss;
	int single_argmax;
	int* batch_argmax;
	int batch_size;
	deul_net() { is_allocated = false; }
	deul_net(number_of_units root, number_of_units value, number_of_units action) {
		root_net = new nn(root);
		value_net = new nn(value);
		action_net = new nn(action);
		is_allocated = false;
	}
	void alloc_single_memory() {
		single_output = km_1d::alloc(action_net->output_size);

		root_net->alloc_single_memory();
		value_net->alloc_single_memory();
		action_net->alloc_single_memory();
	}
	void alloc_batch_memory(int _batch_size) {
		batch_size = _batch_size;
		alloc_single_memory();
		batch_argmax = new int[batch_size];
		batch_output = km_2d::alloc(batch_size, action_net->output_size);
		batch_value_dLoss = km_2d::alloc(batch_size, value_net->output_size);
		batch_root_intersection_port = km_2d::alloc(batch_size, root_net->output_size);
		root_net->alloc_batch_memory(batch_size);
		value_net->alloc_batch_memory(batch_size);
		action_net->alloc_batch_memory(batch_size);
		is_allocated = true;
	}

	void single_feed_forward(double* const& single_input) {
		root_net->single_feed_forward(single_input);
		value_net->single_feed_forward(root_net->single_output);
		action_net->single_feed_forward(root_net->single_output);
		for (int i = 0; i < action_net->output_size; i++) {
			single_output[i] = action_net->single_output[i] + value_net->single_output[0];
		}
		km_1d::argmax(single_argmax, single_output, action_net->output_size);
	}

	void batch_feed_forward(double** const& mini_x) {
		root_net->batch_feed_forward(mini_x);
		value_net->batch_feed_forward(root_net->batch_output);
		action_net->batch_feed_forward(root_net->batch_output);
		for (int i = 0; i < batch_size; i++) {
			for (int j = 0; j < action_net->output_size; j++) {
				batch_output[i][j] = action_net->batch_output[i][j] + value_net->batch_output[i][0];
			}
		}
		km_2d::argmax(batch_argmax, batch_output, batch_size, action_net->output_size);
	}
	void batch_feed_backward(double** const& dLoss) {
		for (int i = 0; i < batch_size; i++) {
			batch_value_dLoss[i][0] = dLoss[i][batch_argmax[i]];
		}
		action_net->batch_feed_backward(dLoss);
		value_net->batch_feed_backward(batch_value_dLoss);
		for (int i = 0; i < batch_size; i++) {
			for (int j = 0; j < root_net->output_size; j++) {
				batch_root_intersection_port[i][j] = action_net->layer[0]->batch_port[i][j] + value_net->layer[0]->batch_port[i][j];
			}
		}
		root_net->batch_feed_backward(batch_root_intersection_port);
	}
	~deul_net() {

		if (is_allocated) {
			km_2d::free(batch_value_dLoss, batch_size);
			km_2d::free(batch_output, batch_size);
			km_2d::free(batch_root_intersection_port, batch_size);
			delete[] batch_argmax;
			delete[] single_output;
		}
		delete root_net;
		delete action_net;
		delete value_net;

	}
	void zero_grad() {
		root_net->zero_grad();
		value_net->zero_grad();
		action_net->zero_grad();
	}

};

class DeulNetOptimizer{
public:
	optimizer* r_optim;
	optimizer* v_optim;
	optimizer* a_optim;
	DeulNetOptimizer(){}
	DeulNetOptimizer(deul_net& trainee) {
		r_optim = new optimizer(*trainee.root_net);
		v_optim = new optimizer(*trainee.value_net);
		a_optim = new optimizer(*trainee.action_net);
	}
	void use_rmsprop(double RMSprop_beta) {
		r_optim->use_RMSprop(RMSprop_beta);
		v_optim->use_RMSprop(RMSprop_beta);
		a_optim->use_RMSprop(RMSprop_beta);
	}
	void set_learning_rate(double _learing_rate) {
		r_optim->set_learning_rate(_learing_rate);
		v_optim->set_learning_rate(_learing_rate);
		a_optim->set_learning_rate(_learing_rate);
	}
	void step() {
		r_optim->step();
		v_optim->step();
		a_optim->step();
	}
	~DeulNetOptimizer() {

		delete r_optim;
		delete v_optim;
		delete a_optim;

	}
};
class deuling_dqn {
public:
	deul_net* net;
	ReplaysManager* memory;
	enviroment* env;
	DeulNetOptimizer* optim;
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
	deuling_dqn() { is_allocated = false; }

	deuling_dqn(enviroment& _env, double _gamma, double _e) {
		env = &_env;
		n_state = env->n_state;
		n_action = env->n_action;
		gamma = _gamma;
		e = _e;
		is_allocated = false;
	}
	~deuling_dqn() {
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
		km_2d::fill_zero(batch_x, batch_size, n_state);
		km_2d::fill_zero(batch_y, batch_size, n_action);
		for (int i = 0; i < batch_size; i++) {
			memory->recall(s, a, next_s, r, isterminal);
			batch_x[i][s.first*env->width + s.second] = 1.0;

			if (isterminal) {
				batch_y[i][a] = r;
			}
			else {
				batch_y[i][a] = r + gamma * net->single_output[select(next_s)];
			}
		}
	}

	ACTION select(const STATE& s) {
		setSvector(s);
		net->single_feed_forward(state_vector);
		return (ACTION)net->single_argmax;
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
		int count = 0;
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
			if (count > 1000) {
				cout <<" Infinity Loop "<< endl;
				return 0.0;
			}
			count++;
		}
		return reward;
	}
	double fit() {
		double loss = 0.0;
		for (int i = 0; i < nb_batch; i++) {
			net->zero_grad();
			set_batch();
			net->batch_feed_forward(batch_x);
			loss += deuling_dqnloss_fn.batch_loss_prime(dLoss, net->batch_output, batch_y, batch_size, n_action);
			net->batch_feed_backward(dLoss);
			optim->step();
		}
		cout <<"Loss : "<< loss << endl;
		return loss;
	}
};