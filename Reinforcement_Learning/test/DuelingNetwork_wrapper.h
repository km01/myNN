#pragma once
#include "Unit.h"

class DuelingNetwork_Wrapper : public Unit {
public:

	Sequential* base = nullptr;
	Sequential* critic = nullptr;
	Sequential* actor = nullptr;
	Darr base_outport = nullptr;
	Darr actor_output_buffer = nullptr;
	DuelingNetwork_Wrapper() {}
	DuelingNetwork_Wrapper(Sequential* base, Sequential* actor, Sequential* critic) {
		this->n_in = base->n_in;
		this->n_out = actor->n_out; // + critic->n_out ?
		this->base = base;
		this->critic = critic;
		this->actor = actor;
		if (critic->n_in == actor->n_in && actor->n_in == base->n_out) {
		}
		else {
			cout << "invalid DuelingNetwork" << endl;
			assert(true);
		}
		grad_port = base->grad_port;
		base_outport = alloc(base->n_out);
		actor_output_buffer = alloc(actor->n_out);
	}

	~DuelingNetwork_Wrapper() {
		free(base_outport);
		free(actor_output_buffer);
	}

	virtual void forward(Darr& next_input, const int& id) {
		cout << " I don't use this function" << endl;
		assert(true);
	}

	virtual void backward(Darr const& output_grad, const int& id) {
		cout << " I don't use this function" << endl;
		assert(true);
	}

	virtual void takeParams(vector<Darr>& param_bag, vector<Darr>& grad_bag, vector<int>& len_bag) {
		base->takeParams(param_bag, grad_bag, len_bag);
		actor->takeParams(param_bag, grad_bag, len_bag);
		critic->takeParams(param_bag, grad_bag, len_bag);
	}

	virtual void setCache(const int& cache_size) {
		base->setCache(cache_size);
		actor->setCache(cache_size);
		critic->setCache(cache_size);
		in_port = base->in_port;
		this->cache_size = cache_size;
	}

	void actor_predict(Darr const& observation, Darr& actor_output, const int& id) {
		base->charge(observation, id);
		base->forward(actor->in_port[id], id);
		actor->forward(actor_output, id);
	}

	int actor_argmax(Darr const& observation) {
		base->charge(observation, 0);
		base->forward(actor->in_port[0], 0);
		actor->forward(actor_output_buffer, 0);
		return argmax(actor_output_buffer, actor->n_out);
	}

	int actor_sampling(Darr const& observation) {
		base->charge(observation, 0);
		base->forward(actor->in_port[0], 0);
		actor->forward(actor_output_buffer, 0);
		return km::multinomial_sampling(actor_output_buffer, actor->n_out);
	}
	
	void actor_sampling(Darr const& observation, int& sampled_action, double& prob, const int& id) {
		base->charge(observation, id);
		base->forward(actor->in_port[id], id);
		actor->forward(actor_output_buffer, id);
		sampled_action = km::multinomial_sampling(actor_output_buffer, actor->n_out);
		prob = actor_output_buffer[sampled_action];
	}

	void critic_predict(Darr const& observation, Darr& critic_output, const int& id) {
		base->charge(observation, id);
		base->forward(critic->in_port[id], id);
		critic->forward(critic_output, id);
	}

	void duel_forward(Darr& actor_output, Darr& critic_output, const int& id) {
		base->forward(actor->in_port[id], id);
		critic->charge(actor->in_port[id], id); /* set actor->in_port[id] == critic->in_port[id] */
		actor->forward(actor_output, id);
		critic->forward(critic_output, id);
	}

	void duel_bacward(Darr& actor_grad, Darr& critic_grad, const int& id) {
		actor->backward(actor_grad, id);
		critic->backward(critic_grad, id);
		for (int i = 0; i < base->n_out; i++) {
			base_outport[i] = actor->grad_port[i] + critic->grad_port[i];
		}
		base->backward(base_outport, id);
	}
};