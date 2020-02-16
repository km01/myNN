#pragma once
#include "Unit.h"

class ActorCritic : public Unit {
public:
	Sequential* root = nullptr;
	Sequential* actor = nullptr;
	Sequential* critic = nullptr;
	darr::v root_output = nullptr;
	darr::v root_grad = nullptr;
	darr::v actor_output = nullptr;
	darr::v actor_grad = nullptr;
	darr::v critic_output = nullptr;
	darr::v critic_grad = nullptr;
	ActorCritic() {}
	~ActorCritic() {
		darr::free(root_output);
		darr::free(root_grad);
		darr::free(actor_output);
		darr::free(actor_grad);
		darr::free(critic_output);
		darr::free(critic_grad);
	}
	ActorCritic(Sequential& root, Sequential& actor, Sequential& critic) {

		if (actor.n_out != critic.n_out) {
			cout << "invalid Model2 .1" << endl;
			assert(false);
		}

		if (actor.n_in != critic.n_in || root.n_out != critic.n_in) {
			cout << "invalid Model2 .1" << endl;
			assert(false);
		}
		
		this->root = &root;
		this->actor = &actor;
		this->critic = &critic;

		n_in = this->root->n_in;
		n_out = this->actor->n_out;

		root_output = darr::alloc(this->root->n_out);
		root_grad = darr::alloc(this->root->n_out);

		actor_output = darr::alloc(this->actor->n_out);
		actor_grad = darr::alloc(this->actor->n_out);

		critic_output = darr::alloc(this->critic->n_out);
		critic_grad = darr::alloc(this->critic->n_out);
	}


	virtual void setCache(const int& new_cache_size) {
		root->setCache(new_cache_size);
		actor->setCache(new_cache_size);
		critic->setCache(new_cache_size);

		this->in_cache = root->in_cache;
		this->in_grad_cache = root->in_grad_cache;
		this->n_cache = new_cache_size;
	}

	virtual void takeParams(vector<darr::v>& param_bag, vector<darr::v>& grad_bag, vector<int>& len_bag) {
		root->takeParams(param_bag, grad_bag, len_bag);
		actor->takeParams(param_bag, grad_bag, len_bag);
		critic->takeParams(param_bag, grad_bag, len_bag);
	}

	int actor_sampling(darr::v const& state) {
		root->charge(state, 0);
		root->forward(actor->in_cache[0], 0);
		actor->forward(actor_output, 0);
		return km::multinoulli_sampling(actor_output, actor->n_out);
	}

	int best_choice(darr::v const& state) {
		root->charge(state, 0);
		root->forward(actor->in_cache[0], 0);
		actor->forward(actor_output, 0);
		return km::argmax(actor_output, actor->n_out);
	}

	void actor_sampling(darr::v const& state, int& sampled_action, double& prob) {
		root->charge(state, 0);
		root->forward(actor->in_cache[0], 0);
		actor->forward(actor_output, 0);
		sampled_action = km::multinoulli_sampling(actor_output, actor->n_out);
		prob = actor_output[sampled_action];
	}

	void forward(darr::v& output, const int& id) {}

	void backward(darr::v& output_grad, const int& id) {}

	double predict_maxQ(darr::v const& state) {
		root->charge(state, 0);
		root->forward(root_output, 0);

		actor->charge(root_output, 0);
		actor->forward(actor_output, 0);
		
		critic->charge(root_output, 0);
		critic->forward(critic_output, 0);
		
		return critic_output[km::argmax(actor_output, actor->n_out)];
	}

	double calculate_gradient(darr::v const& state, const double& td_target, const int& target_action, const double& old_prob) {

		/*_____________________________model forward___________________________*/
		root->charge(state, 0);
		root->forward(root_output, 0);

		actor->charge(root_output, 0);
		actor->forward(actor_output, 0);

		critic->charge(root_output, 0);
		critic->forward(critic_output, 0);
		/*£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ*/


		/*_______________________________critic loss_____________________________*/
		double expected_value = 0.0;
		for (int a = 0; a < actor->n_out; a++) {
			expected_value += actor_output[a] * critic_output[a];
			actor_grad[a] = 0.0;
			critic_grad[a] = 0.0;
		}
		critic_grad[target_action] = 2.0 * (critic_output[target_action] - td_target);
		/*£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ*/


		/*_______________________________actor loss______________________________*/
		double Ep = 0.25;
		if (td_target > expected_value) {
			if (actor_output[target_action] > (1.0 + Ep)* old_prob) {
				actor_grad[target_action] = -(1.0 + Ep) / actor_output[target_action];
			}
			else {
				actor_grad[target_action] = -(td_target - expected_value) / old_prob;
			}
		}
		else {
			if (actor_output[target_action] < (1.0 - Ep) * old_prob) {
				actor_grad[target_action] = -(1.0 - Ep) / actor_output[target_action];
			}
			else {
				actor_grad[target_action] = -(td_target - expected_value) / old_prob;
			}
		}
		/*£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ*/


		/*_____________________________model backward____________________________*/
		actor->backward(actor_grad, 0);
		critic->backward(critic_grad, 0);
		for (int i = 0; i < root->n_out; i++) {
			root_grad[i] = actor->in_grad_cache[0][i] + critic->in_grad_cache[0][i];
		}
		root->backward(root_grad, 0);
		/*£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ*/
		return expected_value;
	}
};
