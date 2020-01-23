#pragma once
#include "Env.h"
#include "DuelingNetwork_wrapper.h"
#include "Optim.h"
class McSamplingDoubleDqn {
public:

	DuelingNetwork_Wrapper* agent = nullptr;
	Env* env = nullptr;
	Optimizer* optim = nullptr;
	double gamma = 0.0;

	int n_buff = 0;

	DarrList ob_buff = nullptr;
	Darr reward_buff = nullptr;
	Darr target_buff = nullptr;
	Iarr action_buff = nullptr;
	bool* terminal_buff = nullptr;

	Darr actor_port = nullptr;
	Darr critic_port = nullptr;

	Darr actor_grad = nullptr;
	Darr critic_grad = nullptr;

	int n_action = 0;
	Darr buff_end_next_observation = nullptr;
	McSamplingDoubleDqn() {}
	McSamplingDoubleDqn(DuelingNetwork_Wrapper* agent_ptr, Env* env_ptr, Optimizer* optim_ptr, const double& gamma, const int& sampling_buffer_size) {
		if (agent_ptr->n_in != env_ptr->observationSize()) {
			cout << "invalid McSamplingDoubleDqn 1" << endl;
			assert(true);
		}

		if (agent_ptr->actor->n_out != agent_ptr->critic->n_out) {
			cout << "invalid McSamplingDoubleDqn 2" << endl;
			assert(true);
		}

		if (agent_ptr->actor->n_out != env_ptr->numOfAction()) {
			cout << "invalid McSamplingDoubleDqn 3" << endl;
			assert(true);
		}

		agent = agent_ptr;
		env = env_ptr;
		optim = optim_ptr;
		this->gamma = gamma;
		this->n_buff = sampling_buffer_size;
		ob_buff = alloc(n_buff, env->observationSize());
		reward_buff = alloc(n_buff);
		target_buff = alloc(n_buff);
		action_buff = ialloc(n_buff);
		terminal_buff = new bool[n_buff];
		n_action = env->numOfAction();
		actor_port = alloc(n_action);
		critic_port = alloc(n_action);
		actor_grad = alloc(n_action);
		critic_grad = alloc(n_action);
		buff_end_next_observation = alloc(env->observationSize());
	}

	~McSamplingDoubleDqn() {
		free(ob_buff, n_buff);
		free(reward_buff);
		free(target_buff);
		free(action_buff);
		free(actor_port);
		free(critic_port);
		free(actor_grad);
		free(critic_grad);
		free(buff_end_next_observation);
		if (terminal_buff != nullptr) {
			delete[] terminal_buff;
		}
	}
	void mc_sampling2(const double& epsilon) {
		int first_idx = 0;
		double future_reward = 0.0;
		if (env->isTerminal()) {
			env->initialize();
		}
		for (int t = 0; t < n_buff; t++) {
			env->getCurrentScene(ob_buff[t], reward_buff[t], terminal_buff[t]);
			if (terminal_buff[t] == TERMINAL) {

				mc_backward_evaluate(t - 1, first_idx, reward_buff[t], 0.0);
				env->initialize();
				env->getCurrentScene(ob_buff[t], reward_buff[t], terminal_buff[t]); /* init buff[t] */
				first_idx = t;
			}
			if (bernoulli_sampling(epsilon)) {
				agent->actor_predict(ob_buff[t], actor_port, 0);
				action_buff[t] = argmax(actor_port, n_action);
			}
			else {
				action_buff[t] = randint(n_action);
			}
			env->transitionOccur(action_buff[t]);
		}
		double buff_end_r = 0.0;
		bool buff_end_t = false;
		env->getCurrentScene(buff_end_next_observation ,buff_end_r, buff_end_t);
		double buff_end_next_target = 0.0;
		if (buff_end_t == NON_TERMINAL) {
			agent->forwardToFit(buff_end_next_observation, actor_port, critic_port, 0);
			buff_end_next_target = critic_port[argmax(actor_port, n_action)];
		}
		mc_backward_evaluate(n_buff - 1, first_idx, buff_end_r, buff_end_next_target);
		
	}

	void mc_backward_evaluate(const int& last, const int& first, const double& next_reward_of_last, const double& next_target_of_last) {
		target_buff[last] = next_reward_of_last + gamma * next_target_of_last;
		for (int t = last - 1; t >= first; t--) {
			target_buff[t] = reward_buff[t + 1] + gamma * target_buff[t + 1];
		}
	}

	void sample_learn() {
		double loss = 0.0;
	
		double a0 = 0.0;
		double a1 = 0.0;
		double a2 = 0.0;

		for (int t = 0; t < n_buff; t++) {
			optim->zero_grad();
			agent->forwardToFit(ob_buff[t], actor_port, critic_port, 0);
			for (int a = 0; a < n_action; a++) {
				actor_grad[a] = 0.0;
				critic_grad[a] = 0.0;
			}
			critic_grad[action_buff[t]] = 2.0 * (critic_port[action_buff[t]] - target_buff[t]);
			actor_grad[action_buff[t]] = 2.0 * (actor_port[action_buff[t]] - critic_port[action_buff[t]]);

			agent->backwardToFit(actor_grad, critic_grad, 0);
			optim->step();
			//	cout << "actor : " << actor_grad[action_buff[t]] << " critic : " << critic_grad[action_buff[t]] << " target : " << target_buff[t] << endl;
			loss += (critic_port[action_buff[t]] - target_buff[t]) * (critic_port[action_buff[t]] - target_buff[t]);
			a0 += actor_port[0];
			a1 += actor_port[1];
			a2 += actor_port[2];
		}
		cout << "target loss:" << a0 / (double)n_buff <<" " << a1 / (double)n_buff <<" " << a2/ (double)n_buff << endl;
	}
};