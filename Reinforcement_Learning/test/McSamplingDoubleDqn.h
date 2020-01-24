#pragma once
#include "Env.h"
#include "DuelingNetwork_wrapper.h"
#include "Optim.h"


class McSamplingActorCritic {
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
	Darr action_prob_buff = nullptr;

	Darr actor_port = nullptr;
	Darr critic_port = nullptr;

	Darr actor_grad = nullptr;
	Darr critic_grad = nullptr;

	int n_action = 0;
	Darr buff_end_next_observation = nullptr;
	McSamplingActorCritic() {}
	McSamplingActorCritic(DuelingNetwork_Wrapper* agent_ptr, Env* env_ptr, Optimizer* optim_ptr, const double& gamma, const int& sampling_buffer_size) {
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
		action_prob_buff = alloc(n_buff);
		n_action = env->numOfAction();
		actor_port = alloc(n_action);
		critic_port = alloc(n_action);
		actor_grad = alloc(n_action);
		critic_grad = alloc(n_action);
		buff_end_next_observation = alloc(env->observationSize());
	}

	~McSamplingActorCritic() {
		free(ob_buff, n_buff);
		free(reward_buff);
		free(target_buff);
		free(action_buff);
		free(actor_port);
		free(critic_port);
		free(actor_grad);
		free(critic_grad);
		free(buff_end_next_observation);
		free(action_prob_buff);
	}
	void mc_sampling() {
		int first_idx = 0;
		double future_reward = 0.0;
		bool terminal_buff = false;
		if (env->isTerminal()) {
			env->initialize();
		}
		for (int t = 0; t < n_buff; t++) {
			env->getCurrentScene(ob_buff[t], reward_buff[t], terminal_buff);
			if (terminal_buff == TERMINAL) {
				mc_backward_evaluate(t - 1, first_idx, reward_buff[t], 0.0);
				env->initialize();
				first_idx = t;
				env->getCurrentScene(ob_buff[t], reward_buff[t], terminal_buff); /* init buff[t] */
			}

			agent->actor_sampling(ob_buff[t], action_buff[t], action_prob_buff[t], 0);
			env->transitionOccur(action_buff[t]);
		}

		double buff_end_r = 0.0;
		bool buff_end_t = false;
		env->getCurrentScene(buff_end_next_observation, buff_end_r, buff_end_t);
		double buff_end_next_target = 0.0;
		if (buff_end_t == NON_TERMINAL) {
			agent->charge(buff_end_next_observation, 0);
			agent->duel_forward(actor_port, critic_port, 0);
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
		double target = 0.0;
		double mean_v = 0.0;
		for (int t = n_buff-1; t >= 0; t--) {
			optim->zero_grad();
			agent->charge(ob_buff[t], 0);
			agent->duel_forward(actor_port, critic_port, 0);
			double v = 0.0;
			for (int a = 0; a < n_action; a++) {
				v += actor_port[a] * critic_port[a];
				actor_grad[a] = 0.0;
				critic_grad[a] = 0.0;
			}
			mean_v += v;
			target += target_buff[t];
			critic_grad[action_buff[t]] = 2.0 * (critic_port[action_buff[t]] - target_buff[t]);
			actor_grad[action_buff[t]] = (v - critic_port[action_buff[t]])/action_prob_buff[t];

			agent->duel_bacward(actor_grad, critic_grad, 0);
			optim->step();

		}
		cout << "mean value : " <<mean_v / (double)n_buff << "	target:"<<target/(double)n_buff<<endl;
	}
};
