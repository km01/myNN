#pragma once

#include "AgentModel.h"
#include "Env.h"
class McSamplingSnowyActorCritic {
public:


	double gamma = 0.0;

	int observation_size = 0;
	int n_buff = 0;

	darr::v2D ob_buff = nullptr;
	darr::v2D next_ob_buff = nullptr;
	darr::v reward_buff = nullptr;
	darr::v target_buff = nullptr;
	iarr::v action_buff = nullptr;
	darr::v action_prob_buff = nullptr;

	darr::v buff_end_next_observation = nullptr;

	McSamplingSnowyActorCritic() {}
	McSamplingSnowyActorCritic(const int& observation_size, const double& gamma, const int& sampling_buffer_size) {

		this->observation_size = observation_size;
		this->gamma = gamma;
		this->n_buff = sampling_buffer_size;
		ob_buff = darr::alloc(n_buff, observation_size);
		next_ob_buff = darr::alloc(n_buff, observation_size);
		buff_end_next_observation = darr::alloc(observation_size);
		reward_buff = darr::alloc(n_buff);
		target_buff = darr::alloc(n_buff);
		action_buff = iarr::alloc(n_buff);
		action_prob_buff = darr::alloc(n_buff);
	}

	~McSamplingSnowyActorCritic() {

	}

	void mc_sampling(Env& env, ActorCritic& agent) {
		int first_idx = 0;
		double R = 0.0;
		bool terminal_buff = false;
		
		for (int t = 0; t < n_buff; t++) {

			env.getCurrentObservation(ob_buff[t]);
			terminal_buff = env.isTerminal_Now();
			reward_buff[t] = env.getCurrentReward();

			if (terminal_buff == TERMINAL) {
				mc_backward_evaluate(t - 1, first_idx, reward_buff[t], 0.0);
				env.initialize();
				first_idx = t;

				env.getCurrentObservation(ob_buff[t]);
				terminal_buff = env.isTerminal_Now();
				reward_buff[t] = env.getCurrentReward();

				if (terminal_buff == TERMINAL) {
					cout << "initial state is terminal" << endl;
					assert(false);
				}
			}
			
			agent.actor_sampling(ob_buff[t], action_buff[t], action_prob_buff[t]);
			env.transitionOccur(action_buff[t]);
		}

		double buff_end_next_target = 0.0;
		if (env.isTerminal_Now() == TERMINAL) {
			env.getCurrentObservation(buff_end_next_observation);
			buff_end_next_target = agent.predict_maxQ(buff_end_next_observation);
		}
		mc_backward_evaluate(n_buff - 1, first_idx, env.getCurrentReward(), buff_end_next_target);
	}

	void mc_backward_evaluate(const int& last, const int& first, const double& next_reward_of_last, const double& next_target_of_last) {
		target_buff[last] = next_reward_of_last + gamma * next_target_of_last;
		for (int t = last - 1; t >= first; t--) {
			target_buff[t] = reward_buff[t + 1] + gamma * target_buff[t + 1];
		}
	}

};

class McSamplingSnowyActorCritic2 {
public:


	double gamma = 0.0;

	int observation_size = 0;
	int n_buff = 0;

	darr::v2D ob_buff = nullptr;
	darr::v2D next_ob_buff = nullptr;
	darr::v reward_buff = nullptr;
	darr::v target_buff = nullptr;
	iarr::v action_buff = nullptr;
	darr::v action_prob_buff = nullptr;

	darr::v buff_end_next_observation = nullptr;

	McSamplingSnowyActorCritic2() {}
	McSamplingSnowyActorCritic2(const int& observation_size, const double& gamma, const int& sampling_buffer_size) {

		this->observation_size = observation_size;
		this->gamma = gamma;
		this->n_buff = sampling_buffer_size;
		ob_buff = darr::alloc(n_buff, observation_size);
		next_ob_buff = darr::alloc(n_buff, observation_size);
		buff_end_next_observation = darr::alloc(observation_size);
		reward_buff = darr::alloc(n_buff);
		target_buff = darr::alloc(n_buff);
		action_buff = iarr::alloc(n_buff);
		action_prob_buff = darr::alloc(n_buff);
	}

	~McSamplingSnowyActorCritic2() {

	}

	void mc_sampling(Env& env, AgentModel& agent) {
		int first_idx = 0;
		double R = 0.0;
		bool terminal_buff = false;

		for (int t = 0; t < n_buff; t++) {

			env.getCurrentObservation(ob_buff[t]);
			terminal_buff = env.isTerminal_Now();
			reward_buff[t] = env.getCurrentReward();

			if (terminal_buff == TERMINAL) {
				mc_backward_evaluate(t - 1, first_idx, reward_buff[t], 0.0);
				env.initialize();
				first_idx = t;

				env.getCurrentObservation(ob_buff[t]);
				terminal_buff = env.isTerminal_Now();
				reward_buff[t] = env.getCurrentReward();

				if (terminal_buff == TERMINAL) {
					cout << "initial state is terminal" << endl;
					assert(false);
				}
			}

			agent.actor_sampling(ob_buff[t], action_buff[t], action_prob_buff[t]);
			env.transitionOccur(action_buff[t]);
			env.getCurrentObservation(next_ob_buff[t]);
		}

		double buff_end_next_target = 0.0;
		if (env.isTerminal_Now() == TERMINAL) {
			env.getCurrentObservation(buff_end_next_observation);
			buff_end_next_target = agent.predict_maxQ(buff_end_next_observation);
		}
		mc_backward_evaluate(n_buff - 1, first_idx, env.getCurrentReward(), buff_end_next_target);
	}

	void mc_backward_evaluate(const int& last, const int& first, const double& next_reward_of_last, const double& next_target_of_last) {
		target_buff[last] = next_reward_of_last + gamma * next_target_of_last;
		for (int t = last - 1; t >= first; t--) {
			target_buff[t] = reward_buff[t + 1] + gamma * target_buff[t + 1];
		}
	}

};
