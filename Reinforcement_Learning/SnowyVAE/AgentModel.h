#pragma once
#include "ActorCritic.h"
#include "DensityNetwork.h"
class AgentModel : public Unit {
public:
	GaussianDensityNetwork* encoder = nullptr;
	ActorCritic* agent = nullptr;
	Sequential* decoder = nullptr;


	darr::v decoder_input_port = nullptr;
	darr::v decoder_output_grad = nullptr;
	darr::v2D decoder_output_cache = nullptr;

	int state_len = 0;


	darr::v prior_mean = nullptr;
	darr::v prior_mean_grad = nullptr;
	darr::v pre_prior_stddev = nullptr;
	darr::v pre_prior_stddev_grad = nullptr;

	darr::v2D e_rand_cache = nullptr;
	darr::v2D e_mean_cache = nullptr;
	darr::v2D e_stddev_cache = nullptr;
	darr::v e_mean_grad = nullptr;
	darr::v e_stddev_grad = nullptr;

	darr::v state_buff = nullptr;

	
	int output_channel = 0;
	int output_plain_size = 0;

	AgentModel(GaussianDensityNetwork* encoder, ActorCritic* agent, Sequential* decoder, const int& output_channel, const int& output_plain_size) {

		this->encoder = encoder;
		this->agent = agent;
		this->decoder = decoder;
		this->output_channel = output_channel;
		this->output_plain_size = output_plain_size;
		state_len = encoder->random_vector_size;

		prior_mean = darr::alloc(state_len);
		prior_mean_grad = darr::alloc(state_len);
		pre_prior_stddev = darr::alloc(state_len);
		pre_prior_stddev_grad = darr::alloc(state_len);
		e_mean_grad = darr::alloc(state_len);
		e_stddev_grad = darr::alloc(state_len);
		for (int i = 0; i < state_len; i++) {
			prior_mean[i] = 0.0;
			pre_prior_stddev[i] = 0.0;
		}
		decoder_input_port = darr::alloc(state_len);
		decoder_output_grad = darr::alloc(decoder->n_out);
		state_buff = darr::alloc(state_len);
	}

	virtual void setCache(const int& new_cache_size) {
		encoder->setCache(new_cache_size);
		agent->setCache(new_cache_size);
		decoder->setCache(new_cache_size);
		darr::resize(e_rand_cache, this->n_cache, new_cache_size, state_len);
		darr::resize(e_mean_cache, this->n_cache, new_cache_size, state_len);
		darr::resize(e_stddev_cache, this->n_cache, new_cache_size, state_len);
		darr::resize(decoder_output_cache, this->n_cache, new_cache_size, decoder->n_out);
		this->in_cache = encoder->in_cache;
		this->in_grad_cache = encoder->in_grad_cache;
		this->n_cache = new_cache_size;
	}

	virtual void takeParams(vector<darr::v>& param_bag, vector<darr::v>& grad_bag, vector<int>& len_bag) {
		encoder->takeParams(param_bag, grad_bag, len_bag);
		agent->takeParams(param_bag, grad_bag, len_bag);
		decoder->takeParams(param_bag, grad_bag, len_bag);
		param_bag.push_back(prior_mean);
		grad_bag.push_back(prior_mean_grad);
		len_bag.push_back(state_len);
		param_bag.push_back(pre_prior_stddev);
		grad_bag.push_back(pre_prior_stddev_grad);
		len_bag.push_back(state_len);
	}

	int actor_sampling(darr::v const& observation) {
		encoder->charge(observation, 0);
		encoder->gdn_forward(e_mean_cache[0], e_stddev_cache[0], 0);
		for (int i = 0; i < state_len; i++) {
			state_buff[i] = e_stddev_cache[0][i] * STD(rEngine) + e_mean_cache[0][i];
		}
		return agent->actor_sampling(state_buff);
	}

	int best_choice(darr::v const& observation) {
		encoder->charge(observation, 0);
		encoder->gdn_forward(e_mean_cache[0], e_stddev_cache[0], 0);
		for (int i = 0; i < state_len; i++) {
			state_buff[i] = e_stddev_cache[0][i] * STD(rEngine) + e_mean_cache[0][i];
		}
		return agent->best_choice(state_buff);
	}

	void actor_sampling(darr::v const& observation, int& sampled_action, double& prob) {
		encoder->charge(observation, 0);
		encoder->gdn_forward(e_mean_cache[0], e_stddev_cache[0], 0);
		for (int i = 0; i < state_len; i++) {
			state_buff[i] = e_stddev_cache[0][i] * STD(rEngine) + e_mean_cache[0][i];
		}
		agent->actor_sampling(state_buff, sampled_action, prob);
	}


	double predict_maxQ(darr::v const& observation) {
		encoder->charge(observation, 0);
		encoder->gdn_forward(e_mean_cache[0], e_stddev_cache[0], 0);
		for (int i = 0; i < state_len; i++) {
			state_buff[i] = e_stddev_cache[0][i] * STD(rEngine) + e_mean_cache[0][i];
		}
		return agent->predict_maxQ(state_buff);
	}

	int best_choice(darr::v const& observation, darr::v& next_a0, darr::v& next_a1, darr::v& next_a2) {
		encoder->charge(observation, 0);
		encoder->gdn_forward(e_mean_cache[0], e_stddev_cache[0], 0);
		for (int i = 0; i < state_len; i++) {
			state_buff[i] = e_stddev_cache[0][i] * STD(rEngine) + e_mean_cache[0][i];
		}
		decoder->charge(state_buff, 0);
		decoder->forward(next_a0, 0);

		return agent->best_choice(state_buff);
	}


	void forward(darr::v& output, const int& id) {}

	void backward(darr::v& output_grad, const int& id) {}

	void calculate_gradient(double& gain, double& decoder_nll, double& kld, darr::v const& observation, const double& td_target, const int& target_action, const double& old_prob, darr::v const& next_observation) {


		/*_____________________________model forward___________________________*/
		encoder->charge(observation, 0);
		encoder->gdn_forward(e_mean_cache[0], e_stddev_cache[0], 0);
		km::normal_sampling(e_rand_cache[0], state_len, 0.0, 1.0);
		for (int i = 0; i < state_len; i++) {
			//e_rand_cache[0][i] = STD(rEngine);
			state_buff[i] = e_stddev_cache[0][i] * e_rand_cache[0][i] + e_mean_cache[0][i];
		}
		decoder->charge(state_buff, 0);
		decoder->forward(decoder_output_cache[0], 0);
		decoder_nll += km::CELoss_2D(decoder_output_grad,
			decoder_output_cache[0],
			observation,
			output_channel,
			output_plain_size);
		decoder->backward(decoder_output_grad, 0);
		gain += agent->calculate_gradient(state_buff, td_target, target_action, old_prob);

		double mean_kld_grad = 0.0;
		double stddev_kld_grad = 0.0;

		double prior_stddev = 0.0;
		double prior_stddev_grad = 0.0;

		double beta = 0.001;

		double x = 0.0;
		for (int i = 0; i < state_len; i++) {
			if (pre_prior_stddev[i] > 0.0) {
				prior_stddev = pre_prior_stddev[i] + 1.0;
			}
			else {
				prior_stddev = exp(pre_prior_stddev[i]);
			}

			x = e_stddev_cache[0][i] * e_rand_cache[0][i] + e_mean_cache[0][i];

			mean_kld_grad = (x - prior_mean[i]) / (prior_stddev * prior_stddev);

			stddev_kld_grad = (((x - prior_mean[i]) * e_rand_cache[0][i])/ (prior_stddev * prior_stddev)) - 1.0 / e_stddev_cache[0][i];
			prior_stddev_grad = (1.0 - (((x - prior_mean[i]) * (x - prior_mean[i])) / (prior_stddev * prior_stddev))) / prior_stddev;
			/*______________________________________________________________________________________________*/
			prior_mean_grad[i] += beta * (-x + prior_mean[i]) / (prior_stddev * prior_stddev);
			if (pre_prior_stddev[i] > 0.0) {
				pre_prior_stddev_grad[i] += beta * prior_stddev_grad;
			}
			else {
				pre_prior_stddev_grad[i] += beta * prior_stddev_grad * prior_stddev;
			}
			e_mean_grad[i] = (beta * mean_kld_grad) + decoder->in_grad_cache[0][i];
			e_stddev_grad[i] = (beta * stddev_kld_grad) + decoder->in_grad_cache[0][i] * e_rand_cache[0][i];
			/*£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ£þ*/
			kld += 0.5 * ((((x - prior_mean[i]) * (x - prior_mean[i])) / (prior_stddev * prior_stddev) - e_rand_cache[0][i] * e_rand_cache[0][i]));
			kld += log(prior_stddev / e_stddev_cache[0][i]);

		}
		encoder->gdn_backward(e_mean_grad, e_stddev_grad, 0);
	}
};