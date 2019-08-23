#pragma once
#include "ReplayMemory.h"
#include "Actor.h"
#include "Critic.h"
#include "Decoder.h"
#include "optimizer.h"
#include "Snowy.h"

class MAC {
public:

	Snowy::envirnoment* env;
	Snowy::envirnoment** env_clone;
	int n_env;
	int sample_size;
	Decoder* decoder;
	optimizer* decoder_optim;
	Actor* actor;
	optimizer* actor_optim;
	Critic* critic;
	optimizer* critic_optim;
	ReplayMemory* memory;
	double gamma;

	double* _Decode_;

	double** decoder_port;
	double** critic_port;
	double** actor_port;

	double** critic_grad;
	double** actor_grad;
	double** decoder_grad;

	double** sample_state;
	double* sample_reward;
	double* sample_td;
	double* sampler_prob;

	Snowy::ACTION* sample_action;

	int batch_size;
	int n_batch;
	int total_transition_size;

	bool allocated = false;

	int COUNT = 0;
	MAC(Snowy::envirnoment* snowy,
		Decoder* _decoder,
		optimizer* _decoder_optim,
		Actor* _actor,
		optimizer* _actor_optim,
		Critic* _critic,
		optimizer* _critic_optim) {
		env = snowy;
		decoder = _decoder;
		decoder_optim = _decoder_optim;
		actor = _actor;
		actor_optim = _actor_optim;
		critic = _critic;
		critic_optim = _critic_optim;
		gamma = 0.9;
	}

	void setGamma(double _gamma) {
		gamma = _gamma;
	}

	void allocate(int _n_env, int _sample_sz, int _batch_size) {
		n_env = _n_env;
		batch_size = _batch_size;
		sample_size = _sample_sz;
		total_transition_size = n_env * sample_size;
		n_batch = total_transition_size / batch_size;
		if (total_transition_size % batch_size != 0) {
			cout << "invaild batch size" << endl;
		}
		env_clone = new Snowy::envirnoment * [n_env];
		for (int i = 0; i < n_env; i++) {
			env_clone[i] = new Snowy::envirnoment(env->grid_width, env->grid_height);
			env_clone[i]->setHumdity(env->Humidity);
			env_clone[i]->InitSession();
		}
		memory = new ReplayMemory(decoder->input_size, n_env, sample_size, batch_size);

		decoder->AllocMemory(batch_size);
		actor->AllocMemory(batch_size);
		critic->AllocMemory(batch_size);
		decoder->UseMemory(batch_size);
		actor->UseMemory(batch_size);
		critic->UseMemory(batch_size);

		_Decode_ = km_1d::alloc(decoder->output_size);

		decoder_port = km_2d::alloc(batch_size, decoder->output_size);
		critic_port = km_2d::alloc(batch_size, critic->output_size);
		actor_port = km_2d::alloc(batch_size, actor->output_size);

		decoder_grad = km_2d::alloc(batch_size, decoder->output_size);
		critic_grad = km_2d::alloc(batch_size, critic->output_size);
		actor_grad = km_2d::alloc(batch_size, actor->output_size);

		sampler_prob = km_1d::alloc(sample_size);
		sample_reward = km_1d::alloc(sample_size);
		sample_td = km_1d::alloc(sample_size);
		sample_state = km_2d::alloc(sample_size, decoder->input_size);
		sample_action = new Snowy::ACTION[sample_size];

		env->InitSession();

		allocated = true;
	}

	~MAC() {
		if (allocated) {
			km_1d::free(_Decode_);
			km_2d::free(decoder_port, batch_size);
			km_2d::free(actor_port, batch_size);
			km_2d::free(critic_port, batch_size);
			km_2d::free(decoder_grad, batch_size);
			km_2d::free(actor_grad, batch_size);
			km_2d::free(critic_grad, batch_size);

			km_2d::free(sample_state, sample_size);
			km_1d::free(sample_reward);
			km_1d::free(sample_td);
			km_1d::free(sampler_prob);
			for (int i = 0; i < n_env; i++) {
				delete env_clone[i];
			}
			delete[] env_clone;
			delete[] sample_action;
			delete memory;

		}
	}
	void Train() {
		memory->clear();
		for (int i = 0; i < n_env; i++) {
			MonteCarloSampling(env_clone[i]);

			memory->push(sample_state, (int*)sample_action, sample_td, sampler_prob);
		}
		Fit();
	}
	void Fit() {
	
		memory->shuffle();
		double Loss = 0.0;
		double Policy_loss = 0.0;
		for (int b = 0; b < n_batch; b++) {
			memory->FetchBatchData();

			decoder_optim->zero_grad();
			critic_optim->zero_grad();
			actor_optim->zero_grad();
		
			decoder->charge(memory->batch_s);
			decoder->forward(decoder_port);
			actor->charge(decoder_port);
			actor->forward(actor_port);
			critic->charge(decoder_port);
			critic->forward(critic_port);

			km_2d::fill_zero(actor_grad, batch_size, actor->output_size);
			for (int i = 0; i < batch_size; i++) {
				Loss += (critic_port[i][0] - memory->batch_target[i]) * (critic_port[i][0] - memory->batch_target[i]);
				Policy_loss += ((memory->batch_target[i] - critic_port[i][0]) / actor_port[i][memory->batch_a[i]]) * ((memory->batch_target[i] - critic_port[i][0]) / actor_port[i][memory->batch_a[i]]);
				critic_grad[i][0] = 2.0 * (critic_port[i][0] - memory->batch_target[i]);
				actor_grad[i][memory->batch_a[i]] = -(memory->batch_target[i] - critic_port[i][0]) / memory->batch_prob[i];	/* importance weight <= current_policy(a|s)/old_policy(a|s)	*/
			}

			critic->backward(critic_grad);
			actor->backward(actor_grad);
			km_2d::add(decoder_grad, actor->downstream, critic->downstream, batch_size, decoder->output_size);
			decoder->backward(decoder_grad);

			decoder_optim->step();
			critic_optim->step();
			actor_optim->step();
		}
		cout << " actor loss : " << Policy_loss / total_transition_size << endl;
		cout << " critic loss : " << Loss / total_transition_size << endl;
	}
	void MonteCarloSampling(Snowy::envirnoment* env) {
		double R = 0.0; 
		int head_point = 0;
		if (env->T == TERMINAL) {
			env->InitSession();
		}
		for (int t = 0; t < sample_size; t++) {
			km_1d::copy(sample_state[t], env->S, decoder->input_size);
			decoder->Decode(sample_state[t], _Decode_);
			sample_action[t] = (Snowy::ACTION)actor->StochasticPolicy(_Decode_);
			sampler_prob[t] = actor->prob[sample_action[t]];

			env->interact(sample_action[t]);

			sample_reward[t] = env->R;
			if (t == sample_size - 1 && env->T == NON_TERMINAL) {
				R = critic->Evaluate(_Decode_);
				for (int k = t; k >= head_point; k--) {
					R = sample_reward[k] + gamma * R;
					sample_td[k] = R;
				}
			}
			else if (env->T == TERMINAL) {
				R = 0.0;
				for (int k = t; k >= head_point; k--) {
					R = sample_reward[k] + gamma * R;
					sample_td[k] = R;
				}
				cout << "end : " << env->Scene << endl;
				env->InitSession();
				head_point = t + 1;
			}
		}
	}

};