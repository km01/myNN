#pragma once
#include "actor.h"
#include "critic.h"
#include "decoder.h"
#include "Snowy.h"
#include "optimizer.h"

class VanillaActorCritic {
public:
	Snowy::envirnoment* env;
	
	Decoder* decoder;
	optimizer* decoder_optim;

	Actor* actor;
	optimizer* actor_optim;
	
	Critic* critic;
	optimizer* critic_optim;
	double gamma;


	VanillaActorCritic(Snowy::envirnoment* snowy,
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

	void InitSession() {
		env->InitSession();
		env->interact(Snowy::HOLD);
	}

	int Run() {
		int Count = 0; double loss = 0.0;
		InitSession();
		double V = 0; Snowy::ACTION A = Snowy::ACTION::HOLD;
		while (env->T == NON_TERMINAL) {
			decoder->WithGradDecode(env->S);
			Snowy::ACTION A = (Snowy::ACTION)actor->WithGradStochasticPolicy(decoder->WithGradResult[0]);
			V = critic->WithGradEvaluate_V(decoder->WithGradResult[0]);
			env->interact(A);
			if (env->T == TERMINAL) {
				double Q = env->R;
				critic_optim->zero_grad();
				critic->TDErorrBackprop(Q);
				critic_optim->step();
				actor_optim->zero_grad();
				actor->OnLinePolicyGradientBackprop(Q - V, A);
				actor_optim->step();
				decoder_optim->zero_grad();
				decoder->GetGradBackprop(actor->downstream, critic->downstream);
				decoder_optim->step();
				loss += (V - Q) * (V - Q);
				break;
			}
			else {
				decoder->NoGradDecode(env->S);
				double Q = env->R + gamma * critic->NoGradEvaluate_V(decoder->NoGradResult);
				critic_optim->zero_grad();
				critic->TDErorrBackprop(Q);
				critic_optim->step();
				actor_optim->zero_grad();
				actor->OnLinePolicyGradientBackprop(Q - V , A);
				actor_optim->step();

				decoder_optim->zero_grad();
				decoder->GetGradBackprop(actor->downstream, critic->downstream);
				decoder_optim->step();
				
				loss += (V - Q) * (V - Q);
			}
			Count++;
		}
		cout << " loss : " << loss / Count << endl;
		return Count;
	}

	int Run2() { // deterministic policy ( argmax ).  
		int Count = 0; double loss = 0.0;
		InitSession();
		double V = 0; Snowy::ACTION A = Snowy::ACTION::HOLD;
		while (env->T == NON_TERMINAL) {
			decoder->WithGradDecode(env->S);
			Snowy::ACTION A = (Snowy::ACTION)actor->WithGradDeterministicPolicy(decoder->WithGradResult[0]);
			V = critic->WithGradEvaluate_V(decoder->WithGradResult[0]);
			env->interact(A);
			if (env->T == TERMINAL) {
				double Q = env->R;
				critic_optim->zero_grad();
				critic->TDErorrBackprop(Q);
				critic_optim->step();
				actor_optim->zero_grad();
				actor->OnLinePolicyGradientBackprop((Q - V) * actor->batch_policy_prob[0][A], A); 
															//actor->batch_policy_prob[0][A] : importance weight
				actor_optim->step();
				decoder_optim->zero_grad();
				decoder->GetGradBackprop(actor->downstream, critic->downstream);
				decoder_optim->step();
				loss += (V - Q) * (V - Q);
				break;
			}
			else {
				decoder->NoGradDecode(env->S);
				double Q = env->R + gamma * critic->NoGradEvaluate_V(decoder->NoGradResult);
				critic_optim->zero_grad();
				critic->TDErorrBackprop(Q);
				critic_optim->step();
				actor_optim->zero_grad();
				actor->OnLinePolicyGradientBackprop((Q - V)*actor->batch_policy_prob[0][A], A);
				actor_optim->step();

				decoder_optim->zero_grad();
				decoder->GetGradBackprop(actor->downstream, critic->downstream);
				decoder_optim->step();

				loss += (V - Q) * (V - Q);
			}
			Count++;
		}
		cout << " loss : " << loss / Count << endl;
		return Count;
	}
};