#pragma once
#include "actor.h"
#include "critic.h"
#include "Snowy.h"
#include "optimizer.h"

class VanillaActorCritic {
public:
	Actor* actor;
	Critic* critic;
	optimizer* actor_optim;
	optimizer* critic_optim;
	Snowy::envirnoment* world;
	double gamma;


	VanillaActorCritic(Snowy::envirnoment* snowy, Actor* _actor, Critic* _critic, optimizer* _actor_optim, optimizer* _critic_optim) {
		world = snowy;
		actor = _actor;
		critic = _critic;
		actor_optim = _actor_optim;
		critic_optim = _critic_optim;
		gamma = 0.9;
	}

	void setGamma(double _gamma) {
		gamma = _gamma;
	}

	void InitSession() {
		world->InitSession();
		world->interact(Snowy::HOLD);
	}

	int Run() {
		int Count = 0;
		InitSession();
		Snowy::ACTION A = (Snowy::ACTION)actor->WithGradPolicy(world->current_S);


		double loss = 0.0;
		double Q = 0.0;
	
		double V = 0.0;
		while (world->current_T == NON_TERMINAL) {


			critic->WithGradEvaluate_A(Q, V, world->current_S, actor->batch_policy_prob[0], A); /* estimate Q(s,a), V(s) */
			/*cout << "****************************************************************************************************************"<< endl;
			cout <<  " prob : "<<actor->batch_policy_prob[0][0] << " " << actor->batch_policy_prob[0][1] << " " << actor->batch_policy_prob[0][2] << " " << endl;
			cout << " critic : " << critic->batch_q[0][0] << " " << critic->batch_q[0][1] << " " << critic->batch_q[0][2] << " " << endl;*/

			actor_optim->zero_grad();
			actor->OnLinePolicyGradientBackprop(Q-V, A); /*  PolicyGradient = -(G /policy(a|s))	*/
			actor_optim->step();

			world->interact(A);

			if (world->current_T == TERMINAL) {
				double G_Target = world->current_R;
				critic_optim->zero_grad();
				critic->TDErorrBackprop(G_Target, A);
				critic_optim->step();
				loss += (Q - G_Target) * (Q - G_Target);
				break;
			}
			else {
				Snowy::ACTION next_a = (Snowy::ACTION)actor->WithGradPolicy(world->current_S); /* estimate policy(s') */
				double G_Target = world->current_R + gamma * critic->NoGradEvaluate_V(world->current_S, actor->batch_policy_prob[0]);

				critic_optim->zero_grad();
				critic->TDErorrBackprop(G_Target, A);
				critic_optim->step();
				A = next_a;
				loss += (Q - G_Target) * (Q - G_Target);
			}
			Count++;
		}
		cout << " loss : " << loss / Count << endl;
		return Count;
	}
};