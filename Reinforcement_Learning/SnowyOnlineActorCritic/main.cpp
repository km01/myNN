#include "SnowyLauncher.h"
#include "VanillaActorCritic.h"

/*
	author : kimin Jeong
	kimin353@gmail.com
	kimin353@cau.ac.kr
*/

using namespace std;
int main(int argc, char* argv[]) {

	glutInit(&argc, argv);
	Snowy::envirnoment* snowy = new Snowy::envirnoment(4, 4);
	snowy->setHumdity(0.7);

	/*---------------------------------------------------
		Decoder decodes 'environment state' to 'actor&critic input' 

													*/
	Decoder* decoder = new Decoder(n_Layer(6)); {
		decoder->layer[0] = new Kernel3D(shape(1, 4, 4), shape(1, 2, 2), shape(12, 3, 3), 1, 1);
		decoder->layer[1] = new ReLU(12 * 3 * 3);
		decoder->layer[2] = new Kernel3D(shape(12, 3, 3), shape(12, 2, 2), shape(12, 2, 2), 1, 1);
		decoder->layer[3] = new ReLU(12 * 2 * 2);
		decoder->layer[4] = new Kernel3D(shape(12, 2, 2), shape(12, 2, 2), shape(24, 1, 1), 1, 1);
		decoder->layer[5] = new ReLU(24);
	}; decoder->publish(); decoder->AllocMemory(1); decoder->UseMemory(1);

	/*---------------------------------------------------
			actor estimates ¥ð(a|s).
			
			gradient = ¥ð(a|s) * ( - A(s,a) )
			
			A(s,a) = Q(s,a) - V(s) 
				   = ( r + gamma * V(s') -  V(s))

			V(s), V(s') = critic output
														*/
	Actor* actor = new Actor(n_Layer(6)); {
		actor->layer[0] = new fully_connected(24, 18);
		actor->layer[1] = new ReLU(18);
		actor->layer[2] = new fully_connected(18, 18);
		actor->layer[3] = new ReLU(18);
		actor->layer[4] = new fully_connected(18, 3);
		actor->layer[5] = new Softmax(3);
	}   actor->publish(); actor->AllocMemory(1); actor->UseMemory(1);
	/*--------------------------------------------------*/

	/*---------------------------------------------------
		critic estimates V(s) := E_a~¥ð_[r+V(s')]
														*/

	Critic* critic = new Critic(n_Layer(5)); {
		critic->layer[0] = new fully_connected(24, 24);
		critic->layer[1] = new ReLU(24);
		critic->layer[2] = new fully_connected(24, 12);
		critic->layer[3] = new ReLU(12);
		critic->layer[4] = new fully_connected(12, 1);
	}   critic->publish(); critic->AllocMemory(1); critic->UseMemory(1);
	/*--------------------------------------------------*/

	/*---------------------------------------------------
		  set optimizer									*/ 
	
	optimizer* actor_optim = new optimizer(actor);
	optimizer* critic_optim = new optimizer(critic);
	optimizer* decoder_optim = new optimizer(decoder);
	decoder_optim->setLearingRate(0.005);
	decoder_optim->use_Momentum(0.9);
	actor_optim->setLearingRate(0.005);
	actor_optim->use_Momentum(0.9);
	critic_optim->setLearingRate(0.005);
	critic_optim->use_Momentum(0.9);
	/*--------------------------------------------------*/

	/*---------------------------------------------------
			  set ActorCritic							
			  Online Learning. 
														*/
	VanillaActorCritic* model = new VanillaActorCritic(snowy,
													   decoder,
													   decoder_optim,
													   actor,
													   actor_optim,
													   critic,
													   critic_optim);
	model->setGamma(0.9);
	/*--------------------------------------------------*/
	for (int i = 0; i < 1000; i++) {
		cout << "sess " << i << " : " << model->Run2() << endl;
		if (i % 200 == 0) {
			SnowyLauncher game(snowy, actor, 0);
			game.Run();
		}
	}
	for (int i = 0; i < 100; i++) {
		cout << "sess " << i << " : " << model->Run2() << endl;
		if (i % 10 == 0) {
			SnowyLauncher game(snowy, actor, 0);
			game.Run();
		}
	}
	delete actor;
	delete actor_optim;
	delete critic;
	delete critic_optim;
	delete decoder;
	delete decoder_optim;
	delete snowy;
}