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
	snowy->setHumdity(0.3);



	/*---------------------------------------------------
			actor estimates policy(a|s) 
							PolicyGradient로 -(Q(s,a) - V(s))/policy(a|s)를 사용. 

														*/
	Actor* actor = new Actor(n_Layer(14)); {
		actor->layer[0] = new Kernel3D(shape(1, 4, 4), shape(1, 2, 2), shape(16, 3, 3), 1, 1);
		actor->layer[1] = new ReLU(16 * 3 * 3);
		actor->layer[2] = new Kernel3D(shape(16, 3, 3), shape(16, 2, 2), shape(24, 2, 2), 1, 1);
		actor->layer[3] = new ReLU(24 * 2 * 2);
		actor->layer[4] = new Kernel3D(shape(24, 2, 2), shape(24, 2, 2), shape(32, 1, 1), 1, 1);
		actor->layer[5] = new ReLU(32);
		actor->layer[6] = new fully_connected(32, 32);
		actor->layer[7] = new ReLU(32);
		actor->layer[8] = new fully_connected(32, 32);
		actor->layer[9] = new ReLU(32);
		actor->layer[10] = new fully_connected(32, 32);
		actor->layer[11] = new ReLU(32);
		actor->layer[12] = new fully_connected(32, 3);
		actor->layer[13] = new Softmax(3);
	}   actor->publish(); actor->AllocMemory(1); actor->UseMemory(1);
	/*--------------------------------------------------*/

	/*---------------------------------------------------
		critic estimates Q(s,a) := E[r+V(s')]
								 = E[r + E[Q(s',a')].
								 Bellan Expectation equation을 사용.
														*/

	Critic* critic = new Critic(n_Layer(9)); {
		critic->layer[0] = new Kernel3D(shape(1, 4, 4), shape(1, 2, 2), shape(12, 3, 3), 1, 1);
		critic->layer[1] = new ReLU(12 * 3 * 3);
		critic->layer[2] = new Kernel3D(shape(12, 3, 3), shape(12, 2, 2), shape(16, 2, 2), 1, 1);
		critic->layer[3] = new ReLU(16 * 2 * 2);
		critic->layer[4] = new Kernel3D(shape(16, 2, 2), shape(16, 2, 2), shape(18, 1, 1), 1, 1);
		critic->layer[5] = new ReLU(18);
		critic->layer[6] = new fully_connected(18, 12);
		critic->layer[7] = new ReLU(12);
		critic->layer[8] = new fully_connected(12, 3);
	}   critic->publish(); critic->AllocMemory(1); critic->UseMemory(1);
	/*--------------------------------------------------*/

	/*---------------------------------------------------
		  set optimizer									*/
	optimizer* actor_optim = new optimizer(actor);
	optimizer* critic_optim = new optimizer(critic);
	actor_optim->setLearingRate(0.03);
	critic_optim->setLearingRate(0.02);
	/*--------------------------------------------------*/

	/*---------------------------------------------------
			  set ActorCritic							
			  Online Learning. 
														*/
	VanillaActorCritic* model = new VanillaActorCritic(snowy, actor, critic, actor_optim, critic_optim);
	model->setGamma(0.5);
	/*--------------------------------------------------*/
	for (int i = 0; i < 10000; i++) {
		cout << "sess " << i << " : " << model->Run() << endl;
		if (i % 1001 == 0) {
			SnowyLauncher Game(snowy, actor, 50);
			Game.run();
		}
	}

	delete actor;
	delete actor_optim;
	delete critic;
	delete critic_optim;
	delete snowy;
}