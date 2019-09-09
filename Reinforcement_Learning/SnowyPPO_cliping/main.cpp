#include "PPO.h"
#include "SnowyLauncher.h"

using namespace std;
int main(int argc, char* argv[]) {

	glutInit(&argc, argv);
	Snowy::environment* snowy = new Snowy::environment(4, 6);
	snowy->setHumdity(0.4);
	Decoder* decoder = new Decoder(n_Layer(6)); {
		decoder->layer[0] = new Kernel3D(shape(1, 6, 4), shape(1, 2, 2), shape(4, 5, 3), 1, 1);
		decoder->layer[1] = new ReLU(4 * 5 * 3);
		decoder->layer[2] = new Kernel3D(shape(4, 5, 3), shape(4, 2, 2), shape(6, 4, 2), 1, 1);
		decoder->layer[3] = new ReLU(6 * 4 * 2);
		decoder->layer[4] = new Kernel3D(shape(6, 4, 2), shape(6, 2, 2), shape(8, 3, 1), 1, 1);
		decoder->layer[5] = new ReLU(24);
	}; decoder->publish();


	Actor* actor = new Actor(n_Layer(6)); {
		actor->layer[0] = new fully_connected(24, 16);
		actor->layer[1] = new ReLU(16);
		actor->layer[2] = new fully_connected(16, 16);
		actor->layer[3] = new ReLU(16);
		actor->layer[4] = new fully_connected(16, 3);
		actor->layer[5] = new Softmax(3);
	}   actor->publish();


	Critic* critic = new Critic(n_Layer(5)); {
		critic->layer[0] = new fully_connected(24, 16);
		critic->layer[1] = new ReLU(16);
		critic->layer[2] = new fully_connected(16, 12);
		critic->layer[3] = new ReLU(12);
		critic->layer[4] = new fully_connected(12, 1);
	}   critic->publish();

	optimizer* actor_optim = new optimizer(actor);
	optimizer* critic_optim = new optimizer(critic);
	optimizer* decoder_optim = new optimizer(decoder);

	int batch_size = 16;
	int n_env = 8;
	int sample_size = 128;
	decoder_optim->setLearingRate(0.05 / batch_size);
	actor_optim->setLearingRate(0.05 / batch_size);
	critic_optim->setLearingRate(0.05 / batch_size);
	PPO* model = new PPO(snowy, decoder, decoder_optim, actor, actor_optim, critic, critic_optim);
	model->allocate(n_env, sample_size, batch_size);
	model->setGamma(0.9);
	model->setClipingEPSILON(0.25);
	for (int iter = 0; iter < 10; iter++) {
		for (int i = 0; i < 500; i++) {
			model->Train();
		}
		SnowyLauncher game(snowy, decoder, actor, 10);
		game.run();
	}
	delete actor;
	delete actor_optim;
	delete critic;
	delete critic_optim;
	delete decoder;
	delete decoder_optim;
	delete snowy;
	delete model;
}