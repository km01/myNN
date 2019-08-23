#include "MAC.h"
#include "SnowyLauncher.h"
using namespace std;
int main(int argc, char* argv[]) {

	glutInit(&argc, argv);
	Snowy::envirnoment* snowy = new Snowy::envirnoment(4, 5);
	snowy->setHumdity(0.3);

	Decoder* decoder = new Decoder(n_Layer(6)); {
		decoder->layer[0] = new Kernel3D(shape(1, 5, 4), shape(1, 3, 2), shape(12, 3, 3), 1, 1);
		decoder->layer[1] = new ReLU(12 * 3 * 3);
		decoder->layer[2] = new Kernel3D(shape(12, 3, 3), shape(12, 2, 2), shape(16, 2, 2), 1, 1);
		decoder->layer[3] = new ReLU(16 * 2 * 2);
		decoder->layer[4] = new Kernel3D(shape(16, 2, 2), shape(16, 2, 2), shape(24, 1, 1), 1, 1);
		decoder->layer[5] = new ReLU(24);
	}; decoder->publish();


	Actor* actor = new Actor(n_Layer(6)); {
		actor->layer[0] = new fully_connected(24, 24);
		actor->layer[1] = new ReLU(24);
		actor->layer[2] = new fully_connected(24, 24);
		actor->layer[3] = new ReLU(24);
		actor->layer[4] = new fully_connected(24, 3);
		actor->layer[5] = new Softmax(3);
	}   actor->publish();


	Critic* critic = new Critic(n_Layer(5)); {
		critic->layer[0] = new fully_connected(24, 24);
		critic->layer[1] = new ReLU(24);
		critic->layer[2] = new fully_connected(24, 12);
		critic->layer[3] = new ReLU(12);
		critic->layer[4] = new fully_connected(12, 1);
	}   critic->publish();
	
	optimizer* actor_optim = new optimizer(actor);
	optimizer* critic_optim = new optimizer(critic);
	optimizer* decoder_optim = new optimizer(decoder);

	int batch_size = 16;
	int n_env = 8;
	int sample_size = 32;
	decoder_optim->setLearingRate(0.01 / batch_size);
	actor_optim->setLearingRate(0.01 / batch_size);
	critic_optim->setLearingRate(0.01 / batch_size);
	MAC* model = new MAC(snowy, decoder, decoder_optim, actor, actor_optim, critic, critic_optim);
	model->allocate(n_env, sample_size, batch_size);
	model->setGamma(0.99);
	for (int iter = 0; iter < 10; iter++) {
		for (int i = 0; i < 1000; i++) {
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