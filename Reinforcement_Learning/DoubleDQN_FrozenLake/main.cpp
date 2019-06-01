#include "DQN.h"
#include "DoubleDQN.h"



#define H 5
#define W 8
UNDER _map[H*W] = { XXXX, XXXX, XXXX, XXXX, XXXX, XXXX, XXXX, XXXX, //1
					XXXX, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, XXXX, //2
					XXXX, HOLE, HOLE, HOLE, LOAD, LOAD, LOAD, XXXX, //10
					XXXX, HOLE, HOLE, HOLE, LOAD, LOAD, GOAL, XXXX, //10
					XXXX, XXXX, XXXX, XXXX, XXXX, XXXX, XXXX, XXXX };

int main() {
	enviroment* frozenLake= new enviroment(_map, H, W, 1.0, -1.0);
	frozenLake->showmap();
	ddqn model(*frozenLake, 0.9, 0.7);

	model.net = new nn(number_of_units(3));
	model.net->layer[0] = new perceptrons(frozenLake->n_state, 100, TANH, true);
	model.net->layer[1] = new perceptrons(100, 50, TANH, true);
	model.net->layer[2] = new perceptrons(50, frozenLake->n_action, IDENTITY, true);
	int batch_size = 10;
	int replaying_step_size = 200;
	model.optim = new optimizer(*model.net);
	model.optim->set_learning_rate(0.05 / batch_size);
	model.optim->use_RMSprop(0.999);
	model.alloc_memorys(replaying_step_size, batch_size);

	STATE start(1, 1);

	double r = 0.0;
	while (true) {
		model.e = 0.97;
		for (int i = 0; i < 2000; i++) {
			r = model.training_run(start);
			cout << r << endl;
		}
		model.e = 0.99;
		for (int i = 0; i < 1000; i++) {
			r = model.training_run(start);
			cout << r << endl;
		}
		if (model.testing_run(start) > 0.9) {
			break;
		}
	}
	model.testing_run(start);
	model.printPath();

	delete model.net;
	delete model.optim;
	delete frozenLake;
}