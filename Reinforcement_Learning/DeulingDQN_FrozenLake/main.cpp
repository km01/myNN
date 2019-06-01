#include "DQN.h"
#include "DoubleDQN.h"
#include "DeulingDQN.h"


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
	deuling_dqn *model = new deuling_dqn(*frozenLake, 0.9, 0.7);

	model->net = new deul_net(number_of_units(2), number_of_units(2), number_of_units(2));

	model->net->root_net->layer[0] = new perceptrons(frozenLake->n_state, 100, TANH, true);
	model->net->root_net->layer[1] = new perceptrons(100, 50, TANH, true);

	model->net->value_net->layer[0] = new perceptrons(50, 10 , TANH, true);
	model->net->value_net->layer[1] = new perceptrons(10, 1, SIGMOID, true);

	model->net->action_net->layer[0] = new perceptrons(50, 50, TANH, true);
	model->net->action_net->layer[1] = new perceptrons(50, frozenLake->n_action, IDENTITY,true);

	int batch_size = 10;
	int replaying_step_size = 100;
	cout << "ok1" << endl;
	model->alloc_memorys(replaying_step_size, batch_size);
	cout << "ok2" << endl;

	model->optim = new DeulNetOptimizer(*model->net);
	cout << "ok3" << endl;

	model->optim->set_learning_rate(0.05 / batch_size);
	model->optim->use_rmsprop(0.999);
	cout << "ok3" << endl;

	STATE start(1, 1);

	double r = 0;
	model->e = 0.90;
	for (int i = 0; i < 4000; i++) {
		r = model->training_run(start);
		//cout << r << endl;
	}
	model->e = 0.95;
	for (int i = 0; i < 4000; i++) {
		r = model->training_run(start);
		//cout << r << endl;
	}
	model->e = 0.99;
	for (int i = 0; i < 3000; i++) {
		r = model->training_run(start);
		//cout << r << endl;
	}
	cout << model->testing_run(start) << endl;
	cout << model->net->value_net->single_output[0] << endl;
	cout << model->net->value_net->single_argmax << endl;

	model->printPath();
	
	delete model->net;
	delete model->optim;
	delete frozenLake;
	delete model;
}