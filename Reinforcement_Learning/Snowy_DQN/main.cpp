#include "DQnet_Snowy.h"
/*
	author : kimin Jeong
	kimin353@gmail.com
	kimin353@cau.ac.kr

	this project include <GL/glew.h> , 
						 <GL/freeglut.h>,
						 <GLFW/glfw3.h> 

*/

using namespace std;
int main(int argc, char* argv[]) {
	glutInit(&argc, argv);

	Snowy::envirnoment* snowy = new Snowy::envirnoment(4, 4);
	snowy->setHumdity(0.1);
	DQnet* agent = new DQnet();
	agent->net = new nn(N_Layer(11)); {
		agent->net->layer[0] = new kernel3D(shape(1, 4, 4), shape(1, 2, 2), shape(16, 3, 3), 1, 1);
		agent->net->layer[1] = new ReLU(16*3*3);
		agent->net->layer[2] = new kernel3D(shape(16, 3, 3), shape(16, 2, 2), shape(32, 2, 2), 1, 1);
		agent->net->layer[3] = new ReLU(32*2*2);
		agent->net->layer[4] = new kernel3D(shape(32, 2, 2), shape(32, 2, 2), shape(64, 1, 1), 1, 1);
		agent->net->layer[5] = new ReLU(64);
		agent->net->layer[6] = new fully_connected(64, 32);
		agent->net->layer[7] = new ReLU(32);
		agent->net->layer[8] = new fully_connected(32,16);
		agent->net->layer[9] = new ReLU(16);
		agent->net->layer[10] = new fully_connected(16, 3);
	}agent->publish();

	Shell* package = new Shell();
	package->setEnv(snowy);
	package->setDQnet(agent);
	optimizer* net_optim = new optimizer(agent->net);
	package->setOptim(net_optim);
	int memory_size = 256;
	int batch_size = 8; //  (memory_size % batch_size) must be 0
	package->net_optim->setLearingRate(0.02 / batch_size);


	package->allocate(memory_size, batch_size);
	double gamma = 0.95;
	double e[] = { 0.7, 0.8, 0.85, 0.90, 0.92, 0.95, 0.99};
	int iter[] = { 10, 10, 100, 50, 30 ,30 ,10 };
	for (int i = 0; i < 6; i++) {
		package->agent->setParams(e[i], gamma);
		for (int j = 0; j < iter[i]; j++) {
			cout << " e = "<<e[i]<<", life span : " << package->AgentTraining() << endl;
		}
	}
	package->who_are_you(AGENT);
	package->InitSession();
	package->run();
	delete snowy;
	delete net_optim;
	delete agent->net;
}