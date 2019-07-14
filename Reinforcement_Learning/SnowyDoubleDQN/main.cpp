#include "SnowyDdqnPackage.h"

/*
	author : kimin Jeong
	kimin353@gmail.com
	kimin353@cau.ac.kr
*/

using namespace std;
int main(int argc, char* argv[]) {


	glutInit(&argc, argv);

	Snowy::envirnoment* snowy = new Snowy::envirnoment(6, 6);
	snowy->setHumdity(0.8);
	nn* DQNnetwork = new nn(N_Layer(15)); {


		DQNnetwork->layer[0] = new kernel3D(shape(1, 6, 6), shape(1, 2, 2), shape(18, 5, 5), 1, 1);
		DQNnetwork->layer[1] = new ReLU(DQNnetwork->layer[0]->output_size);
		DQNnetwork->layer[2] = new kernel3D(shape(18, 5, 5), shape(18, 2, 2), shape(24, 4, 4), 1, 1);
		DQNnetwork->layer[3] = new ReLU(DQNnetwork->layer[2]->output_size);
		DQNnetwork->layer[4] = new kernel3D(shape(24, 4, 4), shape(24, 2, 2), shape(36, 3, 3), 1, 1);
		DQNnetwork->layer[5] = new ReLU(DQNnetwork->layer[4]->output_size);
		DQNnetwork->layer[6] = new kernel3D(shape(36, 3, 3), shape(36, 2, 2), shape(48, 2, 2), 1, 1);
		DQNnetwork->layer[7] = new ReLU(DQNnetwork->layer[6]->output_size);
		DQNnetwork->layer[8] = new kernel3D(shape(48, 2, 2), shape(48, 2, 2), shape(64, 1, 1), 1, 1);
		DQNnetwork->layer[9] = new ReLU(DQNnetwork->layer[8]->output_size);
		DQNnetwork->layer[10] = new fully_connected(64, 32);
		DQNnetwork->layer[11] = new ReLU(DQNnetwork->layer[10]->output_size);
		DQNnetwork->layer[12] = new fully_connected(32, 16);
		DQNnetwork->layer[13] = new ReLU(DQNnetwork->layer[12]->output_size);
		DQNnetwork->layer[14] = new fully_connected(16, 3);

	} DQNnetwork->publish();
	//nn* DQNnetwork = new nn("dqnn881399.txt"); DQNnetwork->publish();
	//nn* Targetnetwork = new nn("targetn881399.txt");
	nn* Targetnetwork = (nn*)DQNnetwork->clone();
	Targetnetwork->publish();
	optimizer* dqn_optim = new optimizer(DQNnetwork);
	dqn_optim->use_RMSProp(0.999);
	followeroptimizer* target_optim = new followeroptimizer(Targetnetwork, DQNnetwork);
	int memory_size = 1024;
	int batch_size = 4; //  (memory_size % batch_size) must be 0

	SnowyPackage package;
	double gamma = 0.7;
	package.setDQnet(new DDQnet(DQNnetwork, Targetnetwork, gamma));
	dqn_optim->setLearingRate(0.01/batch_size);
	target_optim->setLearingRate(0.0015);
	package.setEnv(snowy);
	package.setOptim(dqn_optim, target_optim);
	package.allocate(memory_size, batch_size);
	double startE = 0.3;
	double endE = 0.95;
	int iteration = 2000;
	package.Run(USER);
	for (int i = 0; i < iteration; i++) {
		package.agent->SetEpsilon(startE + (i*(endE - startE)) / (iteration));
		cout << "ep - " << i << "(e" << package.agent->e << ") : " << package.AgentTraining() << endl;
		if (i % 200 == 199) {
			package.Run(AGENT);
			DQNnetwork->save_nn("DQN66" + to_string(i) + ".txt");
			Targetnetwork->save_nn("TG66" + to_string(i)+".txt");
		}
	}
	package.agent->SetEpsilon(0.96);
	for (int i = 0; i < 100; i++) {
		cout << "ep -  0.9" << i << " : " << package.AgentTraining() << endl;
		if (i % 100 == 99) {
			package.Run(AGENT);
		}
	}
	package.Run(AGENT);
	
	delete Targetnetwork;
	delete DQNnetwork;
	delete dqn_optim;
	delete target_optim;
	delete snowy;
}