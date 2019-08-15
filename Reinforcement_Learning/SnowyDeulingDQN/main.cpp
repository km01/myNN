#include "nn.h"
#include "optimizer.h"
#include "SnowySolution.h"
/*
	author : kimin Jeong
	kimin353@gmail.com
	kimin353@cau.ac.kr
*/

using namespace std;
/*	

	my setting :
	
	max(A(s,a)) expect to be 0
	
	V(s) expect to be max(Q(s,-))

	Q(s,a) = V(s) + A(s,a) - max(A(s,-))  

	Target Q(s,a) = r + gamma * V(s') 
				  = r + gamma * max(Target Q(s,-))

	Target A(s,a) = Target Q(s,a) - max(Target Q(s,-))


*/
int main(int argc, char* argv[]) {

	glutInit(&argc, argv);
	Snowy::envirnoment* snowy = new Snowy::envirnoment(4, 5);
	snowy->setHumdity(0.9);
	DeulNet* net = new DeulNet(n_Layer(8), n_Layer(5), n_Layer(7));
	{
		net->root->layer[0] = new Kernel3D(shape(1, 5, 4), shape(1, 3, 2), shape(12, 3, 3), 1, 1);
		net->root->layer[1] = new ReLU(12 * 3 * 3);
		net->root->layer[2] = new Kernel3D(shape(12, 3, 3), shape(12, 2, 2), shape(16, 2, 2), 1, 1);
		net->root->layer[3] = new ReLU(16 * 2 * 2);
		net->root->layer[4] = new Kernel3D(shape(16, 2, 2), shape(16, 2, 2), shape(18, 1, 1), 1, 1);
		net->root->layer[5] = new ReLU(18);
		net->root->layer[6] = new fully_connected(18, 18);
		net->root->layer[7] = new ReLU(18);

		{	net->val_branch->layer[0] = new fully_connected(18, 18);
			net->val_branch->layer[1] = new ReLU(18);
			net->val_branch->layer[2] = new fully_connected(18, 3);
			net->val_branch->layer[3] = new ReLU(3);
			net->val_branch->layer[4] = new fully_connected(3, 1);	}

		{	net->adv_branch->layer[0] = new fully_connected(18, 18);
			net->adv_branch->layer[1] = new ReLU(18);
			net->adv_branch->layer[2] = new fully_connected(18, 12);
			net->adv_branch->layer[3] = new ReLU(12);
			net->adv_branch->layer[4] = new fully_connected(12, 8);
			net->adv_branch->layer[5] = new ReLU(8);
			net->adv_branch->layer[6] = new fully_connected(8, 3);	}

	} net->publish();

	net->SetGamma(0.8);
	optimizer* root_optim = new optimizer(net->root);
	optimizer* value_optim = new optimizer(net->val_branch);
	optimizer* advantage_optim = new optimizer(net->adv_branch);

	SnowySolution* package = new SnowySolution(snowy, net, root_optim, value_optim, advantage_optim);
	int memory_size = 1024;
	int batch_size = 4; //  (memory_size % batch_size) must be 0
	root_optim->use_RMSProp(0.999);
	root_optim->setLearingRate(0.003 / batch_size);
	value_optim->use_RMSProp(0.999);
	value_optim->setLearingRate(0.003 / batch_size);
	advantage_optim->use_RMSProp(0.999);
	advantage_optim->setLearingRate(0.003 / batch_size);

	package->allocate(memory_size, batch_size);
	double startE = 0.3;
	double endE = 0.95;
	int iteration = 5000;
	for (int i = 0; i < iteration; i++) {
		package->agent->SetEpsilon(startE + (i * (endE - startE)) / (iteration));
		cout << "ep - " << i << "(e" << package->agent->e << ") : " << package->AgentTraining() << endl;
		if (i % 1001 == 0) {
			SnowyGraphicLauncher tester(snowy, 20);
			tester.Run(net);
		}
	}
	for (int i = 0; i < iteration; i++) {
		cout << "ep - " << i << "(e" << package->agent->e << ") : " << package->AgentTraining() << endl;
		if (i % 101 == 0) {
			SnowyGraphicLauncher tester(snowy, 20);
			tester.Run(net);
		}
	}
	delete net;
	delete package;
	delete snowy;
}