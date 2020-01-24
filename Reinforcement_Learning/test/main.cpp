#include "McSamplingDoubleDqn.h"
#include "SnowyLauncher.h"

int main(int argc, char* argv[]) {

	glutInit(&argc, argv);
	snowy::Snowy env(h_w(6, 4), 0.2);
	Sequential root;

	root.push(new Conv2D(c_h_w(1, h_w(6, 4)),  h_w(2, 2), c_h_w(8, h_w(5, 3)), h_w(1, 1)));
	root.push(new ReLU(8 * 5 * 3));
	root.push(new Conv2D(c_h_w(8, h_w(5, 3)), h_w(2, 2), c_h_w(12, h_w(4, 2)), h_w(1, 1)));
	root.push(new ReLU(12 * 4 * 2));
	root.push(new Conv2D(c_h_w(12, h_w(4, 2)), h_w(2, 2), c_h_w(18, h_w(3, 1)), h_w(1, 1)));
	root.push(new ReLU(18*3));

	Sequential actor;
	actor.push(new Dense(root.n_out, 32));
	actor.push(new ReLU(32));
	actor.push(new Dense(32, 24));
	actor.push(new ReLU(24));
	actor.push(new Dense(24, 3));
	actor.push(new softmax(3));

	Sequential critic;
	critic.push(new Dense(root.n_out, 32));
	critic.push(new ReLU(32));
	critic.push(new Dense(32, 12));
	critic.push(new ReLU(12));
	critic.push(new Dense(12, 3));

	DuelingNetwork_Wrapper agent(&root, &actor, &critic);
	agent.setCache(1);

	Optimizer optim(agent);
	optim.setLearningRate(0.00005);

	/* */
	double gamma = 0.9;
	int sampling_buffer_size = 200;
	/* */
	McSamplingActorCritic pkg(&agent, &env, &optim, gamma, sampling_buffer_size);

	int iter = 10000;
	for (int t = 0; t < iter; t++) {
		cout << "-----------------" << t << "----------------" << endl;
		pkg.mc_sampling();
		pkg.sample_learn();
		if (t % 2000 == 1999) {
			SnowyLauncher launcher(&env, 10);
			launcher.Run(&agent);
		}
	}
}
