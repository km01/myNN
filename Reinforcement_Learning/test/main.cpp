#include "McSamplingDoubleDqn.h"
#include "SnowyLauncher.h"

int main(int argc, char* argv[]) {

	glutInit(&argc, argv);
	snowy::Snowy env(h_w(4, 4), 0.12);

	Sequential root;

	//root.push(new Conv2D(c_h_w(1, h_w(5, 4)), h_w(2, 2), c_h_w(12, h_w(4, 3)), h_w(1, 1)));
	//root.push(new ReLU(12 * 4 * 3));
	//root.push(new Conv2D(c_h_w(12, h_w(4, 3)), h_w(2, 2), c_h_w(12, h_w(3, 2)), h_w(1, 1)));
	//root.push(new ReLU(12 * 3 * 2));
	//root.push(new Conv2D(c_h_w(12, h_w(3, 2)), h_w(2, 2), c_h_w(18, h_w(2, 1)), h_w(1, 1)));
	//root.push(new ReLU(36));

	root.push(new Dense(16, 128));
	root.push(new ReLU(128));
	root.push(new Dense(128, 96));
	root.push(new ReLU(96));
	root.push(new Dense(96, 48));
	root.push(new ReLU(48));
	root.push(new Dense(48, 32));
	root.push(new ReLU(32));


	Sequential actor;
	actor.push(new Dense(root.n_out, 24));
	actor.push(new ReLU(24));
	actor.push(new Dense(24, 12));
	actor.push(new ReLU(12));
	actor.push(new Dense(12, 3));

	Sequential critic;
	critic.push(new Dense(root.n_out, 24));
	critic.push(new ReLU(24));
	critic.push(new Dense(24, 12));
	critic.push(new ReLU(12));
	critic.push(new Dense(12, 3));

	DuelingNetwork_Wrapper agent(&root, &actor, &critic);
	agent.setCache(1);

	Optimizer optim(agent);
	optim.setLearningRate(0.001);

	/* */
	double gamma = 0.95;
	int sampling_buffer_size = 100;
	/* */
	McSamplingDoubleDqn pkg(&agent, &env, &optim, gamma, sampling_buffer_size);

	double start_epsilon = 0.7;
	int exploration_time = 2000;
	double end_epsilon = 0.999;
	
	for (int t = 0; t < exploration_time; t++) {
		pkg.mc_sampling2(start_epsilon + t * (end_epsilon - start_epsilon) / (double)(exploration_time));
		pkg.sample_learn();
	}
	for (int t = 0; t < 2000; t++) {
		pkg.mc_sampling2(end_epsilon);
		pkg.sample_learn();

	}
}
