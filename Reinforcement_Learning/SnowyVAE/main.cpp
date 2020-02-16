#include "SnowyLauncher.h"
#include "Optim.h"
#include "MonteCarloSampler.h"
#include "AgentModel.h"


int main(int argc, char* argv[]) {

	glutInit(&argc, argv);
	snowy::Snowy env(h_w(5, 4), 0.2);
	SnowyLauncher launcher(&env, 20);
	launcher.Run();
	int sample_buff_size = 512;


	int state_size = 32;

	Sequential root;
	root.append(new Dense(state_size, 32));
	root.append(new ReLU(32));
	root.append(new Dense(32, 24));
	root.append(new ReLU(24));

	Sequential actor;
	actor.append(new Dense(24, 12));
	actor.append(new ReLU(12));
	actor.append(new Dense(12, 3));
	actor.append(new Softmax(3));

	Sequential critic;
	critic.append(new Dense(24, 12));
	critic.append(new ReLU(12));
	critic.append(new Dense(12, 3));

	ActorCritic agent(root, actor, critic);
	
	GaussianDensityNetwork encoder;
	encoder.append(new Conv2D(c_h_w(3, h_w(5, 4)), h_w(2, 2), c_h_w(14, h_w(4, 3)), h_w(1, 1)));
	encoder.append(new ReLU(14 * 4 * 3));
	encoder.append(new Conv2D(c_h_w(14, h_w(4, 3)), h_w(2, 2), c_h_w(16, h_w(3, 2)), h_w(1, 1)));
	encoder.append(new ReLU(16 * 3 * 2));
	encoder.append(new Dense(16 * 3 * 2, state_size + state_size));

	Sequential decoder;
	decoder.append(new Dense(state_size, 12 * 3 * 2));
	decoder.append(new LeakyReLU(12 * 3 * 2, 0.1));
	decoder.append(new Conv2DTransposed(c_h_w(12, h_w(3, 2)), h_w(2, 2), c_h_w(8, h_w(4, 3)), h_w(1, 1)));
	decoder.append(new LeakyReLU(8 * 4 * 3, 0.1));
	decoder.append(new Conv2DTransposed(c_h_w(8, h_w(4, 3)), h_w(2, 2), c_h_w(3, h_w(5, 4)), h_w(1, 1)));


	AgentModel model(&encoder, &agent, &decoder, 3, 20);
	model.setCache(1);
	Optimizer optim(model); optim.setLearningRate(0.001);
	McSamplingSnowyActorCritic2 mc(env.observationSize(), 0.9, sample_buff_size);
	int n_iteration = 10001;
	for (int iter = 0; iter < n_iteration; iter++) {
		double nll = 0.0;
		double kld = 0.0;
		double gain = 0.0;
		mc.mc_sampling(env, model);
		for (int i = 0; i < sample_buff_size; i++) {
			optim.zero_grad();
			model.calculate_gradient(gain, nll, kld, mc.ob_buff[i], mc.target_buff[i], mc.action_buff[i], mc.action_prob_buff[i], mc.next_ob_buff[i]);
			optim.step();
		}
		cout << "-------" << iter << "--------" << endl;
		cout << "gain:" << gain / (double)sample_buff_size << "		nll:" << nll / (double)sample_buff_size << "		kld:" << kld / (double)sample_buff_size << endl;
		if (iter % 100 == 0) {
			SnowyLauncher launcher(&env, 100);
			launcher.Run(&model);
		}
	}
}
