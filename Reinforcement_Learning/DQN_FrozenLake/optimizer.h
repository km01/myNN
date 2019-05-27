#pragma once
#include "unit.h"
#include "model.h"
enum TERM { GRADIENT_DESCENT, MOMENTUM, RMSPROP, ADAPIVE_MOMENTUM };
class unit_manager {
public:
	UNIT u_type;
	TERM term;
	double beta1;
	double beta1_inv;
	double beta2;
	double beta2_inv;
	double curr_beta1;
	double curr_beta2;
	double denumerater1;
	double denumerater2;
	virtual void use_momentum(double beta) {}
	virtual void use_RMSprop(double beta) {}
	virtual void use_adaptive_momentum(double momentum_beta, double rms_beta) {}
	virtual void reset() {}
	virtual void feed_grad(const double& learning_rate) {}
	virtual ~unit_manager() {}
};
class perceptron_manager : public unit_manager {
public:
	perceptrons* trainee;
	double* momentum;
	double* momentum_bias;
	double* rms;
	double* rms_bias;


	bool use_bias;
	int weight_len;
	int bias_len;
	perceptron_manager() {}
	perceptron_manager(perceptrons* target) {
		u_type = PERCEPTRONS;
		trainee = target;
		weight_len = trainee->weight_len;

		bias_len = trainee->bias_len;
		use_bias = trainee->use_bias;
		term = GRADIENT_DESCENT;
	}
	virtual void reset() {
		if (term == MOMENTUM) {
			curr_beta1 = 1.0;
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = 0.0;
				}
			}
		}
		else if (term == RMSPROP) {
			curr_beta2 = 1.0;
			for (int i = 0; i < weight_len; i++) {
				rms[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					rms_bias[i] = 0.0;
				}
			}
		}
		else if (term == ADAPIVE_MOMENTUM) {
			curr_beta1 = 1.0;
			curr_beta2 = 1.0;
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = 0.0;
				rms[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = 0.0;
					rms_bias[i] = 0.0;
				}
			}
		}
	}
	virtual void use_momentum(double momentum_beta) {
		term = MOMENTUM;
		beta1 = momentum_beta;
		beta1_inv = 1.0 - beta1;

		momentum = new double[trainee->weight_len];
		if (use_bias == true) {
			momentum_bias = new double[trainee->bias_len];
		}
		curr_beta1 = 1.0;
		reset();
	}

	virtual void use_rmsprop(double RMSprop_beta) {
		term = RMSPROP;
		beta2 = RMSprop_beta;
		beta2_inv = 1.0 - beta2;

		rms = new double[trainee->weight_len];
		if (use_bias == true) {
			rms_bias = new double[trainee->bias_len];
		}
		curr_beta2 = 1.0;
		reset();
	}

	virtual void use_adaptive_momentum(double momentum_beta, double RMSprop_beta) {

		beta1 = momentum_beta;
		beta1_inv = 1.0 - beta1;
		term = ADAPIVE_MOMENTUM;
		beta2 = RMSprop_beta;
		beta2_inv = 1.0 - beta2;
		momentum = new double[trainee->weight_len];
		rms = new double[trainee->weight_len];
		if (use_bias == true) {
			rms_bias = new double[trainee->bias_len];
			momentum_bias = new double[trainee->bias_len];
		}
		curr_beta1 = 1.0;
		curr_beta2 = 1.0;
		reset();
	}

	virtual void feed_grad(const double& learning_rate) {

		if (term == MOMENTUM) {

			curr_beta1 *= beta1;
			denumerater1 = (1.0 - curr_beta1);
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = beta1 * momentum[i] + beta1_inv * trainee->gradients_weight[i];
				trainee->weight[i] -= learning_rate * momentum[i] / denumerater1;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = beta1 * momentum_bias[i] + beta1_inv * trainee->gradients_bias[i];
					trainee->bias[i] -= learning_rate * momentum_bias[i] / denumerater1;
				}
			}
		}
		else if (term == RMSPROP) {
			curr_beta2 *= beta2;
			denumerater2 = 1.0 - curr_beta2;
			for (int i = 0; i < weight_len; i++) {
				rms[i] = beta2 * rms[i] + beta2_inv * (trainee->gradients_weight[i])* (trainee->gradients_weight[i]);
				trainee->weight[i] -= (learning_rate / sqrt((rms[i] / denumerater2) + epsilon)) * trainee->gradients_weight[i];
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					rms_bias[i] = beta2 * rms_bias[i] + beta2_inv * (trainee->gradients_bias[i])* (trainee->gradients_bias[i]);
					trainee->bias[i] -= (learning_rate / sqrt((rms_bias[i] / denumerater2) + epsilon)) * trainee->gradients_bias[i];
				}
			}
		}
		else if (term == ADAPIVE_MOMENTUM) {
			curr_beta1 *= beta1;
			curr_beta2 *= beta2;
			denumerater1 = 1.0 - curr_beta1;
			denumerater2 = 1.0 - curr_beta2;
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = beta1 * momentum[i] + beta1_inv * trainee->gradients_weight[i];
				rms[i] = beta2 * rms[i] + beta2_inv * (trainee->gradients_weight[i])* (trainee->gradients_weight[i]);
				trainee->weight[i] -= (learning_rate / sqrt((rms[i] / denumerater2) + epsilon)) * (momentum[i] / denumerater1);
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = beta1 * momentum_bias[i] + beta1_inv * trainee->gradients_bias[i];
					rms_bias[i] = beta2 * rms_bias[i] + beta2_inv * (trainee->gradients_bias[i])* (trainee->gradients_bias[i]);
					trainee->bias[i] -= (learning_rate / sqrt((rms_bias[i] / denumerater2) + epsilon)) * (momentum_bias[i] / denumerater1);
				}
			}
		}
		else {
			trainee->feed_grad_itself(learning_rate);
		}
	}
	~perceptron_manager() {
		if (term == MOMENTUM || term == ADAPIVE_MOMENTUM) {
			if (use_bias == true) {
				delete[] momentum_bias;
			}
			delete[] momentum;
		}

		if (term == RMSPROP || term == ADAPIVE_MOMENTUM) {
			if (use_bias == true) {
				delete[] rms_bias;
			}
			delete[] rms;
		}
	}
};
class bn_perceptron_manager : public unit_manager {
public:
	bn_perceptrons* trainee;
	double* momentum;
	double* momentum_bias;
	double* rms;
	double* rms_bias;

	double* momentum_bn_scaler;
	double* momentum_bn_shift;
	double* rms_bn_scaler;
	double* rms_bn_shift;
	int bn_group;

	bool use_bias;
	int weight_len;
	int bias_len;
	bn_perceptron_manager() {}

	bn_perceptron_manager(bn_perceptrons* target) {
		u_type = BN_PERCEPTRONS;
		trainee = target;
		weight_len = trainee->weight_len;
		bn_group = trainee->normalizer->nb_group;
		bias_len = trainee->bias_len;
		use_bias = trainee->use_bias;
		term = GRADIENT_DESCENT;
	}
	virtual void reset() {
		if (term == MOMENTUM) {
			curr_beta1 = 1.0;
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = 0.0;
				}
			}
			for (int i = 0; i < bn_group; i++) {
				momentum_bn_scaler[i] = 0.0;
				momentum_bn_shift[i] = 0.0;
			}
		}
		else if (term == RMSPROP) {
			curr_beta2 = 1.0;
			for (int i = 0; i < weight_len; i++) {
				rms[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					rms_bias[i] = 0.0;
				}
			}
			for (int i = 0; i < bn_group; i++) {
				rms_bn_scaler[i] = 0.0;
				rms_bn_shift[i] = 0.0;
			}
		}
		else if (term == ADAPIVE_MOMENTUM) {
			curr_beta1 = 1.0;
			curr_beta2 = 1.0;
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = 0.0;
				rms[i] = 0.0;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = 0.0;
					rms_bias[i] = 0.0;
				}
			}
			for (int i = 0; i < bn_group; i++) {
				momentum_bn_scaler[i] = 0.0;
				momentum_bn_shift[i] = 0.0;
				rms_bn_scaler[i] = 0.0;
				rms_bn_shift[i] = 0.0;
			}
		}
	}
	virtual void use_momentum(double momentum_beta) {
		term = MOMENTUM;
		beta1 = momentum_beta;
		beta1_inv = 1.0 - beta1;

		momentum = new double[trainee->weight_len];

		momentum_bn_scaler = new double[bn_group];
		momentum_bn_shift = new double[bn_group];

		if (use_bias == true) {
			momentum_bias = new double[trainee->bias_len];
		}
		curr_beta1 = 1.0;
		reset();
	}

	virtual void use_rmsprop(double RMSprop_beta) {
		term = RMSPROP;
		beta2 = RMSprop_beta;
		beta2_inv = 1.0 - beta2;

		rms = new double[trainee->weight_len];
		rms_bn_scaler = new double[bn_group];
		rms_bn_shift = new double[bn_group];

		if (use_bias == true) {
			rms_bias = new double[trainee->bias_len];
		}
		curr_beta2 = 1.0;
		reset();
	}

	virtual void use_adaptive_momentum(double momentum_beta, double RMSprop_beta) {

		beta1 = momentum_beta;
		beta1_inv = 1.0 - beta1;
		term = ADAPIVE_MOMENTUM;
		beta2 = RMSprop_beta;
		beta2_inv = 1.0 - beta2;
		momentum = new double[trainee->weight_len];
		rms = new double[trainee->weight_len];


		momentum_bn_scaler = new double[bn_group];
		momentum_bn_shift = new double[bn_group];
		rms_bn_scaler = new double[bn_group];
		rms_bn_shift = new double[bn_group];


		if (use_bias == true) {
			rms_bias = new double[trainee->bias_len];
			momentum_bias = new double[trainee->bias_len];
		}
		curr_beta1 = 1.0;
		curr_beta2 = 1.0;
		reset();
	}

	virtual void feed_grad(const double& learning_rate) {

		if (term == MOMENTUM) {

			curr_beta1 *= beta1;
			denumerater1 = (1.0 - curr_beta1);
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = beta1 * momentum[i] + beta1_inv * trainee->gradients_weight[i];
				trainee->weight[i] -= learning_rate * momentum[i] / denumerater1;
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = beta1 * momentum_bias[i] + beta1_inv * trainee->gradients_bias[i];
					trainee->bias[i] -= learning_rate * momentum_bias[i] / denumerater1;
				}
			}
			for (int i = 0; i < bn_group; i++) {
				momentum_bn_scaler[i] = beta1 * momentum_bn_scaler[i] + beta1_inv * trainee->normalizer->gradients_scale[i];
				trainee->normalizer->learned_scale[i] -= learning_rate * momentum_bn_scaler[i] / denumerater1;
				momentum_bn_shift[i] = beta1 * momentum_bn_shift[i] + beta1_inv * trainee->normalizer->gradients_shift[i];
				trainee->normalizer->learned_shift[i] -= learning_rate * momentum_bn_shift[i] / denumerater1;
			}
		}
		else if (term == RMSPROP) {
			curr_beta2 *= beta2;
			denumerater2 = 1.0 - curr_beta2;
			for (int i = 0; i < weight_len; i++) {
				rms[i] = beta2 * rms[i] + beta2_inv * (trainee->gradients_weight[i])* (trainee->gradients_weight[i]);
				trainee->weight[i] -= (learning_rate / sqrt((rms[i] / denumerater2) + epsilon)) * trainee->gradients_weight[i];
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					rms_bias[i] = beta2 * rms_bias[i] + beta2_inv * (trainee->gradients_bias[i])* (trainee->gradients_bias[i]);
					trainee->bias[i] -= (learning_rate / sqrt((rms_bias[i] / denumerater2) + epsilon)) * trainee->gradients_bias[i];
				}
			}
			for (int i = 0; i < bn_group; i++) {
				rms_bn_scaler[i] = beta2 * rms_bn_scaler[i] + beta2_inv * (trainee->normalizer->gradients_scale[i])* (trainee->normalizer->gradients_scale[i]);
				trainee->normalizer->learned_scale[i] -= (learning_rate / sqrt((rms_bn_scaler[i] / denumerater2) + epsilon))*trainee->normalizer->gradients_scale[i];
				rms_bn_shift[i] = beta2 * rms_bn_shift[i] + beta2_inv * (trainee->normalizer->gradients_shift[i])* (trainee->normalizer->gradients_shift[i]);
				trainee->normalizer->learned_shift[i] -= (learning_rate / sqrt((rms_bn_shift[i] / denumerater2) + epsilon))*trainee->normalizer->gradients_shift[i];
			}
		}
		else if (term == ADAPIVE_MOMENTUM) {
			curr_beta1 *= beta1;
			curr_beta2 *= beta2;
			denumerater1 = 1.0 - curr_beta1;
			denumerater2 = 1.0 - curr_beta2;
			for (int i = 0; i < weight_len; i++) {
				momentum[i] = beta1 * momentum[i] + beta1_inv * trainee->gradients_weight[i];
				rms[i] = beta2 * rms[i] + beta2_inv * (trainee->gradients_weight[i])* (trainee->gradients_weight[i]);
				trainee->weight[i] -= (learning_rate / sqrt((rms[i] / denumerater2) + epsilon)) * (momentum[i] / denumerater1);
			}
			if (use_bias == true) {
				for (int i = 0; i < bias_len; i++) {
					momentum_bias[i] = beta1 * momentum_bias[i] + beta1_inv * trainee->gradients_bias[i];
					rms_bias[i] = beta2 * rms_bias[i] + beta2_inv * (trainee->gradients_bias[i])* (trainee->gradients_bias[i]);
					trainee->bias[i] -= (learning_rate / sqrt((rms_bias[i] / denumerater2) + epsilon)) * (momentum_bias[i] / denumerater1);
				}
			}
			for (int i = 0; i < bn_group; i++) {
				momentum_bn_scaler[i] = beta1 * momentum_bn_scaler[i] + beta1_inv * trainee->normalizer->gradients_scale[i];
				rms_bn_scaler[i] = beta2 * rms_bn_scaler[i] + beta2_inv * (trainee->normalizer->gradients_scale[i]) * (trainee->normalizer->gradients_scale[i]);
				trainee->normalizer->learned_scale[i] -= (learning_rate / sqrt((rms_bn_scaler[i] / denumerater2) + epsilon)) * (momentum_bn_scaler[i] / denumerater1);

				momentum_bn_shift[i] = beta1 * momentum_bn_shift[i] + beta1_inv * trainee->normalizer->gradients_shift[i];
				rms_bn_shift[i] = beta2 * rms_bn_shift[i] + beta2_inv * (trainee->normalizer->gradients_shift[i]) * (trainee->normalizer->gradients_shift[i]);
				trainee->normalizer->learned_shift[i] -= (learning_rate / sqrt((rms_bn_shift[i] / denumerater2) + epsilon)) * (momentum_bn_shift[i] / denumerater1);
			}

		}
		else {
			trainee->feed_grad_itself(learning_rate);
		}
	}
	~bn_perceptron_manager() {
		if (term == MOMENTUM || term == ADAPIVE_MOMENTUM) {
			if (use_bias == true) {
				delete[] momentum_bias;
			}
			delete[] momentum;
			delete[] momentum_bn_scaler;
			delete[] momentum_bn_shift;
		}

		if (term == RMSPROP || term == ADAPIVE_MOMENTUM) {
			if (use_bias == true) {
				delete[] rms_bias;
			}
			delete[] rms;
			delete[] rms_bn_scaler;
			delete[] rms_bn_shift;
		}
	}
};
class optimizer {

private:
	double learning_rate;
public:
	nn* trainee;
	unit_manager** manager;
	int n_layer;
	TERM term;
	int time;
	optimizer(nn& target) {
		trainee = &target;
		n_layer = trainee->n_layer;
		term = GRADIENT_DESCENT;
		manager = new unit_manager*[n_layer];
		for (int i = 0; i < n_layer; i++) {
			if (trainee->layer[i]->unit_type == PERCEPTRONS) {
				manager[i] = new perceptron_manager((perceptrons*)trainee->layer[i]);
			}

			else if (trainee->layer[i]->unit_type == BN_PERCEPTRONS) {
				manager[i] = new bn_perceptron_manager((bn_perceptrons*)trainee->layer[i]);
			}

		}
	}
	void use_momentum(double beta) {
		term = MOMENTUM;
		for (int i = 0; i < n_layer; i++) {
			manager[i]->use_momentum(beta);
		}
		reset();
	}
	void use_RMSprop(double beta) {
		term = RMSPROP;
		for (int i = 0; i < n_layer; i++) {
			manager[i]->use_RMSprop(beta);
		}
		reset();
	}
	void use_adaptive_momentum(double momentum_beta, double rms_beta) {
		term = ADAPIVE_MOMENTUM;
		for (int i = 0; i < n_layer; i++) {
			manager[i]->use_adaptive_momentum(momentum_beta, rms_beta);
		}
		reset();
	}
	void reset() {
		time = 0;
		for (int i = 0; i < n_layer; i++) {
			manager[i]->reset();
		}
	}
	void set_learning_rate(double _learing_rate) {
		learning_rate = _learing_rate;
	}

	void step() {
		for (int i = 0; i < n_layer; i++) {
			manager[i]->feed_grad(learning_rate);
		}
	}

	~optimizer() {
		for (int i = 0; i < n_layer; i++) {
			delete manager[i];
		}
		delete[] manager;
	}
};