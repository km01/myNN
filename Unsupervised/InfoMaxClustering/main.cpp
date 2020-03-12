#include "CustomDataSet.h"
#include "Unit.h"
#include "Optim.h"
#include "LatentSpace.h"


double ce_loss(darr::v& param_grad, darr::v const& param, darr::v const& target, const int& len) {
	double loss = 0.0;
	for (int i = 0; i < len; i++) {
		loss += -target[i] * log(param[i]);
		param_grad[i] = -target[i] / param[i];
	}
	return loss;
}

double negative_mutual_information(darr::v& p_of_y_grad, darr::v& p_of_y_bar_x_grad, darr::v const& p_of_y, darr::v const& p_of_y_bar_x, const int& len) {
	double loss = 0.0;
	for (int i = 0; i < len; i++) {
		loss += p_of_y[i] * log(p_of_y[i]) - p_of_y_bar_x[i] * log(p_of_y_bar_x[i]);
		p_of_y_grad[i] = 1.0 + log(p_of_y[i]);
		p_of_y_bar_x_grad[i] = -(1.0 + log(p_of_y_bar_x[i]));
	}
	return loss;
}

int main() {

	CustomDataSet set;	
	int n_data = 60000;
	darr::v2D data = darr::alloc(n_data, 2);
	for (int i = 0; i < n_data; i++) {
		set.getData(data[i]);
	}

	int n_category = 4;

	
	double learning_rate = 0.1;
	
	Sequential classifier;
	classifier.append(new Dense(2, 8));
	classifier.append(new ReLU(8));
	classifier.append(new Dense(8, 8));
	classifier.append(new ReLU(8));
	classifier.append(new Dense(8, 4));
	classifier.append(new ReLU(4));
	classifier.append(new Dense(4, 4));
	classifier.append(new ReLU(4));
	classifier.append(new Dense(4, n_category));
	classifier.append(new Softmax(n_category));
	classifier.setCache(1);
	Optimizer optim(classifier);
	optim.setLearningRate(learning_rate);


	Categorical_Distribution prior(n_category);
	darr::v sum = darr::alloc(n_category);
	darr::v p_y_grad = darr::alloc(n_category);
	darr::v p_y_bar_x = darr::alloc(n_category);
	darr::v p_y_bar_x_grad = darr::alloc(n_category);

	double entropy_y = 0.0;
	double entropy_y_given_x = 0.0;
	for (int iter = 0; iter < 100; iter++) {
		km::setZero(sum, n_category);
		entropy_y_given_x = 0.0;
		entropy_y = 0.0;

		for (int i = 0; i < n_data; i++) {
			classifier.charge(data[i], 0);
			classifier.forward(p_y_bar_x, 0);
			for (int k = 0; k < n_category; k++) {	
				sum[k] += p_y_bar_x[k];
				entropy_y_given_x += -p_y_bar_x[k] * log(p_y_bar_x[k]);
			}
		}
		entropy_y_given_x /= n_data;
		for (int k = 0; k < n_category; k++) {
			entropy_y += -((sum[k]) / n_data) * log(((sum[k]) / n_data));
		}
		cout << "iteration : " << iter + 1 << endl;
		cout << "entropy_y : " << entropy_y << endl;
		cout << "entropy_y_given_x : " << entropy_y_given_x << endl;
		cout << "mutual information:" << entropy_y - entropy_y_given_x << endl << endl;
		for (int i = 0; i < n_data; i++) {
			optim.zero_grad();
			classifier.charge(data[i], 0);
			classifier.forward(p_y_bar_x, 0);

			for (int k = 0; k < n_category; k++) {
				p_y_bar_x_grad[k] = (-log(n_data) + log(sum[k]) - log(p_y_bar_x[k])) / (double)n_data;
			}
			classifier.backward(p_y_bar_x_grad, 0);
			optim.step();
		}
	}
	while (true) {

		cin >> classifier.in_cache[0][0];
		cin >> classifier.in_cache[0][1];
		classifier.forward(p_y_bar_x, 0);
		cout << p_y_bar_x[0] << " " << p_y_bar_x[1] << " " << p_y_bar_x[2] << " " << p_y_bar_x[3] << endl;
	}

	darr::free(data, n_data);
}