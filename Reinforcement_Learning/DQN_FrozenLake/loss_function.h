#pragma once
#include "mynn_core.h"
class MSELoss {
public:

	double batch_loss_prime(double** batch_port, double** batch_pred, double** batch_label, int batch_size, int len) {
		double mean_loss = 0.0;
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				batch_port[m][i] = 2.0 * (batch_pred[m][i] - batch_label[m][i]);
				mean_loss += (batch_pred[m][i] - batch_label[m][i])*(batch_pred[m][i] - batch_label[m][i]);
			}
		}
		return mean_loss / (double)batch_size;
	}
	
	double single_loss_prime(double* single_port, double* single_pred, double* single_label, int len) {
		double mean_loss = 0.0;
		for (int i = 0; i < len; i++) {
			single_port[i] = 2.0 * (single_pred[i] - single_label[i]);
			mean_loss += (single_pred[i] - single_label[i])* (single_pred[i] - single_label[i]);
		}
		return mean_loss;
	}

	double batch_loss(double** batch_pred, double** batch_label, int batch_size, int len) {
		double mean_loss = 0.0;
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				mean_loss += (batch_pred[m][i] - batch_label[m][i])*(batch_pred[m][i] - batch_label[m][i]);
			}
		}
		return mean_loss / (double)batch_size;
	}
	double single_loss(double* single_pred, double*  single_label, int len) {
		double mean_loss = 0.0;
		for (int i = 0; i < len; i++) {
			mean_loss += (single_pred[i] - single_label[i])* (single_pred[i] - single_label[i]);
		}
		return mean_loss;
	}
};
