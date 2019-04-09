#pragma once
#include "mynn_core.h"
class cross_entropy {
public:
	
	double batch_loss_prime(double** batch_port, double** batch_pred, int* batch_label, int batch_size, int len) {
		double mean_loss = 0.0;
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				if (batch_label[m] == i) {
					batch_port[m][i] = batch_pred[m][i] - 1.0;
					mean_loss -= log(batch_pred[m][i]);
				}
				else {

					batch_port[m][i] = batch_pred[m][i];
				}
			}
		}
		return mean_loss / (double)batch_size;
	}
	double single_loss_prime(double* single_port, double* single_pred, int single_label,int len) {
		double mean_loss = 0.0;
		for (int i = 0; i < len; i++) {
			if (single_label == i) {
				single_port[i] = single_pred[i] - 1.0;
				mean_loss -= log(single_pred[i]);
			}
			else {
				single_port[i] = single_pred[i];
			}
		}
		return mean_loss;
	}
	double batch_loss(double** batch_pred, int* batch_label, int batch_size, int len) {
		double mean_loss = 0.0;
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < len; i++) {
				if (batch_label[m] == i) {
					mean_loss -= log(batch_pred[m][i]);
				}
			}
		}
		return mean_loss / (double)batch_size;
	}
	double single_loss(double* single_pred, int single_label, int len) {
		return -log(single_pred[single_label]);
	}
};