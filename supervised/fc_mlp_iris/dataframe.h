#pragma once
#include "mynn_core.h"
#include <cstdlib>
void shuffle(int* order_list, int size) {
	int rand_idx;
	int temp;
	for (int i = 0; i < size; i++) {
		order_list[i] = i;
	}
	for (int iter = 0; iter < 10; iter++) {
		for (int i = 0; i < size; i++) {
			rand_idx = std::rand() % size;
			temp = order_list[rand_idx];
			order_list[rand_idx] = order_list[i];
			order_list[i] = temp;
		}
	}
}
class DataSet {
public:
	int n_rows;
	int n_cols;
	double *max;
	double *min;
	double** data;
	DataSet() {}
	DataSet(string path) {
		ifstream fin(path);
		if (fin.fail()) {
			cout << "file opening failure" << endl;
		}
		else {
			string head;
			getline(fin, head);
			n_rows = 0;
			n_cols = 1;
			for (int i = 0; i < (int)head.length(); i++) {
				if (head[i] == ',') {
					n_cols++;
				}
			}
			while (fin.good()) {
				string line;
				getline(fin, line);
				if (!fin.good()) {
					break;
				}
				n_rows++;
			}
			fin.close();
			fin.open(path);
			getline(fin, head);
			data = new double*[n_rows];
			max = new double[n_cols];
			min = new double[n_cols];

			int LINE = 0;
			while (fin.good()) {
				string line;
				getline(fin, line);
				if (!fin.good()) {
					break;
				}
				data[LINE] = new double[n_cols];
				int count = 0;
				string value_buffer;

				for (int i = 0; i <= (int)line.length(); i++) {
					if (line[i] == ',' || i == line.length()) {
						data[LINE][count] = atof(value_buffer.c_str());
						value_buffer.clear();
						count++;
					}
					else {
						value_buffer.push_back(line[i]);
					}
				}
				LINE++;
			}
			fin.close();
		}
	}
	void print() {
		for (int i = 0; i < n_rows; i++) {
			for (int j = 0; j < n_cols; j++) {
				cout << data[i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
	void min_max_scaling() {

		for (int c = 0; c < n_cols; c++) {
			double MIN = data[0][c];
			double MAX = data[0][c];
			for (int n = 1; n < n_rows; n++) {
				if (MIN > data[n][c]) {
					MIN = data[n][c];
				}
				if (MAX < data[n][c]) {
					MAX = data[n][c];
				}
			}
			max[c] = MAX;
			min[c] = MIN;
			for (int n = 0; n < n_rows; n++) {
				data[n][c] = ((data[n][c] - min[c]) / (max[c] - min[c])) - 0.5;
			}
		}
	}
	~DataSet() {
		for (int i = 0; i < n_rows; i++) {
			delete[] data[i];
		}
		delete[] data;
		delete[] max;
		delete[] min;
	}
};

class Labelset {
public:
	int n_rows;
	int* label;
	Labelset() {}
	Labelset(string path) {
		ifstream fin(path);
		if (fin.fail()) {
			cout << "file opening failure" << endl;
		}
		else {
			string head;
			getline(fin, head);
			n_rows = 0;

			while (fin.good()) {
				string line;
				getline(fin, line);
				if (!fin.good()) {
					break;
				}
				n_rows++;
			}
			fin.close();
			fin.open(path);
			getline(fin, head);
			label = new int[n_rows];

			int LINE = 0;
			while (fin.good()) {
				string line;
				getline(fin, line);
				if (!fin.good()) {
					break;
				}
				int count = 0;
				string value_buffer;

				for (int i = 0; i <= (int)line.length(); i++) {
					if (line[i] == ',' || i == line.length()) {
						label[LINE] = atoi(value_buffer.c_str());
						value_buffer.clear();
						count++;
						break;
					}
					else {
						value_buffer.push_back(line[i]);
					}
				}
				LINE++;
			}
			int min = 10000;
			for (int i = 0; i < n_rows; i++) {
				if (min > label[i]) {
					min = label[i];
				}
			}
			for (int i = 0; i < n_rows; i++) {
				label[i] = label[i] - min;
			}
			fin.close();
		}
	}

	void print() {
		for (int i = 0; i < n_rows; i++) {
			cout << label[i] << endl;
		}
		cout << endl;
	}


	~Labelset() {
		delete[] label;
	}
};
class batch_loader {
public:
	DataSet* x_data;
	Labelset* y_data;
	double** mini_x;
	int* mini_y;
	int batch_size;
	int set_size;
	int x_feature_size;
	bool has_x;
	bool has_y;
	bool batch_memory_allocated;

	int nb_batch;
	int curr_order;

	int* order_list;
	batch_loader(DataSet* target_data, Labelset* label_data) {
		x_data = target_data;
		y_data = label_data;
		if (x_data->n_rows != y_data->n_rows) {
			cout << "?" << endl;
		}
		set_size = x_data->n_rows;
		x_feature_size = x_data->n_cols;
		has_x = true;
		has_y = true;
		batch_memory_allocated = false;

		order_list = new int[set_size];
		for (int i = 0; i < set_size; i++) {
			order_list[i] = i;
		}
		shuffle_order();
	}
	batch_loader(DataSet* target_data) {
		x_data = target_data;
		set_size = x_data->n_rows;
		x_feature_size = x_data->n_cols;
		has_x = true;
		has_y = false;
		batch_memory_allocated = false;
	}
	void shuffle_order() {
		shuffle(order_list, set_size);
	}
	void alloc_batch_storage(int new_batch_size) {
		if (batch_memory_allocated) {
			if (has_x) {
				km_2d::free(mini_x, batch_size);
			}
			if (has_y) {
				delete[] mini_y;
			}
		}
		batch_size = new_batch_size;
		curr_order = -1;
		nb_batch = set_size / batch_size;

		if (has_x) {
			mini_x = km_2d::alloc(batch_size, x_feature_size);
			batch_memory_allocated = true;
		}
		if (has_y) {
			mini_y = new int[batch_size];
			batch_memory_allocated = true;
		}
		next_batch();
	}
	void next_batch() {
		curr_order++;
		int step = curr_order * batch_size;
		for (int m = 0; m < batch_size; m++) {
			for (int i = 0; i < x_feature_size; i++) {
				mini_x[m][i] = x_data->data[order_list[step + m]][i];
			}
			if (has_y) {
				mini_y[m] = y_data->label[order_list[m + step]];
			}
		}
		if (curr_order >= nb_batch - 1) {
			curr_order = -1;
			shuffle_order();
		}
	}
	void cur_batch_show() {
		int step = curr_order * batch_size;
		cout << "batch_idx : " << curr_order << endl;
		for (int i = 0; i < batch_size; i++) {
			cout << order_list[step + i] << " : ";
			for (int j = 0; j < x_feature_size; j++) {
				cout << mini_x[i][j] << " ";
			}
			if (has_y) {
				cout << "y : " << mini_y[i] << endl;
			}
		}
	}
	~batch_loader() {
		if (batch_memory_allocated) {
			if (has_x) {
				km_2d::free(mini_x, batch_size);
			}
			if (has_y) {
				delete[] mini_y;
			}
		}
		delete[] order_list;
	}
};