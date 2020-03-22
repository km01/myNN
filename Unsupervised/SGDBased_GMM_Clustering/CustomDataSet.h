#pragma once
#include "Core.h"

class CustomDataSet {
public:


	double x_1 = 1.0;
	double y_1 = 1.0;

	double x_2 = 1.0;
	double y_2 = -1.0;

	double x_3 = -1.0;
	double y_3 = 1.0;

	double x_4 = -1.0;
	double y_4 = -1.0;

	double stddev = 0.2;
	
	void getOneData(darr::v& data) {
		static int target_group = 0;
		target_group = km::randint(4);
		if (target_group == 0) {
			data[0] = stddev * STD(rEngine) + x_1;
			data[1] = stddev * STD(rEngine) + y_1;
		}
		else if (target_group == 1) {
			data[0] = stddev * STD(rEngine) + x_2;
			data[1] = stddev * STD(rEngine) + y_2;
		}
		else if (target_group == 2) {
			data[0] = stddev * STD(rEngine) + x_3;
			data[1] = stddev * STD(rEngine) + y_3;
		}
		else {
			data[0] = stddev * STD(rEngine) + x_4;
			data[1] = stddev * STD(rEngine) + y_4;
		}
	}

	void getData(darr::v2D& data, const int& n_data) {
		for (int i = 0; i < n_data; i++) {
			getOneData(data[i]);
		}
	}

};