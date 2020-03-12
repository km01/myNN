#pragma once
#include "Core.h"

class CustomDataSet {
public:
	
	
	double x_1 = 0.5;
	double y_1 = 0.5;

	double x_2 = 0.5;
	double y_2 = -0.5;

	double x_3 = -0.5;
	double y_3 = 0.5;

	double x_4 = -0.5;
	double y_4 = -0.5;

	double stddev = 0.02;
	void getData(darr::v& data) {
		int target_group = km::randint(4);
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
};