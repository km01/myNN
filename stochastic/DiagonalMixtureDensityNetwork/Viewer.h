#pragma once
#include "Jm.h"
#include "Core.h"


class Canvas {
public:

	float dH, dW;
	float time = 0.0f;
	float left_most, right_most, upper_most, under_most;
	float frame_scalar = 0.001f;
	int n_data = 0;
	double x_max = DBL_MIN, y_max = DBL_MIN, x_min = DBL_MAX, y_min = DBL_MAX;
	vector<pair<double, double>> data;
	Canvas() {
		dH = 0.0f; dW = 0.0f; time = 0.0f; frame_scalar = 0.0f;
		left_most = -0.9f; right_most = 0.9f; upper_most = 0.9f; under_most = -0.9;
	}

	void plot(km::DarrList& point_x, km::DarrList& point_y, const int& n_data) {
		for (int i = 0; i < n_data; i++) {
			if (x_max < point_x[i][0]) {
				x_max = point_x[i][0];
			}
			if (y_max < point_y[i][0]) {
				y_max = point_y[i][0];
			}
			if (x_min > point_x[i][0]) {
				x_min = point_x[i][0];
			}
			if (y_min > point_y[i][0]) {
				y_min = point_y[i][0];
			}
			this->n_data = n_data;
			data.push_back(pair<double, double>(point_x[i][0], point_y[i][0]));
		}
	}

	void draw_data() {
		double half = (x_max - x_min)*0.5;
		if ((y_max - y_min)*0.5 > half) {
			half = (y_max - y_min) * 0.5;
		}
		double ratio = (right_most - left_most) / (half * 2.0);
		double scaled_half = (right_most - left_most) * 0.5;
		glColor3dv(Colors::red.data);
		glPointSize(1.0f);
		glBegin(GL_POINTS);
		double x = 0, y = 0;
		for (int i = 0; i < n_data; i++) {
			x = data[i].first;
			y = data[i].second;
			x += half;
			y -= half;
			x *= ratio;
			y *= ratio;
			x -= scaled_half;
			y += scaled_half;
			glVertex2f(x, y);
		}
		glEnd();
	}

};

class Viewer : public Game {
public:


	Canvas canv;

	Viewer(km::DarrList& x, km::DarrList& y, const int& n_data) {
		canv.plot(x,y, n_data);
	}

	~Viewer() {

	}

	void show() {
		THE_END = false;
		run();
	}

	void update() override {
		canv.draw_data();
	}
};
