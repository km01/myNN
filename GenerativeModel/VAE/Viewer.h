#pragma once
#include "Jm.h"
#include "core.h"

class GrayCanvas {
public:

	float dH, dW;
	float time = 0.0f;
	float left_most, right_most, upper_most, under_most;
	float frame_scalar = 0.001f;
	double** imgs = nullptr;
	int img_w = 0;	int img_h = 0;
	int n_row = 0;	int n_col = 0;
	float term = 0.01f;
	GrayCanvas(int col, int row, int img_ht, int img_wd, double** img) {
		dH = 0.0f; dW = 0.0f; time = 0.0f; frame_scalar = 0.0f;
		left_most = -0.9f; right_most = 0.9f; upper_most = 0.9f; under_most = -0.9;
		img_w = img_wd;
		img_h = img_ht;
		n_row = row;
		n_col = col;
		imgs = km::alloc(n_row * n_col, img_w * img_h);
		for (int i = 0; i < n_row * n_col; i++) {
			setOneImg(img[i], i);
		}
	}
	~GrayCanvas() {
		km::free(imgs, n_row * n_col);
	}

	void setOneImg(double* one_img, int id) {
		double min = one_img[0];
		double max = one_img[0];
		for (int j = 0; j < img_w * img_h; j++) {
			if (min > one_img[j]) {
				min = one_img[j];
			}
			if (max < one_img[j]) {
				max = one_img[j];
			}
		}
		if (max > min) {
			for (int j = 0; j < img_w * img_h; j++) {
				imgs[id][j] = (one_img[j] - min) / (max - min);
			}
		}
		else {
			km::setZero(imgs[id], img_w * img_h);
		}
	}

	void drawOneImg(int id, double left, double right, double upper, double under) {
		double dw = (right - left) / img_w;
		double dh = (upper - under) / img_h;
		for (int c = 0; c < img_h; c++) {
			for (int r = 0; r < img_w; r++) {
				glColor3d(imgs[id][c * img_w + r], imgs[id][c * img_w + r], imgs[id][c * img_w + r]);
				glPushMatrix(); glBegin(GL_POLYGON); {
					glVertex2f(left + dw * r, upper - dh * c);
					glVertex2f(left + dw * (r + 1), upper - dh * c);
					glVertex2f(left + dw * (r + 1), upper - dh * (c + 1));
					glVertex2f(left + dw * r, upper - dh * (c + 1));
				} glEnd(); glPopMatrix();
			}
		}
	}
	void visualize() {
		double step = (right_most - left_most) / n_row;
		for (int i = 0; i < n_col; i++) {
			for (int j = 0; j < n_row; j++) {
				drawOneImg(i * n_row + j, left_most + j * step, left_most + (j + 1) * step - term, upper_most - (i * step), upper_most - (i + 1) * step + term);
			}
		}
	}
};

class Viewer : public Game {
public:


	GrayCanvas* canv = nullptr;


	Viewer(int col, int row, int img_ht, int img_wd, double** img) {
		canv = new GrayCanvas(col, row, img_ht, img_wd, img);
	}

	~Viewer() {
		if (canv != nullptr) {
			delete canv;
		}
	}

	void show() {
		THE_END = false;
		run();
	}

	void update() override {
		canv->visualize();
	}

	void drawBitmapText(char* str, float x, float y, float z) {
		glColor3dv(Colors::black.data);
		glRasterPos3f(x, y, z);
		while (*str) {
			glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_10, *str);
			str++;
		}
	}
};