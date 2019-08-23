#pragma once
#include "Snowy.h"
#include "Jm.h"


class SnowyVisualizer {
public:
	Snowy::envirnoment* snowy;
	float dH, dW;
	float time = 0.0f;
	float left_most, right_most, upper_most, under_most;
	float frame_scalar = 0.02f;

	SnowyVisualizer() {
		snowy = nullptr; dH = 0.0f; dW = 0.0f; time = 0.0f; frame_scalar = 0.0f;
		left_most = 0.0f; right_most = 0.0f; upper_most = 0.0f; under_most = 0.0f;
	}

	SnowyVisualizer(Snowy::envirnoment* _snowy) {
		snowy = _snowy;
		left_most = -snowy->grid_width * frame_scalar;
		right_most = snowy->grid_width * frame_scalar;
		under_most = -snowy->grid_height * frame_scalar;
		upper_most = snowy->grid_height * frame_scalar;
		dH = (upper_most - under_most) / (float)snowy->grid_height;
		dW = (right_most - left_most) / (float)snowy->grid_width;
	}

	void draw_Frame() {
		glLineWidth(1.f);
		glColor3dv(Colors::silver.data);
		glBegin(GL_POLYGON); {
			glVertex2d(left_most, upper_most);
			glVertex2d(right_most, upper_most);
			glVertex2d(right_most, under_most);
			glVertex2d(left_most, under_most);
		} glEnd();
	}

	void draw_world() {
		for (int h = 0; h < snowy->grid_height; h++) {
			for (int w = 0; w < snowy->grid_width; w++) {
				if (snowy->S[h * snowy->grid_width + w] == Snowy::ID::_void_) {
					glColor3dv(Colors::silver.data);
				}
				else if (snowy->S[h * snowy->grid_width + w] == Snowy::ID::_snow_) {
					glColor3dv(Colors::white.data);
				}
				else if (snowy->S[h * snowy->grid_width + w] == Snowy::ID::_man_) {
					glColor3dv(Colors::red.data);
				}
				else {
					glColor3dv(Colors::yellow.data);
				}
				glPushMatrix(); glBegin(GL_POLYGON); {
					glVertex2f(left_most + dW * w, under_most + dH * h);
					glVertex2f(left_most + dW * (w + 1), under_most + dH * h);
					glVertex2f(left_most + dW * (w + 1), under_most + dH * (h + 1));
					glVertex2f(left_most + dW * w, under_most + dH * (h + 1));
				} glEnd(); glPopMatrix();
			}
		}
	}
	void draw_Cell() {
		glLineWidth(0.1f);
		glColor3dv(Colors::black.data);
		for (int i = 0; i < snowy->grid_width; i++) {
			glBegin(GL_LINES); {
				glVertex2f(left_most + i * (dW), upper_most);
				glVertex2f(left_most + i * (dW), under_most);

			} glEnd();
		}
		for (int i = 0; i < snowy->grid_height; i++) {
			glBegin(GL_LINES); {
				glVertex2f(left_most, under_most + i * (dH));
				glVertex2f(right_most, under_most + i * (dH));
			} glEnd();
		}
	}
	void drawBitmapText(char* str, float x, float y, float z) {
		glColor3dv(Colors::black.data);
		glRasterPos3f(x, y, z);
		while (*str) {
			glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_10, *str);
			str++;
		}
	}
	void visualize() {
		draw_Frame();
		draw_world();
		draw_Cell();
		if (snowy->T) {
			string cur_Scene = "score :" + std::to_string(snowy->TotalScore);
			string cur_Time = "scene : THE END";
			string code = "https://github.com/jeongkimin/myNN";
			drawBitmapText((char*)cur_Time.c_str(), left_most, upper_most + 0.05f, 0.f);
			drawBitmapText((char*)cur_Scene.c_str(), left_most, upper_most + 0.01f, 0.f);
			drawBitmapText((char*)code.c_str(), left_most, under_most - 0.05f, 0.f);
		}
		else {

			string cur_Scene = "score :" + std::to_string(snowy->TotalScore);
			string cur_Time = "scene :" + std::to_string(snowy->Scene);
			string cur_maxAdv = " : adv(s,argmax(adv))";
			string cur_v = " : V(s)";
			string code = "https://github.com/jeongkimin/myNN";
			drawBitmapText((char*)cur_Time.c_str(), left_most, upper_most + 0.05f, 0.f);
			drawBitmapText((char*)cur_Scene.c_str(), left_most, upper_most + 0.01f, 0.f);
			drawBitmapText((char*)code.c_str(), left_most, under_most - 0.05f, 0.f);
		}
	}
};