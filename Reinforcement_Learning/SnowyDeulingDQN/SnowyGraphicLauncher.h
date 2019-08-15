#pragma once
#include "Snowy.h"
#include "DeulNet.h"
#include "optimizer.h"
#include "Jm.h"


class SnowyVisualizer {
public:
	Snowy::envirnoment* snowy;
	float dH, dW;
	float time = 0.0f;
	float left_most, right_most, upper_most, under_most;
	float frame_scalar = 0.02f;
	vector<double> value;
	vector<double> max_adv;
	SnowyVisualizer() {
		snowy = nullptr; dH = 0.0f; dW = 0.0f; time = 0.0f; frame_scalar = 0.0f;
		left_most = 0.0f; right_most = 0.0f; upper_most = 0.0f; under_most = 0.0f;
		value.clear();
		max_adv.clear();
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
				if (snowy->current_S[h * snowy->grid_width + w] == Snowy::ID::_void_) {
					glColor3dv(Colors::silver.data);
				}
				else if (snowy->current_S[h * snowy->grid_width + w] == Snowy::ID::_snow_) {
					glColor3dv(Colors::white.data);
				}
				else if (snowy->current_S[h * snowy->grid_width + w] == Snowy::ID::_man_) {
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
	void draw_value_graph() {
		glLineWidth(0.2f);
		glColor3dv(Colors::black.data);
		glBegin(GL_LINES); {
			glVertex2d(-0.1, under_most - 0.2);
			glVertex2d(0.4, under_most - 0.2);
		} glEnd();
		int head;
		int tail = value.size();
		if (tail < 100) {
			head = 0;
		}
		else {
			head = tail - 100;
		}
		double dx = (0.005);
		for (int v = head; v < tail-1; v++) {
			glBegin(GL_LINES); {
				glVertex2d(-0.1 + (dx*((double)v- (double)head)), under_most - 0.2 + value[v]*0.1);
				glVertex2d(-0.1+ (dx * ((double)v - (double)head + 1.0)), under_most - 0.2 + value[v+1]*0.1);
			} glEnd();
		}

		glBegin(GL_LINES); {
			glVertex2d(-0.1, under_most - 0.5);
			glVertex2d(0.4, under_most - 0.5);
		} glEnd();
		glColor3dv(Colors::red.data);
		for (int v = head; v < tail - 1; v++) {
			glBegin(GL_LINES); {
				glVertex2d(-0.1 + (dx * ((double)v - (double)head)), under_most - 0.5 + max_adv[v] * 0.1);
				glVertex2d(-0.1 + (dx * ((double)v - (double)head + 1.0)), under_most - 0.5 + max_adv[v + 1] * 0.1);
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
		draw_value_graph();
		if (snowy->current_T) {
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
			drawBitmapText((char*)cur_maxAdv.c_str(), 0.4, under_most - 0.5, 0.f);
			drawBitmapText((char*)cur_v.c_str(), 0.4, under_most - 0.2, 0.f);
		}
	}
};

enum COMMANDER { USER, AGENT };
class SnowyGraphicLauncher : public Game {
public:
	SnowyVisualizer* vis;
	Snowy::ACTION command;
	COMMANDER commander;
	int delaymillisecond;
	DeulNet* agent;
	SnowyGraphicLauncher(Snowy::envirnoment* snowy, int additional_delay = 0) {
		vis = new SnowyVisualizer(snowy);
		command = Snowy::HOLD;
		delaymillisecond = additional_delay + 20;
	}
	void Run(DeulNet* _agent) {
		agent = _agent;
		commander = AGENT;
		THE_END = false;
		vis->snowy->InitSession();
		run();
	}
	void Run() {
		commander = USER;
		THE_END = false;
		vis->snowy->InitSession();
		run();
	}
	Snowy::ACTION getAgentCommand() {
		int command = agent->select(vis->snowy->current_S);
		vis->value.push_back(agent->val_out[0]);
		vis->max_adv.push_back(km::max(agent->adv_out, agent->output_size));
		return (Snowy::ACTION)command;
	}
	Snowy::ACTION getUserCommand() {
		if (isKeyPressed(GLFW_KEY_LEFT)) {
			return Snowy::LEFT;
		}
		else if (isKeyPressed(GLFW_KEY_RIGHT)) {
			return Snowy::RIGHT;
		}
		else {
			return Snowy::HOLD;
		}
	}
	Snowy::ACTION getCommand() {
		if (commander == AGENT) {
			return getAgentCommand();
		}
		else {
			return getUserCommand();
		}
	}
	void update() override {
		vis->snowy->interact(getCommand());
		vis->visualize();
		std::this_thread::sleep_for(std::chrono::milliseconds(delaymillisecond));
		if (vis->snowy->current_T == TERMINAL) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			THE_END = true;
		}
	}
};