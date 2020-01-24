#pragma once
#include "Snowy.h"
#include "Jm.h"
#include "McSamplingDoubleDqn.h"

class SnowyVisualizer {
public:
	snowy::Snowy* snowy = nullptr;
	float dH = 0.0;
	float dW = 0.0;
	float time = 0.0f;
	float left_most = 0.0, right_most = 0.0, upper_most = 0.0, under_most = 0.0;
	float frame_scalar = 0.02f;
	int observation_size = 0;
	Darr frame_buffer = nullptr;
	SnowyVisualizer() {

	}

	SnowyVisualizer(snowy::Snowy* _snowy) {
		snowy = _snowy;
		left_most = -snowy->width * frame_scalar;
		right_most = snowy->width * frame_scalar;
		under_most = -snowy->height * frame_scalar;
		upper_most = snowy->height * frame_scalar;
		dH = (upper_most - under_most) / (float)snowy->height;
		dW = (right_most - left_most) / (float)snowy->width;
		observation_size = _snowy->observationSize();
		frame_buffer = alloc(observation_size);
	}

	~SnowyVisualizer() {
		free(frame_buffer);
	}

	void drawFrame() {
		glLineWidth(1.f);
		glColor3dv(Colors::silver.data);
		glBegin(GL_POLYGON); {
			glVertex2d(left_most, upper_most);
			glVertex2d(right_most, upper_most);
			glVertex2d(right_most, under_most);
			glVertex2d(left_most, under_most);
		} glEnd();
	}

	void drawWorld() {
		snowy->getCurrentObservation(frame_buffer);
		for (int h = 0; h < snowy->height; h++) {
			for (int w = 0; w < snowy->width; w++) {
				if (frame_buffer[h * snowy->width + w] == snowy::SCENE_ID::VOID_) {
					glColor3dv(Colors::silver.data);
				}
				else if (frame_buffer[h * snowy->width + w] == snowy::SCENE_ID::SNOW) {
					glColor3dv(Colors::white.data);
				}
				else if (frame_buffer[h * snowy->width + w] == snowy::SCENE_ID::MAN_DIED) {
					glColor3dv(Colors::red.data);
				}
				else if (frame_buffer[h * snowy->width + w] == snowy::SCENE_ID::MAN_SURVIVED) {
					glColor3dv(Colors::yellow.data);
				}
				else {
					glColor3dv(Colors::black.data);
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
		for (int i = 0; i < snowy->width; i++) {
			glBegin(GL_LINES); {
				glVertex2f(left_most + i * (dW), upper_most);
				glVertex2f(left_most + i * (dW), under_most);

			} glEnd();
		}
		for (int i = 0; i < snowy->height; i++) {
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

	void drawTextInfo() {
		bool is_terminal = 0;
		double reward = 0;
		snowy->getCurrentRT(reward, is_terminal);
		if (is_terminal) {
			string cur_time = "time : END";
			string cur_reward = "REWARD : " + std::to_string(reward);
			string cur_terminal = "TERMINAL :TRUE";
			string github = "https://github.com/km01/myNN";
			drawBitmapText((char*)cur_time.c_str(), left_most, upper_most + 0.10f, 0.f);
			drawBitmapText((char*)cur_reward.c_str(), left_most, upper_most + 0.06f, 0.f);
			drawBitmapText((char*)cur_terminal.c_str(), left_most, upper_most + 0.02f, 0.f);
			drawBitmapText((char*)github.c_str(), left_most, under_most - 0.05f, 0.f);
		}
		else {
			string cur_time = "time : " + std::to_string(snowy->survival_time);
			string cur_reward = "REWARD : " + std::to_string(reward);
			string cur_terminal = "TERMINAL : FALSE";
			string github = "https://github.com/km01/myNN";
			drawBitmapText((char*)cur_time.c_str(), left_most, upper_most + 0.10f, 0.f);
			drawBitmapText((char*)cur_reward.c_str(), left_most, upper_most + 0.06f, 0.f);
			drawBitmapText((char*)cur_terminal.c_str(), left_most, upper_most + 0.02f, 0.f);
			drawBitmapText((char*)github.c_str(), left_most, under_most - 0.05f, 0.f);
		}
	}

	void visualize() {
		drawFrame();
		drawWorld();
		draw_Cell();
		drawTextInfo();
	}
};

class SnowyLauncher : public Game {
public:
	SnowyVisualizer* vis = nullptr;
	DuelingNetwork_Wrapper* agent = nullptr;
	int command = 0;
	int delaymillisecond = 20;
	SnowyLauncher(snowy::Snowy* snowy, int additional_delay = 0) {
		vis = new SnowyVisualizer(snowy);
		command = snowy::ACTION::HOLD;
		delaymillisecond = additional_delay + 20;
	}


	~SnowyLauncher() {
		delete vis;
	}

	void Run(DuelingNetwork_Wrapper* agent = nullptr) {
		this->agent = agent;
		THE_END = false;
		vis->snowy->initialize();
		run();
	}

	int getAgentCommand() {
		return agent->actor_argmax(vis->frame_buffer);
	}

	int getUserCommand() {
		if (isKeyPressed(GLFW_KEY_LEFT)) {
			return snowy::ACTION::LEFT;
		}
		else if (isKeyPressed(GLFW_KEY_RIGHT)) {
			return snowy::ACTION::RIGHT;
		}
		else {
			return snowy::ACTION::HOLD;
		}
	}

	int getCommand() {
		if (agent == nullptr) {
			return getUserCommand();
		}
		else {
			return getAgentCommand();
		}
	}

	void update() override {
		vis->visualize();
		vis->snowy->transitionOccur(getCommand());
		std::this_thread::sleep_for(std::chrono::milliseconds(delaymillisecond));
		if (vis->snowy->isTerminal()) {
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
			THE_END = true;
		}
	}
};