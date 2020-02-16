#pragma once
#include "Env.h"
#include "AgentModel.h"

class SnowyVisualizer {
public:
	snowy::Snowy* snowy = nullptr;
	float time = 0.0f;

	float frame_x = 0.0;
	float frame_y = 0.0;
	float frame_height_len = 0.0;
	float frame_width_len = 0.0;


	float f0_x = 0.0;
	float f0_y = 0.0;

	float f1_x = 0.0;
	float f1_y = 0.0;

	float f2_x = 0.0;
	float f2_y = 0.0;

	int n_cells = 0;
	float cell_size = 0.0;
	int observation_size = 0;
	int frame_buffer_size = 0;
	double* frame_buffer = nullptr;


	double void_r = 0.0;
	double void_g = 1.0;
	double void_b = 0.0;

	double snow_r = 0.0;
	double snow_g = 0.0;
	double snow_b = 1.0;

	double man_r = 1.0;
	double man_g = 0.0;
	double man_b = 0.0;


	darr::v next_a0 = nullptr;
	darr::v next_a1 = nullptr;
	darr::v next_a2 = nullptr;


	SnowyVisualizer() {

	}

	SnowyVisualizer(snowy::Snowy* _snowy) {
		snowy = _snowy;
		cell_size = 0.03;

		frame_height_len = cell_size * snowy->height;
		frame_width_len = cell_size * snowy->width;

		frame_x = -(frame_width_len * 0.5);
		frame_y = (frame_height_len * 0.5);



		f0_x = -0.8;
		f1_x = -0.6;
		f2_x = -0.4;

		f0_y = frame_y;
		f1_y = frame_y;
		f2_y = frame_y;

		n_cells = snowy->width * snowy->height;
		observation_size = snowy->observationSize();
		frame_buffer = new double[observation_size];
		next_a0 = new double[observation_size];
		next_a1 = new double[observation_size];
		next_a2 = new double[observation_size];

	}

	~SnowyVisualizer() {
		delete[] next_a0;
		delete[] next_a1;
		delete[] next_a2;
		delete[] frame_buffer;
	}

	void drawFrame() {
		glLineWidth(1.f);
		glColor3dv(Colors::silver.data);
		glBegin(GL_POLYGON); {
			glVertex2d(frame_x, frame_y);
			glVertex2d(frame_x + frame_width_len, frame_y);
			glVertex2d(frame_x + frame_width_len, frame_y - frame_height_len);
			glVertex2d(frame_x, frame_y - frame_height_len);
		} glEnd();
	}

	void drawWorld() {

		GLdouble color_r = 0.0;
		GLdouble color_g = 0.0;
		GLdouble color_b = 0.0;

		for (int h = 0; h < snowy->height; h++) {
			for (int w = 0; w < snowy->width; w++) {

				color_r = (frame_buffer[h * snowy->width + w]) * void_r
					+ (frame_buffer[n_cells + h * snowy->width + w]) * snow_r
					+ (frame_buffer[n_cells + n_cells + h * snowy->width + w]) * man_r;
				color_g = (frame_buffer[h * snowy->width + w]) * void_g
					+ (frame_buffer[n_cells + h * snowy->width + w]) * snow_g
					+ (frame_buffer[n_cells + n_cells + h * snowy->width + w]) * man_g;
				color_b = (frame_buffer[h * snowy->width + w]) * void_b
					+ (frame_buffer[n_cells + h * snowy->width + w]) * snow_b
					+ (frame_buffer[n_cells + n_cells + h * snowy->width + w]) * man_b;

				glColor3d(color_r, color_g, color_b);
				glPushMatrix(); glBegin(GL_POLYGON); {
					glVertex2f(frame_x + cell_size * w, frame_y - cell_size * (snowy->height - 1 - h));
					glVertex2f(frame_x + cell_size * (w + 1), frame_y - cell_size * (snowy->height - 1 - h));
					glVertex2f(frame_x + cell_size * (w + 1), frame_y - cell_size * (snowy->height - h));
					glVertex2f(frame_x + cell_size * w, frame_y - cell_size * (snowy->height - h));
				} glEnd(); glPopMatrix();


				color_r = (next_a0[h * snowy->width + w]) * void_r
					+ (next_a0[n_cells + h * snowy->width + w]) * snow_r
					+ (next_a0[n_cells + n_cells + h * snowy->width + w]) * man_r;
				color_g = (next_a0[h * snowy->width + w]) * void_g
					+ (next_a0[n_cells + h * snowy->width + w]) * snow_g
					+ (next_a0[n_cells + n_cells + h * snowy->width + w]) * man_g;
				color_b = (next_a0[h * snowy->width + w]) * void_b
					+ (next_a0[n_cells + h * snowy->width + w]) * snow_b
					+ (next_a0[n_cells + n_cells + h * snowy->width + w]) * man_b;

				glColor3d(color_r, color_g, color_b);
				glPushMatrix(); glBegin(GL_POLYGON); {
					glVertex2f(f0_x + cell_size * w, f0_y - cell_size * (snowy->height - 1 - h));
					glVertex2f(f0_x + cell_size * (w + 1), f0_y - cell_size * (snowy->height - 1 - h));
					glVertex2f(f0_x + cell_size * (w + 1), f0_y - cell_size * (snowy->height - h));
					glVertex2f(f0_x + cell_size * w, f0_y - cell_size * (snowy->height - h));
				} glEnd(); glPopMatrix();

				color_r = (next_a1[h * snowy->width + w]) * void_r
					+ (next_a1[n_cells + h * snowy->width + w]) * snow_r
					+ (next_a1[n_cells + n_cells + h * snowy->width + w]) * man_r;
				color_g = (next_a1[h * snowy->width + w]) * void_g
					+ (next_a1[n_cells + h * snowy->width + w]) * snow_g
					+ (next_a1[n_cells + n_cells + h * snowy->width + w]) * man_g;
				color_b = (next_a1[h * snowy->width + w]) * void_b
					+ (next_a1[n_cells + h * snowy->width + w]) * snow_b
					+ (next_a1[n_cells + n_cells + h * snowy->width + w]) * man_b;

				glColor3d(color_r, color_g, color_b);
				glPushMatrix(); glBegin(GL_POLYGON); {
					glVertex2f(f1_x + cell_size * w, f1_y - cell_size * (snowy->height - 1 - h));
					glVertex2f(f1_x + cell_size * (w + 1), f1_y - cell_size * (snowy->height - 1 - h));
					glVertex2f(f1_x + cell_size * (w + 1), f1_y - cell_size * (snowy->height - h));
					glVertex2f(f1_x + cell_size * w, f1_y - cell_size * (snowy->height - h));
				} glEnd(); glPopMatrix();

				color_r = (next_a2[h * snowy->width + w]) * void_r
					+ (next_a2[n_cells + h * snowy->width + w]) * snow_r
					+ (next_a2[n_cells + n_cells + h * snowy->width + w]) * man_r;
				color_g = (next_a2[h * snowy->width + w]) * void_g
					+ (next_a2[n_cells + h * snowy->width + w]) * snow_g
					+ (next_a2[n_cells + n_cells + h * snowy->width + w]) * man_g;
				color_b = (next_a2[h * snowy->width + w]) * void_b
					+ (next_a2[n_cells + h * snowy->width + w]) * snow_b
					+ (next_a2[n_cells + n_cells + h * snowy->width + w]) * man_b;

				glColor3d(color_r, color_g, color_b);
				glPushMatrix(); glBegin(GL_POLYGON); {
					glVertex2f(f2_x + cell_size * w, f2_y - cell_size * (snowy->height - 1 - h));
					glVertex2f(f2_x + cell_size * (w + 1), f2_y - cell_size * (snowy->height - 1 - h));
					glVertex2f(f2_x + cell_size * (w + 1), f2_y - cell_size * (snowy->height - h));
					glVertex2f(f2_x + cell_size * w, f2_y - cell_size * (snowy->height - h));
				} glEnd(); glPopMatrix();
			}
		}
	}

	void draw_Cell() {
		glLineWidth(0.2f);
		glColor3dv(Colors::white.data);
		for (int i = 0; i < snowy->width; i++) {
			glBegin(GL_LINES); {
				glVertex2f(frame_x + i * cell_size, frame_y - frame_height_len);
				glVertex2f(frame_x + i * cell_size, frame_y);

			} glEnd();
		}
		for (int i = 0; i < snowy->height; i++) {
			glBegin(GL_LINES); {
				glVertex2f(frame_x, frame_y - i * cell_size);
				glVertex2f(frame_x + frame_width_len, frame_y - i * cell_size);
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

		bool is_terminal = snowy->isTerminal_Now();
		double reward = snowy->getCurrentReward();
		if (is_terminal) {
			string cur_time = "time : END";
			string cur_reward = "REWARD : " + std::to_string(reward);
			string cur_terminal = "TERMINAL :TRUE";
			string github = "https://github.com/km01/myNN";
			string ob = "O:";
			string ob2 = "E[p(O|S)]:";
			drawBitmapText((char*)ob2.c_str(), f0_x - 0.17f, f0_y - 0.02f, 0.f);
			drawBitmapText((char*)ob.c_str(), frame_x - 0.05f, frame_y - 0.02f, 0.f);
			drawBitmapText((char*)cur_time.c_str(), frame_x, frame_y + 0.10f, 0.f);
			drawBitmapText((char*)cur_reward.c_str(), frame_x, frame_y + 0.06f, 0.f);
			drawBitmapText((char*)cur_terminal.c_str(), frame_x, frame_y + 0.02f, 0.f);
			drawBitmapText((char*)github.c_str(), frame_x, frame_y - frame_height_len - 0.05f, 0.f);
		}
		else {
			string cur_time = "time : " + std::to_string(snowy->survival_time);
			string cur_reward = "REWARD : " + std::to_string(reward);
			string cur_terminal = "TERMINAL : FALSE";
			string github = "https://github.com/km01/myNN";
			string ob = "O:";
			string ob2 = "E[p(O|S)]:";
			drawBitmapText((char*)ob2.c_str(), f0_x - 0.17f, f0_y - 0.02f, 0.f);
			drawBitmapText((char*)ob.c_str(), frame_x - 0.05f, frame_y - 0.02f, 0.f);
			drawBitmapText((char*)cur_time.c_str(), frame_x, frame_y + 0.10f, 0.f);
			drawBitmapText((char*)cur_reward.c_str(), frame_x, frame_y + 0.06f, 0.f);
			drawBitmapText((char*)cur_terminal.c_str(), frame_x, frame_y + 0.02f, 0.f);
			drawBitmapText((char*)github.c_str(), frame_x, frame_y - frame_height_len - 0.05f, 0.f);
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
	AgentModel* agent = nullptr;
	SnowyVisualizer* vis = nullptr;
	int command = 0;
	bool game_state = false;
	int delaymillisecond = 20;



	SnowyLauncher(snowy::Snowy* snowy, int additional_delay = 0) {
		vis = new SnowyVisualizer(snowy);
		command = snowy::ACTION::HOLD;
		delaymillisecond = additional_delay + 20;
	}

	~SnowyLauncher() {
		delete vis;
	}

	void Run(AgentModel* player = nullptr) {
		agent = player;
		vis->snowy->initialize();
		run();
	}

	int getAgentCommand() {
		vis->snowy->getCurrentObservation(vis->frame_buffer);
		return agent->best_choice(vis->frame_buffer, vis->next_a0, vis->next_a1, vis->next_a2);
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

	bool isEnd() {
		return game_state;
	}

	void update() override {
		if (agent == nullptr) {
			vis->snowy->getCurrentObservation(vis->frame_buffer);
			vis->visualize();
			if (game_state == NON_TERMINAL) {
				vis->snowy->transitionOccur(getCommand());
				game_state = vis->snowy->isTerminal_Now();
				std::this_thread::sleep_for(std::chrono::milliseconds(delaymillisecond));
			}
		}
		else {
			vis->snowy->getCurrentObservation(vis->frame_buffer);
			int a = agent->best_choice(vis->frame_buffer, vis->next_a0, vis->next_a1, vis->next_a2);
			vis->visualize();
			if (game_state == NON_TERMINAL) {
				vis->snowy->transitionOccur(a);
				game_state = vis->snowy->isTerminal_Now();
				std::this_thread::sleep_for(std::chrono::milliseconds(delaymillisecond));
			}
		}
	}
};