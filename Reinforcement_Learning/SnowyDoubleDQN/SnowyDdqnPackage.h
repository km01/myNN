#pragma once
#include "Snowy.h"
#include "DoubleDQN.h"
#include "optimizer.h"
#include "Jm.h"

class Snowy_Vis : public Game {
public:
	Snowy::envirnoment* snowy;
	DDQnet* agent;
	double dH, dW;
	double time = 0.0;
	double left_most, right_most, upper_most, under_most;
	double frame_scalar = 0.015;
	COMMANDER commander;
	Snowy::ACTION command;
	Snowy_Vis() {}
	Snowy_Vis(Snowy::envirnoment* _snowy, DDQnet* _agent) {
		snowy = _snowy;
		left_most = -snowy->grid_width * frame_scalar;
		right_most = snowy->grid_width  * frame_scalar;
		under_most = -snowy->grid_height* frame_scalar;
		upper_most = snowy->grid_height * frame_scalar;
		dH = (upper_most - under_most) / (double)snowy->grid_height;
		dW = (right_most - left_most) / (double)snowy->grid_width;
		agent = _agent;
		THE_END = false;
	}
	void who_are_you(COMMANDER who) {
		commander = who;
	}
	void get_command() {
		if (commander == AGENT) {
			command = (Snowy::ACTION)agent->select(snowy->current_S);
		}
		else {
			if (isKeyPressed(GLFW_KEY_LEFT)) {
				command = Snowy::LEFT;
			}
			else if (isKeyPressed(GLFW_KEY_RIGHT)) {
				command = Snowy::RIGHT;
			}
			else {
				command = Snowy::HOLD;
			}
		}
	}

	void update() override {
		get_command();
		snowy->interact(command);
		visualize();
		if (snowy->current_T == TERMINAL) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			THE_END = true;
		}
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
				if (snowy->current_S[h*snowy->grid_width + w] == Snowy::ID::_void_) {
					glColor3dv(Colors::silver.data);
				}
				else if (snowy->current_S[h*snowy->grid_width + w] == Snowy::ID::_snow_) {
					glColor3dv(Colors::white.data);
				}
				else if (snowy->current_S[h*snowy->grid_width + w] == Snowy::ID::_man_) {
					glColor3dv(Colors::red.data);
				}
				else {
					glColor3dv(Colors::yellow.data);
				}
				glPushMatrix(); glBegin(GL_POLYGON); {
					glVertex2d(left_most + dW * w, under_most + dH * h);
					glVertex2d(left_most + dW * (w + 1), under_most + dH * h);
					glVertex2d(left_most + dW * (w + 1), under_most + dH * (h + 1));
					glVertex2d(left_most + dW * w, under_most + dH * (h + 1));
				} glEnd(); glPopMatrix();
			}
		}
	}
	void draw_Cell() {
		glLineWidth(0.1f);
		glColor3dv(Colors::black.data);
		for (int i = 0; i < snowy->grid_width; i++) {
			glBegin(GL_LINES); {
				glVertex2d(left_most + i * (dW), upper_most);
				glVertex2d(left_most + i * (dW), under_most);

			} glEnd();
		}
		for (int i = 0; i < snowy->grid_height; i++) {
			glBegin(GL_LINES); {
				glVertex2d(left_most, under_most + i * (dH));
				glVertex2d(right_most, under_most + i * (dH));
			} glEnd();
		}
	}
	void drawBitmapText(char *str, float x, float y, float z) {
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
		std::this_thread::sleep_for(std::chrono::milliseconds(30));
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
			string code = "https://github.com/jeongkimin/myNN";
			string esc = "ESC -> next learning";
			drawBitmapText((char*)cur_Time.c_str(), left_most, upper_most + 0.05f, 0.f);
			drawBitmapText((char*)cur_Scene.c_str(), left_most, upper_most + 0.01f, 0.f);
			drawBitmapText((char*)code.c_str(), left_most, under_most - 0.05f, 0.f);
			drawBitmapText((char*)esc.c_str(), left_most, under_most - 0.09f, 0.f);
		}
	}
};


class SnowyPackage {
public:
	DDQnet* agent;
	optimizer* net_optim;
	followeroptimizer* follower_optim;
	ReplayMemory* memory;
	Snowy::envirnoment* world;
	Snowy_Vis* visualizer;
	COMMANDER commander;
	Snowy::ACTION command;
	int batch_size;
	int memory_size;
	int n_batch;
	bool allocated;
	double** batch_x;
	double** batch_y;
	double** batch_dLoss;
	double** virtual_S;
	double* virtual_R;
	bool* virtual_T;
	double* virtual_Q;
	SnowyPackage() { commander = USER; allocated = false; }
	void setEnv(Snowy::envirnoment* snowy) {
		world = snowy;
	}
	void setDQnet(DDQnet* ddq_net) {
		agent = ddq_net;
		memory = new ReplayMemory(agent->input_size, agent->output_size);
	}

	void setOptim(optimizer* _net_optim, followeroptimizer* _follower_optim) {
		net_optim = _net_optim;
		follower_optim = _follower_optim;
	}

	void allocate(int _memory_size, int _batch_size) {
		memory_size = _memory_size;
		batch_size = _batch_size;
		n_batch = memory_size / batch_size;
		batch_x = km_2d::alloc(batch_size, agent->input_size);
		batch_y = km_2d::alloc(batch_size, agent->output_size);
		batch_dLoss = km_2d::alloc(batch_size, agent->output_size);
		virtual_S = km_2d::alloc(agent->output_size, agent->input_size);
		virtual_R = km_1d::alloc(agent->output_size);
		virtual_T = new bool[agent->output_size];
		virtual_Q = km_1d::alloc(agent->output_size);
		agent->DQNNetwork->alloc(batch_size);
		agent->DQNNetwork->UseMemory(1);
		agent->TargetNetwork->alloc(1);
		memory->allocate(memory_size);
		allocated = true;
	}

	double fit() {
		double loss = 0.0;
		agent->StartLearning();
		for (int i = 0; i < n_batch; i++) {
			net_optim->zero_grad();
			memory->fetch(batch_x, batch_y, batch_size);
			agent->predict(batch_x);
			loss += km_2d::MSEloss(batch_dLoss, agent->DQNNetwork->output_port, batch_y, batch_size, agent->output_size);
			agent->backward(batch_dLoss);
			net_optim->step();
		}
		follower_optim->step();
		cout << "loss : " << loss << endl;
		agent->EndLearning();
		return loss;
	}
	void Run(COMMANDER who) {
		world->InitSession();
		
		Snowy_Vis game(world, agent);
		game.who_are_you(who);
		game.run();
	}
	void InitSession() {
		world->InitSession();
	}

	int AgentTraining() {
		int Count = 0;
		InitSession();
		world->interact(Snowy::HOLD);
		Snowy::ACTION a;
		world->current_T = NON_TERMINAL;
		while (true) {
			if (world->current_T == TERMINAL) {
				break;
			}

			a = (Snowy::ACTION)agent->stochastic_select(world->current_S);
			world->SupposeVirtualSituation(virtual_S, virtual_R, virtual_T);

			agent->evaluate(virtual_Q, virtual_S, virtual_R, virtual_T);
			if (memory->IsFull()) {
				fit();
				memory->InitSession();
			}
			memory->push(world->current_S, virtual_Q);
			world->interact(a);
			Count++;
		}
		return Count;
	}
	~SnowyPackage() {
		if (allocated) {
			km_2d::free(batch_x, batch_size);
			km_2d::free(batch_y, batch_size);
			km_2d::free(batch_dLoss, batch_size);
			km_2d::free(virtual_S, agent->output_size);
			km_1d::free(virtual_R);
			delete[] virtual_T;
			km_1d::free(virtual_Q);
		}
	}
};