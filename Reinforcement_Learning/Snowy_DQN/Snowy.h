#include "core.h"
#include "Jm.h"


#define TERMINAL true
#define NON_TERMINAL false

namespace Snowy {
	namespace ID {
		const double  _man_ = 0.9;
		const double _snow_ = -0.9;
		const double _void_ = 0.0;
	};

	namespace REWARD {
		const double _hit_ = -1.0;
		const double _good_ = 0.01;
	};
	enum ACTION {LEFT, RIGHT, HOLD};
	const int n_action = 3;
	const int n_snow = 15;
	class envirnoment {
	public:
		int grid_height, grid_width;
		int *Snow_x, *Snow_y;
		int *future_Snow_x, *future_Snow_y;
		int state_len;
		int Scene;

		double* semi_future;
		double Humidity;

		int currentScene;

		double* current_S;
		bool current_T;
		double current_R;

		bool allocated;
		double TotalScore;

		int man_x;
		envirnoment() { allocated = false; }
		envirnoment(int _w, int _h) {
			grid_width = _w;
			grid_height = _h;
			state_len = grid_width * grid_height;
			allocate();
			srand((unsigned int)time(NULL));
		}
		void setHumdity(const double humidity) {
			this->Humidity = humidity;
		}
		void allocate() {
			Snow_x = new int[n_snow]; Snow_y = new int[n_snow];
			future_Snow_x = new int[n_snow]; future_Snow_y = new int[n_snow];
			current_S = new double[state_len]; semi_future = new double[state_len];
			allocated = true;
		}
		void InitSession() {
			currentScene = 0;
			TotalScore = 0.0;
			Scene = 0;
			current_R = 0.0; current_T = false;
			man_x = grid_width / 2;
			for (int i = 0; i < n_snow; i++) {
				future_Snow_y[i] = -1;
			}
		}
		void snow_update() {
			/* snow <- futue snow */
			for (int i = 0; i < n_snow; i++) {
				Snow_x[i] = future_Snow_x[i];
				Snow_y[i] = future_Snow_y[i];
			}
			/* future - snow fallen */
			for (int i = 0; i < n_snow; i++) {
				if (future_Snow_y[i] >= 0) {
					future_Snow_y[i] -= 1;
				}
			}
			/* future - create a snow */
			if (Humidity > km::rand0to1(km::rEngine)) {
				for (int i = 0; i < n_snow; i++) {
					if (future_Snow_y[i] < 0) {

						future_Snow_x[i] = km::randint(grid_width);
						future_Snow_y[i] = grid_height - 1;
						break;
					}
				}
			}
		}

		void GetCurrentBackGround() {
			for (int i = 0; i < state_len; i++) { //engrave empty space;
				current_S[i] = ID::_void_;
			}
			for (int i = 0; i < n_snow; i++) { //engrave snow space;
				if (0<= Snow_y[i] && Snow_y[i] < grid_height && 0 <= Snow_x[i] && Snow_x[i] < grid_width) {
					current_S[Snow_y[i] * grid_width + Snow_x[i]] = ID::_snow_;
				}
			}
		}

		void GetFutureBackGround(double*& container) {
			for (int i = 0; i < state_len; i++) {
				container[i] = ID::_void_;
			}
			for (int i = 0; i < n_snow; i++) {
				if (0 <= future_Snow_y[i] * grid_width + future_Snow_x[i] && future_Snow_y[i] * grid_width + future_Snow_x[i] < state_len) {
					container[future_Snow_y[i] * grid_width + future_Snow_x[i]] = ID::_snow_;
				}
			}
		}

		void interact(ACTION const& current_A) {
			snow_update();
			if (current_A == LEFT && man_x > 0) {
				man_x -= 1;
			}
			else if (current_A == RIGHT && man_x < grid_width - 1) {
				man_x += 1;
			}
			GetCurrentBackGround();

			if (current_S[man_x] == ID::_snow_) {
				current_R = REWARD::_hit_;
				current_T = TERMINAL;
			}
			else {
				current_R = REWARD::_good_;
				current_T = NON_TERMINAL;
			}
			current_S[man_x] = ID::_man_;
			
			Scene += 1; TotalScore += current_R;
		}

		void SupposeVirtualSituation(double**& virtual_S, double*& virtual_R, bool*& virtual_T) {
			int VirtualManX = man_x;
			GetFutureBackGround(virtual_S[LEFT]);
			GetFutureBackGround(virtual_S[RIGHT]);
			GetFutureBackGround(virtual_S[HOLD]);
			/*  L E F T  */
			if (man_x > 0) {
				VirtualManX = man_x - 1;
			}
			else {
				VirtualManX = man_x;
			}
			if (virtual_S[LEFT][VirtualManX] == ID::_snow_) {
				virtual_T[LEFT] = TERMINAL;
				virtual_R[LEFT] = REWARD::_hit_;
				virtual_S[LEFT][VirtualManX] = ID::_man_;
			}
			else {
				virtual_T[LEFT] = NON_TERMINAL;
				virtual_R[LEFT] = REWARD::_good_;
				virtual_S[LEFT][VirtualManX] = ID::_man_;
			}
			/*  R I G H T  */
			if (man_x < grid_width - 1) {
				VirtualManX = man_x + 1;
			}
			else {
				VirtualManX = man_x;
			}
			if (virtual_S[RIGHT][VirtualManX] == ID::_snow_) {
				virtual_T[RIGHT] = TERMINAL;
				virtual_R[RIGHT] = REWARD::_hit_;
				virtual_S[RIGHT][VirtualManX] = ID::_man_;
			}
			else {
				virtual_T[RIGHT] = NON_TERMINAL;
				virtual_R[RIGHT] = REWARD::_good_;
				virtual_S[RIGHT][VirtualManX] = ID::_man_;
			}
			/*  H O L D  */
			VirtualManX = man_x;
			if (virtual_S[HOLD][VirtualManX] == ID::_snow_) {
				virtual_T[HOLD] = TERMINAL;
				virtual_R[HOLD] = REWARD::_hit_;
				virtual_S[HOLD][VirtualManX] = ID::_man_;
			}
			else {
				virtual_T[HOLD] = NON_TERMINAL;
				virtual_R[HOLD] = REWARD::_good_;
				virtual_S[HOLD][VirtualManX] = ID::_man_;
			}
		}
	};
	class Snowy_Vis {
	public:
		envirnoment* snowy;
		double dH, dW;
		double time = 0.0;
		double left_most, right_most, upper_most, under_most;
		double frame_scalar = 0.01;
		Snowy_Vis() {}
		Snowy_Vis(envirnoment* _snowy) {
			snowy = _snowy;
			left_most  = -snowy->grid_width * frame_scalar;
			right_most = snowy->grid_width  * frame_scalar;
			under_most = -snowy->grid_height* frame_scalar;
			upper_most = snowy->grid_height * frame_scalar;
			dH = (upper_most - under_most) / (double)snowy->grid_height;
			dW = (right_most - left_most) / (double)snowy->grid_width;
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
					if (snowy->current_S[h*snowy->grid_width + w] == ID::_void_) {
						glColor3dv(Colors::silver.data);
					}
					else if (snowy->current_S[h*snowy->grid_width + w] == ID::_snow_) {
						glColor3dv(Colors::white.data);
					}
					else if (snowy->current_S[h*snowy->grid_width + w] == ID::_man_) {
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
				drawBitmapText((char*)cur_Time.c_str(), left_most, upper_most + 0.05f, 0.f);
				drawBitmapText((char*)cur_Scene.c_str(), left_most, upper_most + 0.01f, 0.f);
			}
			else {
				string cur_Scene = "score :" + std::to_string(snowy->TotalScore);
				string cur_Time = "scene :" + std::to_string(snowy->Scene);
				drawBitmapText((char*)cur_Time.c_str(), left_most, upper_most + 0.05f, 0.f);
				drawBitmapText((char*)cur_Scene.c_str(), left_most, upper_most + 0.01f, 0.f);
			}
		}
	};
};