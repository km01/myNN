#pragma once
#include "core.h"

namespace Snowy {
	namespace ID {
		const double  _man_ = 0.9;
		const double _snow_ = -0.9;
		const double _void_ = 0.0;
	};

	namespace REWARD {
		const double _good_ = 0.1;
		const double _die_ = -0.5;
		const double _bad_ = -0.2;
		const double _small_penalty_ = -0.1;
	};
	enum ACTION { LEFT, RIGHT, HOLD };
	const int n_action = 3;
	const int n_snow = 15;
	class envirnoment {
	public:
		int grid_height, grid_width;
		int* Snow_x, * Snow_y;
		int* future_Snow_x, * future_Snow_y;
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
		int* around;
		envirnoment() { allocated = false; }
		envirnoment(int _w, int _h) {
			grid_width = _w;
			grid_height = _h;
			state_len = grid_width * grid_height;

			allocate();
			srand((unsigned int)time(NULL));
		}
		~envirnoment() {
			if (allocated) {
				delete[] Snow_x;
				delete[] Snow_y;
				delete[] future_Snow_x;
				delete[] future_Snow_y;
				delete[] current_S;
				delete[] semi_future;
				delete[] around;
			}
		}
		void setHumdity(const double humidity) {
			this->Humidity = humidity;
		}
		void allocate() {
			Snow_x = new int[n_snow]; Snow_y = new int[n_snow];
			future_Snow_x = new int[n_snow]; future_Snow_y = new int[n_snow];
			current_S = new double[state_len]; semi_future = new double[state_len];
			around = new int[5];
			around[0] = -1; //left
			around[1] = 1; //right
			around[2] = grid_width; // up
			around[3] = grid_width - 1; //up and left
			around[4] = grid_width + 1; //up and right
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
			current_T = NON_TERMINAL;
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
				if (0 <= Snow_y[i] && Snow_y[i] < grid_height && 0 <= Snow_x[i] && Snow_x[i] < grid_width) {
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
			PutAgentAndScoring(man_x, current_S, current_R, current_T);
			Scene += 1; TotalScore += current_R;
		}

		/*  s c o r i n g   */
		void PutAgentAndScoring(const int& man_pos, double*& background, double& r, bool& t) {

			r = 0.0;
			if (background[man_pos + grid_width] == Snowy::ID::_snow_) { //upper
				r += Snowy::REWARD::_bad_;
			}
			if (man_pos != 0) {
				if (background[man_pos - 1] == Snowy::ID::_snow_) { //left
					r += Snowy::REWARD::_bad_;
				}
				if (background[man_pos + grid_width - 1] == Snowy::ID::_snow_) { //left& upper
					r += Snowy::REWARD::_bad_;
				}
			}
			if (man_pos != grid_width - 1) {
				if (background[man_pos + 1] == Snowy::ID::_snow_) { // right
					r += Snowy::REWARD::_bad_;
				}
				if (background[man_pos + grid_width + 1] == Snowy::ID::_snow_) { //right& upper
					r += Snowy::REWARD::_bad_;
				}
			}
			if (man_pos == 0 || man_pos == grid_width - 1) {
				r += Snowy::REWARD::_small_penalty_;
			}
			if (background[man_pos] == Snowy::ID::_snow_) {
				t = TERMINAL;
				r += Snowy::REWARD::_die_;
			}
			else {
				t = NON_TERMINAL;
				r += Snowy::REWARD::_good_;
			}

			background[man_pos] = Snowy::ID::_man_;
		}
	};
};