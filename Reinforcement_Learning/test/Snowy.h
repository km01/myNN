#pragma once
#include "Core.h"
#include "Env.h"

namespace snowy {

	namespace REWARD {
		const double GOOD = 0.02;
		const double DIE = -0.5;
		const double BAD = -0.05;
		const double NOT_GOOD = -0.02;
	}
	namespace SCENE_ID {

		const double MAN_DIED = 1.0;
		const double MAN_SURVIVED = 0.5;
		const double SNOW = -0.5;
		const double VOID_ = 0.0;
	}
	namespace ACTION {
		const int _TOTAL_ = 3;
		const int LEFT = 0;
		const int RIGHT = 1;
		const int HOLD = 2;
	}

	class Snowy : public Env {
	public:
		int height = 0;
		int width = 0;
		Iarr snow_x = nullptr;
		Iarr snow_y = nullptr;
	
		int man_x = 0;
		int survival_time = 0;
		double cum_reward = 0.0;
		const int MAX_SNOW = 12;
		double reward = 0.0;

		double snow_gen_prob = 0.0;
		Snowy() {}

		Snowy(const h_w& hw, const double& humidity = 0.2) {
			create(hw, humidity);
		}

		void create(const h_w& hw, const double& humidity) {
			height = hw.first;
			width = hw.second;
			setEnvparam(ACTION::_TOTAL_, height * width);
			snow_x = ialloc(MAX_SNOW);
			snow_y = ialloc(MAX_SNOW);
			snow_gen_prob = humidity;
			initialize();
		}

		void snow_update() {
			/* future - snow fallen */
			for (int i = 0; i < MAX_SNOW; i++) {
				if (snow_y[i] >= 0) {
					snow_y[i] -= 1;
				}
			}

			/* future - generate new snow */
			if (bernoulli_sampling(snow_gen_prob)) {
				for (int i = 0; i < MAX_SNOW; i++) {
					if (snow_y[i] < 0) {
						snow_x[i] = km::randint(width);
						snow_y[i] = height - 1;
						break;
					}
				}
			}
		}

		void embody(Darr& scene) {
			for (int i = 0; i < observation_size; i++) {
				scene[i] = SCENE_ID::VOID_;
			}

			for (int i = 0; i < MAX_SNOW; i++) {
				if (0 <= snow_y[i] && snow_y[i] < height) {
					scene[snow_y[i] * width + snow_x[i]] = SCENE_ID::SNOW;
				}
			}
			if (scene[man_x] == SCENE_ID::SNOW) {
				scene[man_x] = SCENE_ID::MAN_DIED;
			}
			else {
				scene[man_x] = SCENE_ID::MAN_SURVIVED;
			}
		}

		virtual void getCurrentObservation(Darr& observation) {
			embody(observation);
		}

		virtual void getCurrentScene(Darr& observation, double& reward, bool& is_terminal) {
			embody(observation);
			getCurrentRT(reward, is_terminal);
		}

		virtual void getCurrentRT(double& reward, bool& is_terminal) {
			for (int i = 0; i < MAX_SNOW; i++) {
				if (snow_y[i] == 0) {
					if (snow_x[i] == man_x) {
						is_terminal = TERMINAL;
						reward = REWARD::DIE;
						return;
					}
				}
			}
			is_terminal = NON_TERMINAL;
			reward = REWARD::GOOD;
		}

		virtual void transitionOccur(const int& cause) {
			snow_update();
			if (cause == ACTION::LEFT) {
				if (man_x > 0) {
					man_x = man_x - 1;
				}
			}
			else if (cause == ACTION::RIGHT) {
				if (man_x < width - 1) {
					man_x = man_x + 1;
				}
			}

			else if (cause == ACTION::HOLD) {

			}

			else {
				cout << " Snowy.effect error" << endl;
				assert(true);
			}

			survival_time += 1;
		}

		virtual bool isTerminal() {
			static double trash = 0;
			bool is_terminal = false;
			getCurrentRT(trash, is_terminal);
			return is_terminal;
		}

		virtual void initialize() {/* after this function is called, the Env must be in non_terminal_state*/
			cout << "surv_time :" << survival_time << endl;
			survival_time = 0;
			cum_reward = 0.0;
			man_x = width / 2;
			for (int i = 0; i < MAX_SNOW; i++) {
				snow_y[i] = -1;
			}

			snow_update();

			if (isTerminal()) { 
				cout << "Snowy.initialize error" << endl; 
				assert(true);
			}
		}
	};
}