#pragma once
#include "Core.h"
#include "Jm.h"
#define TERMINAL true
#define NON_TERMINAL false
class Env {
public:
	int n_action = 0;
	int observation_size = 0;
	void setEnvparam(const int& n_action, const int& observation_size) {
		this->n_action = n_action;
		this->observation_size = observation_size;
	}
	int numOfAction() { return n_action; }
	int observationSize() { return observation_size; }
	virtual void getCurrentObservation(darr::v& observation) = 0;
	virtual bool isTerminal_Now() = 0;
	virtual double getCurrentReward() = 0;
	virtual void transitionOccur(const int& cause) = 0;

	virtual void initialize() = 0; /* after this function is called, the Env must be in non_terminal_state*/
};


namespace snowy {

	namespace REWARD {
		const double GOOD = 0.02;
		const double DIE = -0.7;
	}

	namespace SCENE_ID {
		const int MAN_SURVIVED = 0;
		const int SNOW = 1;
		const int VOID_ = 2;
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
		iarr::v snow_x = nullptr;
		iarr::v snow_y = nullptr;
		int man_x = 0;
		int survival_time = 0;
		const int MAX_SNOW = 12;
		double snow_gen_prob = 0.0;

		int scene_buffer_size = 0;
		iarr::v scene_buffer = nullptr;
		bool am_i_terminal = true;
		double what_is_my_reward = 0.0;

		Snowy() {}

		Snowy(const h_w& hw, const double& humidity = 0.2) {
			create(hw, humidity);
		}

		void create(const h_w& hw, const double& humidity) {
			height = hw.first;
			width = hw.second;
			scene_buffer_size = height * width;
			int observation_size_of_this_env = 3 * height * width;
			setEnvparam(ACTION::_TOTAL_, observation_size_of_this_env);
			snow_x = iarr::alloc(MAX_SNOW);
			snow_y = iarr::alloc(MAX_SNOW);
			scene_buffer = iarr::alloc(observation_size_of_this_env);
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
			if (km::bernoulli_sampling(snow_gen_prob)) {
				for (int i = 0; i < MAX_SNOW; i++) {
					if (snow_y[i] < 0) {
						snow_x[i] = km::randint(width);
						snow_y[i] = height - 1;
						break;
					}
				}
			}
		}

		void embody(darr::v& scene) {
			for (int i = 0; i < scene_buffer_size; i++) {
				if (scene_buffer[i] == SCENE_ID::VOID_) {
					scene[i] = 1.0;
					scene[scene_buffer_size + i] = 0.0;
					scene[scene_buffer_size + scene_buffer_size + i] = 0.0;

				}
				else if (scene_buffer[i] == SCENE_ID::SNOW) {
					scene[i] = 0.0;
					scene[scene_buffer_size + i] = 1.0;
					scene[scene_buffer_size + scene_buffer_size + i] = 0.0;
				}
				else if (scene_buffer[i] == SCENE_ID::MAN_SURVIVED) {
					scene[i] = 0.0;
					scene[scene_buffer_size + i] = 0.0;
					scene[scene_buffer_size + scene_buffer_size + i] = 1.0;
				}
				else {
					cout << "Snowy.embody error" << endl;
					assert(false);
				}
			}
		}

		void update_frame_buffer() {
			for (int i = 0; i < scene_buffer_size; i++) {
				scene_buffer[i] = SCENE_ID::VOID_;
			}
			scene_buffer[man_x] = SCENE_ID::MAN_SURVIVED;
			for (int i = 0; i < MAX_SNOW; i++) {
				if (0 <= snow_y[i] && snow_y[i] < height) {
					scene_buffer[snow_y[i] * width + snow_x[i]] = SCENE_ID::SNOW; /* SNOW */
				}
			}
		}

		virtual void getCurrentObservation(darr::v& observation) {
			update_frame_buffer();
			embody(observation);
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
				assert(false);
			}
			
			update_reward_and_terminal();
			survival_time++;
		}

		virtual void update_reward_and_terminal() {
			what_is_my_reward = REWARD::GOOD;
			am_i_terminal = NON_TERMINAL;
			for (int i = 0; i < MAX_SNOW; i++) {
				if (snow_y[i] == 0) {
					if (snow_x[i] == man_x) {
						am_i_terminal = TERMINAL;
						what_is_my_reward = REWARD::DIE;
						break;
					}
				}
			}
		}

		virtual bool isTerminal_Now() {
			return am_i_terminal;
		}

		virtual double getCurrentReward() {
			return what_is_my_reward;
		}

		virtual void initialize() {/* after this function is called, the Env must be in non_terminal_state*/
			cout << "surv_time :" << survival_time << endl;
			survival_time = 0;
			man_x = width / 2;
			for (int i = 0; i < MAX_SNOW; i++) {
				snow_y[i] = -1;
			}
			update_reward_and_terminal();

			if (isTerminal_Now()) {
				cout << "Snowy.initialize error" << endl;
				assert(false);
			}

		}
	};
}