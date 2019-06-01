#pragma once
#include "mynn_core.h"
#include <vector>
#include <utility>

enum ACTION { LEFT, RIGHT, UP, DOWN };
enum UNDER { HOLE, LOAD, GOAL, XXXX };
enum MODE { TEST, TRAIN };
typedef pair<int, int> STATE;

class Transition {
public:
	STATE cur_state;
	ACTION cur_action;
	STATE next_state;
	double cur_reward;
	bool TERMINAL;
	Transition() {}
	void store(const STATE& cur_state, const ACTION& cur_action, const STATE& next_state, const double& cur_reward, const bool& _TERMINAL) {
		this->cur_state = cur_state;
		this->cur_action = cur_action;
		this->next_state = next_state;
		this->cur_reward = cur_reward;
		this->TERMINAL = _TERMINAL;
	}

	void fetch(STATE& cur_state, ACTION& cur_action, STATE& next_state, double& cur_reward, bool& TERMINAL) {

		cur_state = this->cur_state;
		cur_action = this->cur_action;
		next_state = this->next_state;
		cur_reward = this->cur_reward;
		TERMINAL = this->TERMINAL;
	}
};
class ReplaysManager {
public:
	int size;
	Transition* memory;
	int mem_counter;
	int recall_counter;
	int* order_list;
	bool is_allocated;
	ReplaysManager() {}
	ReplaysManager(int _size) {
		create(_size);
	}
	void create(int _size) {
		size = _size;
		memory = new Transition[size];
		order_list = new int[size];
		recall_counter = 0;
		mem_counter = 0;
		for (int i = 0; i < size; i++) {
			order_list[i] = i;
		}
		is_allocated = true;
	}
	void memorize(const STATE& cur_state, const ACTION& cur_action, const STATE& next_state, const double& cur_reward, const bool& _TERMINAL) {
		memory[mem_counter].store(cur_state, cur_action, next_state, cur_reward,_TERMINAL);
		//cout << "mem : " << mem_counter << endl;
		mem_counter++;

	}

	void recall(STATE& cur_state, ACTION& cur_action, STATE& next_state, double& cur_reward, bool& TERMINAL) {
		//cout << "rec : " << recall_counter << "order : "<< order_list[recall_counter]<<endl;
		memory[order_list[recall_counter]].fetch(cur_state, cur_action, next_state, cur_reward, TERMINAL);
		recall_counter++;
	}

	bool isfull() {
		if (mem_counter>= size) {
			return true;
		}
		else {
			return false;
		}
	}

	void shuffle() {
		int rand_idx;
		int temp;
		for (int i = 0; i < size; i++) {
			order_list[i] = i;
		}
		for (int iter = 0; iter < 10; iter++) {
			for (int i = 0; i < size; i++) {
				rand_idx = std::rand() % size;
				temp = order_list[rand_idx];
				order_list[rand_idx] = order_list[i];
				order_list[i] = temp;
			}
		}
	}

	void init_session() {
		srand((unsigned int)time(NULL));
		shuffle();
		recall_counter = 0;
		mem_counter = 0;
	}

	~ReplaysManager() {
		if (is_allocated) {
			delete[] order_list;
			delete[] memory;
		}
	}

};

class enviroment {
public:
	int n_state;
	int n_action;

	int width;
	int height;

	UNDER** map = nullptr;
	double penalty = -1.0;
	double goal_score = 1.0;

	enviroment() {}
	enviroment(UNDER* _map, int H, int W,double _reward, double _penalty) {
		penalty = _penalty;
		goal_score = _reward;
		srand((unsigned int)time(NULL));
		n_action = 4;
		n_state = H * W;
		height = H;
		width = W;
		map = new UNDER*[height];
		for (int i = 0; i < height; i++) {
			map[i] = new UNDER[width];
			for (int j = 0; j < width; j++) {
				map[i][j] = _map[i*width + j];
			}
		}
	}
	~enviroment() {
		
		for (int i = 0; i < height; i++) {
			delete[] map[i];
		}
		delete[] map;
	}

	double reward(const STATE& s) {

		if (map[s.first][s.second] == XXXX) {
			return penalty;
		}
		else if (map[s.first][s.second] == LOAD) {
			return 0.0;
		}
		else if (map[s.first][s.second] == HOLE) {
			return penalty;
		}
		else {
			return goal_score;
		}
	}

	bool isTerminal(const STATE& s) {
		if (map[s.first][s.second] == LOAD) {
			return false;
		}
		else {
			return true;
		}
	}

	STATE next(const STATE& s, const ACTION& a) {
		STATE next;
		if (a == LEFT) {
			next.first = s.first;
			next.second = s.second - 1;
		}
		else if (a == RIGHT) {
			next.first = s.first;
			next.second = s.second + 1;
		}
		else if (a == UP) {
			next.first = s.first - 1;
			next.second = s.second;
		}
		else if (a == DOWN) {
			next.first = s.first + 1;
			next.second = s.second;
		}
		return next;
	}

	void showmap() {
		for (int h = 0; h < height; h++) {
			for (int w = 0; w < width; w++) {
				if (map[h][w] == HOLE) {
					cout << "H" << " ";
				}
				else if (map[h][w] == LOAD) {
					cout << "L" << " ";
				}
				else if (map[h][w] == GOAL) {
					cout << "G" << " ";
				}
				else if (map[h][w] == XXXX) {
					cout << "X" << " ";
				}
			}
			cout << endl;
		}
		cout << endl;
	}
};