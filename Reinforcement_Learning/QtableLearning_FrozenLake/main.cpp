#include "mynn_core.h"
#include <vector>
#include <utility>
#include <time.h>
using namespace std;
#define DOUBLE_MIN -100000000.0
enum ACTION { LEFT, RIGHT, UP, DOWN };
enum UNDER { HOLE, LOAD, GOAL, XXXX };
enum MODE { TEST, TRAIN };
typedef pair<int, int> STATE;

#define W 8
#define H 4

int randint(const int& start, const int& end) {
	return rand() % (end - start);
}

class enviroment {
public:
	int n_state = W * H;
	int width = W;
	int height = H;
	UNDER map[H][W] = { {XXXX, XXXX, XXXX, XXXX, XXXX, XXXX, XXXX, XXXX}, //1
						{XXXX, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, XXXX}, //2
						{XXXX, HOLE, LOAD, HOLE, LOAD, LOAD, GOAL, XXXX}, //10
						{XXXX, XXXX, XXXX, XXXX, XXXX, XXXX, XXXX, XXXX} }; //11


	double penalty = -1.0;
	double goal_score = 1.0;
	enviroment() {}
	enviroment(double _reward, double _penalty) {
		penalty = _penalty;
		goal_score = _reward;
		srand((unsigned int)time(NULL));
	}

	double reward(const STATE& s) {

		if (map[s.first][s.second] == XXXX) {
			return penalty * 1.3;
		}
		else if (map[s.first][s.second] == LOAD) {
			return -0.01;
		}
		else if (map[s.first][s.second] == HOLE) {
			return penalty;
		}
		else {
			return goal_score;
		}
	}

	void next(STATE& next, const STATE& s, const ACTION& a) {
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

class FrozenLake_Agent {
public:
	double*** Q_table;
	double e;
	double gamma;
	enviroment frozenLake;
	vector<ACTION> ACTIONlist;
	vector<STATE> STATElist;
	MODE mode = TRAIN;
	int goodidxList[4];
	FrozenLake_Agent(enviroment& env, const double _gamma, const double _e) {
		frozenLake = env;
		Q_table = new double**[frozenLake.height];
		for (int i = 0; i < frozenLake.height; i++) {
			Q_table[i] = km_2d::alloc(frozenLake.width, 4);
			km_2d::fill_zero(Q_table[i], frozenLake.width, 4);
		}
		this->e = _e;
		this->gamma = _gamma;
	}
	~FrozenLake_Agent() {
		for (int i = 0; i < frozenLake.height; i++) {
			km_2d::free(Q_table[i], frozenLake.width);
		}
		delete[] Q_table;
	}

	void showpath() {
		for (int i = 0; i < ACTIONlist.size(); i++) {
			if (ACTIONlist[i] == LEFT) {
				cout << "LEFT" << endl;
			}
			else if (ACTIONlist[i] == RIGHT) {
				cout << "RIGHT" << endl;
			}
			else if (ACTIONlist[i] == UP) {
				cout << "UP" << endl;
			}
			else if (ACTIONlist[i] == DOWN) {
				cout << "DOWN" << endl;
			}
		}
	}

	double Run(STATE s) {

		STATE next_s = s;
		ACTION a = RIGHT;
		ACTION next_a = select(s);
		ACTIONlist.clear();
		STATElist.clear();
		double reward = 0.0;
		double eval = 0.0;
		while (true) {
			s = next_s;
			a = next_a;
			ACTIONlist.push_back(a);
			STATElist.push_back(s);
			frozenLake.next(next_s, s, a);
			eval = frozenLake.reward(next_s);
			reward += eval;
			if (frozenLake.map[next_s.first][next_s.second] != LOAD) {
				Q_table[s.first][s.second][a] = eval;
				break;
			}
			next_a = select(next_s);
			Q_table[s.first][s.second][a] = eval + gamma * Q_table[next_s.first][next_s.second][next_a];
			if (this->mode == TRAIN) {
				if (this->e < RandGen(rEngine)) {
					next_a = a;
				}
			}
		}
		return reward;
	}

	void setMode(MODE _mode) {
		this->mode = _mode;
	}

	ACTION select(const STATE& s) {
		double max = DOUBLE_MIN;
		int count = 0;
		for (int i = 0; i < 4; i++) {
			if (max < Q_table[s.first][s.second][i]) {
				max = Q_table[s.first][s.second][i];
				goodidxList[0] = i;
				count = 1;
			}
			else if (max == Q_table[s.first][s.second][i]) {
				goodidxList[count] = i;
				count += 1;
			}
		}
		return (ACTION)goodidxList[randint(0, count)];
	}
};

int main() {
	enviroment frozenLake(1, -1);
	double gamma = 0.99;
	double e = 0.9;
	FrozenLake_Agent agent(frozenLake, gamma, e);

	STATE start(1, 1);
	double reward = 0.0;
	for (int i = 0; i < 100; i++) {
		reward = agent.Run(start);
		cout << "sess " << i + 1 << ": " << reward << endl;
	}
	agent.setMode(TEST);
	agent.showpath();


	frozenLake.showmap();
}