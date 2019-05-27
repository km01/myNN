#include "mynn_core.h"
#include "unit.h"
#include "loss_function.h"
#include "model.h"
#include "optimizer.h"
#include <vector>
#include <ctime>
#include <utility>
using namespace std;
#define DOUBLE_MIN -100000000.0
enum ACTION {LEFT, RIGHT, UP, DOWN};
enum UNDER {HOLE, LOAD, GOAL, XXXX};
enum MODE {TEST, TRAIN};
typedef pair<int, int> STATE;

#define W 8
#define H 4

int randint(const int& start, const int& end) {
	return rand() % (end - start);
}

class enviroment {
public:
	int n_state = W*H;
	int width = W;
	int height = H;
	UNDER map[H][W] = { {XXXX, XXXX, XXXX, XXXX, XXXX, XXXX, XXXX, XXXX}, //1
						{XXXX, LOAD, LOAD, LOAD, LOAD, LOAD, LOAD, XXXX}, //2
						{XXXX, HOLE, LOAD, HOLE, LOAD, LOAD, GOAL, XXXX}, //10
						{XXXX, XXXX, XXXX, XXXX, XXXX, XXXX, XXXX, XXXX}}; //11

	
	double penalty = -10.0;
	double goal_score = 1.0;
	enviroment(){}
	enviroment(double _reward, double _penalty) {
		penalty = _penalty;
		goal_score = _reward;
		srand((unsigned int)time(NULL));
	}

	double reward(const STATE& s) {

		if (map[s.first][s.second] == XXXX) {
			return penalty * 1.3;
		}
		else if(map[s.first][s.second] == LOAD) {
			return 0.0;
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
	double e;
	double gamma;

	nn* net;
	optimizer* optim;
	MSELoss* loss_fn;

	enviroment frozenLake;
	vector<ACTION> ACTIONlist;
	vector<STATE> STATElist;
	MODE mode = TRAIN;

	double* selectVector;
	int goodidxList[4];
	int batch_size;
	int batch_counter;
	double** batch_x;
	double** batch_y;
	double** dLoss;
	FrozenLake_Agent(enviroment& env, const double _gamma, const double _e) {
		frozenLake = env;
		this->e = _e;
		this->gamma = _gamma;
		selectVector = new double[frozenLake.n_state];
	}
	
	void set_batch_size(int _batch_size) {
		batch_size = _batch_size;
		optim->set_learning_rate(0.01 / batch_size);
		net->alloc_batch_memory(batch_size);
		net->alloc_single_memory();
		batch_x = km_2d::alloc(batch_size, frozenLake.n_state);
		batch_y = km_2d::alloc(batch_size, 4);
		dLoss = km_2d::alloc(batch_size, 4);
		km_2d::fill_zero(batch_x, batch_size, frozenLake.n_state);
		km_2d::fill_zero(batch_y, batch_size, 4);
		batch_counter = 0;
	}

	void setNN(nn* _net, optimizer* _optim, MSELoss* _loss_fn) {
		net = _net;
		optim = _optim;
		loss_fn = _loss_fn;
	}

	~FrozenLake_Agent() {
		km_2d::free(dLoss, batch_size);
		km_2d::free(batch_x, batch_size);
		km_2d::free(batch_y, batch_size);
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
				take(s, a, eval);
				break;
			}
			next_a = select(next_s);
			if (this->mode == TRAIN) {
				if (this->e < RandGen(rEngine)) {
					next_a = a;
				}
			}
			else {
				if (ACTIONlist.size() > 1000) {
					cout << "infinite Loop..." << endl;
				}
			}
		}
		return reward;
	}
	void take(STATE& s, ACTION a, double target) {
		if (batch_counter >= batch_size) {
			net->batch_feed_forward(batch_x);
			loss_fn->batch_loss_prime(dLoss, net->batch_output, batch_y, batch_size, 4);
			net->zero_grad();
			net->batch_feed_backward(dLoss);
			optim->step();
			km_2d::fill_zero(batch_x, batch_size, frozenLake.n_state);
			km_2d::fill_zero(batch_y, batch_size, 4);
			batch_counter = 0;
		}

		batch_x[batch_counter][frozenLake.width*s.first + s.second] = 1.0;
		batch_y[batch_counter][(int)a] = target;
		batch_counter++;
	}
	
	void setMode(MODE _mode) {
		this->mode = _mode;
	}

	void set_selectVector(const STATE& s) {
		km_1d::fill_zero(selectVector, frozenLake.n_state);
		selectVector[frozenLake.width*s.first + s.second] = 1.0;
	}

	ACTION select(const STATE& s) {
		set_selectVector(s);
		net->single_feed_forward(selectVector);
		return (ACTION)net->single_argmax;
	}
};
int main() {

	nn model(number_of_units(3));
	model.layer[0] = new perceptrons(H*W, 100, TANH);
	model.layer[1] = new perceptrons(100, 30, TANH);
	model.layer[2] = new perceptrons(30, 4, IDENTITY);
	MSELoss loss_fn;
	optimizer optim(model);
	optim.use_RMSprop(0.999);
	enviroment world(1,-1);
	FrozenLake_Agent walker(world, 0.99, 0.9);
	walker.setNN(&model, &optim, &loss_fn);
	walker.set_batch_size(100);

	STATE start(1, 1);

	double r = 0.0;


	walker.e = 0.7;
	for (int i = 0; i < 3000; i++) {
		r = walker.Run(start);
		cout << r << endl;
	}
	walker.e = 0.8;
	for (int i = 0; i < 3000; i++) {
		r = walker.Run(start);
		cout << r << endl;
	}
	walker.e = 0.9;
	for (int i = 0; i < 3000; i++) {
		r = walker.Run(start);
		cout << r << endl;
	}
	walker.frozenLake.showmap();
	walker.setMode(TEST);
	cout << walker.Run(start) << endl;
	walker.showpath();
}