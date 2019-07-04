#include "Snowy.h"
#include "nn.h"
#include "optimizer.h"
class ReplayMemory {
public:
	int size;
	int head;
	int tail;
	int* OrderList;
	bool allocated;
	double** state;
	double** Qvalue;
	int state_len;
	int qeval_len;
	ReplayMemory() { allocated = false; }
	ReplayMemory(int _state_len, int _qval_len){
		state_len = _state_len;
		qeval_len = _qval_len;
		allocated = false;
	}
	void allocate(int _size) {
		size = _size;
		state = km_2d::alloc(size, state_len);
		Qvalue = km_2d::alloc(size, qeval_len);
		OrderList = new int[size];
		allocated = true;
		for (int i = 0; i < size; i++) {
			OrderList[i] = i;
		}
		InitSession();
	}
	
	void pop(double*& state_container, double*& target_container) {
		km_1d::copy(state_container, state[OrderList[head]], state_len);
		km_1d::copy(target_container, Qvalue[OrderList[head]], qeval_len);
		head++;
	}
	void push(double* const& _state, double* const& _Qeval) {

		km_1d::copy(state[tail], _state, state_len);
		km_1d::copy(Qvalue[tail], _Qeval, qeval_len);
		tail++;
	}
	bool IsFull() {
		if (tail >= size) {
			return true;
		}
		else {
			return false;
		}
	}
	void fetch(double**& batch_x, double**& batch_y, const int& batch_size) {
		for (int i = 0; i < batch_size; i++) {
			pop(batch_x[i], batch_y[i]);
		}
	}
	void InitSession() {
		km::shuffle(OrderList, size);
		head = 0;
		tail = 0;
	}
	~ReplayMemory() {
		if (allocated) {
			km_2d::free(state, size);
			km_2d::free(Qvalue, size);
			delete[] OrderList;
		}
	}
	
};

class DQnet {
public:
	nn* net;
	double e, gamma;
	int input_size;
	int output_size;
	DQnet(){}
	void publish() {
		net->publish();
		input_size = net->input_size;
		output_size = net->output_size;
	}
	void setParams(const double& _e, const double& _gamma) {
		e = _e;
		gamma = _gamma;
	}
	int stochastic_select(double* const&state) {
		if (e < km::rand0to1(km::rEngine)) {
			return km::randint(output_size);
		}
		else {
			return select(state);
		}
	}
	void predict(double** const& batch_x) {
		net->set_input(batch_x);
		net->forward(net->output_port);
		
	}
	void backward(double** const& batch_dLoss) {
		net->backward(batch_dLoss);
	}

	int select(double* const& input) {
		km_1d::copy(net->input_port[0], input, input_size);
		net->forward(net->output_port);

		km_2d::argmax(net->argmax, net->output_port, 1, output_size);
		return net->argmax[0];
	}

	void evaluate(double*& virtual_Q, double** const& virtual_S, double* const& virtual_R, bool* const& virtual_T) {
		if (virtual_T[Snowy::LEFT] == TERMINAL) {
			virtual_Q[Snowy::LEFT] = virtual_R[Snowy::LEFT];
		}
		else {
			km_1d::copy(net->input_port[0], virtual_S[Snowy::LEFT], input_size);
			net->forward(net->output_port);
			km_2d::argmax(net->argmax, net->output_port, 1, output_size);
			virtual_Q[Snowy::LEFT] = virtual_R[Snowy::LEFT] + gamma * net->output_port[0][net->argmax[0]];
		}
		if (virtual_T[Snowy::RIGHT] == TERMINAL) {
			virtual_Q[Snowy::RIGHT] = virtual_R[Snowy::RIGHT];
		}
		else {
			km_1d::copy(net->input_port[0], virtual_S[Snowy::RIGHT], input_size);
			net->forward(net->output_port);
			km_2d::argmax(net->argmax, net->output_port, 1, output_size);
			virtual_Q[Snowy::RIGHT] = virtual_R[Snowy::RIGHT] + gamma * net->output_port[0][net->argmax[0]];
		}
		if (virtual_T[Snowy::HOLD] == TERMINAL) {
			virtual_Q[Snowy::HOLD] = virtual_R[Snowy::HOLD];
		}
		else {
			km_1d::copy(net->input_port[0], virtual_S[Snowy::HOLD], input_size);
			net->forward(net->output_port);
			km_2d::argmax(net->argmax, net->output_port, 1, output_size);
			virtual_Q[Snowy::HOLD] = virtual_R[Snowy::HOLD] + gamma * net->output_port[0][net->argmax[0]];
		}
	
	}
};

enum COMMANDER {USER, AGENT};
class Shell : public Game {
public:
	DQnet* agent;
	optimizer* net_optim;
	ReplayMemory* memory;
	Snowy::envirnoment* world;
	Snowy::Snowy_Vis* visualizer;
	COMMANDER commander;
	Snowy::ACTION command;
	int batch_size;
	int memory_size;
	int n_batch;
	double** batch_x;
	double** batch_y;
	double** batch_dLoss;
	double** virtual_S;
	double* virtual_R;
	bool* virtual_T;
	double* virtual_Q;
	Shell() { commander = USER; }
	void setEnv(Snowy::envirnoment* snowy) {
		world = snowy;
		visualizer = new Snowy::Snowy_Vis(world);
	}
	void setDQnet(DQnet* dq_net) {
		agent = dq_net;
		memory = new ReplayMemory(agent->input_size, agent->output_size);
	}
	void setOptim(optimizer* optim) {
		net_optim = optim;
	}
	void who_are_you(COMMANDER who) {
		commander = who;
	}
	void get_command() {
		if (commander == AGENT) {
			command = (Snowy::ACTION)agent->select(world->current_S);
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
		agent->net->alloc(batch_size);
		memory->allocate(memory_size);
	}
	double fit() {
		double loss = 0.0;
		agent->net->UseMemory(batch_size);
		for (int i = 0; i < n_batch; i++) {
			net_optim->zero_grad();
			memory->fetch(batch_x, batch_y,batch_size);
			agent->predict(batch_x);
			loss += km_2d::MSEloss(batch_dLoss, agent->net->output_port, batch_y, batch_size, agent->output_size);
			agent->backward(batch_dLoss);
			net_optim->step();
		}
		cout << "loss : " << loss << endl;
		agent->net->UseMemory(1);
		return loss;
	}
	
	void InitSession() {
		THE_END = false;
		world->InitSession();
	}
	int AgentTraining() {
		int Count = 0;
		InitSession();
		world->interact(Snowy::HOLD);
		agent->net->UseMemory(1);
		Snowy::ACTION a;
		world->current_T = NON_TERMINAL;
		while (true) {
			if (world->current_T == TERMINAL) {
				THE_END = true;
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

			//cout << evaluated_Q[0] << " "<<evaluated_Q[1] << " "<<evaluated_Q[2] << endl;
			world->interact(a);
			Count++;
		}
		return Count;
	}
	void update() override {
		get_command();
		world->interact(command);
		visualizer->visualize();
		if (world->current_T == TERMINAL) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			THE_END = true;
		}
	}
};
