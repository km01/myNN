#pragma once
#include "Core.h"

#define TERMINAL true
#define NON_TERMINAL false

namespace PLAYER {
	const int HUMAN = 0;
	const int NEURAL_NETWORK = 1;
};

using namespace km;

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

	virtual void getCurrentObservation(Darr& observation) = 0;
	virtual void getCurrentScene(Darr& observation, double& reward, bool& is_terminal) = 0;
	virtual void getCurrentRT(double& reward, bool& is_terminal) = 0;
	virtual void transitionOccur(const int& cause) = 0;
	virtual bool isTerminal() = 0;
	virtual void initialize() = 0; /* after this function is called, the Env must be in non_terminal_state*/
};