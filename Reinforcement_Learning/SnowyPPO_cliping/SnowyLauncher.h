#pragma once
#include "Actor.h"
#include "decoder.h"
#include "SnowyVisualizer.h"

enum MODE { USER, AGENT };
class SnowyLauncher : public Game {
public:
	SnowyVisualizer* vis;
	Snowy::ACTION command;
	MODE mode;
	int delaymillisecond;
	Actor* agent;
	Decoder* decoder;
	SnowyLauncher(Snowy::environment* snowy, int additional_delay = 0) {
		vis = new SnowyVisualizer(snowy);
		command = Snowy::HOLD;
		mode = USER;
		delaymillisecond = additional_delay + 20;
	}
	SnowyLauncher(Snowy::environment* snowy, Decoder* _decoder, Actor* _agent, int additional_delay = 0) {
		vis = new SnowyVisualizer(snowy);
		command = Snowy::HOLD;
		mode = AGENT;
		agent = _agent;
		decoder = _decoder;
		delaymillisecond = additional_delay + 20;
	}
	~SnowyLauncher() {
		delete vis;
	}
	void Run() {

		THE_END = false;
		vis->snowy->InitSession();
		vis->snowy->interact(Snowy::HOLD);
		run();
	}
	Snowy::ACTION getAgentCommand() {
		decoder->Decode(vis->snowy->S, agent->_input_);
		int command = agent->DeterministicPolicy(agent->_input_);
		return (Snowy::ACTION)command;
	}
	Snowy::ACTION getUserCommand() {
		if (isKeyPressed(GLFW_KEY_LEFT)) {
			return Snowy::LEFT;
		}
		else if (isKeyPressed(GLFW_KEY_RIGHT)) {
			return Snowy::RIGHT;
		}
		else {
			return Snowy::HOLD;
		}
	}
	Snowy::ACTION getCommand() {
		if (mode == AGENT) {
			return getAgentCommand();
		}
		else {
			return getUserCommand();
		}
	}
	void update() override {
		vis->snowy->interact(getCommand());
		vis->visualize();
		std::this_thread::sleep_for(std::chrono::milliseconds(delaymillisecond));
		if (vis->snowy->T == TERMINAL) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			THE_END = true;
		}
	}
};