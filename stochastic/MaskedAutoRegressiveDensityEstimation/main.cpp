#include "Flow.h"
#include "Optimizer.h"
#include "Viewer.h"
int main() {
	MADE af;
	GaussianAutoRegressiveBlock* block;
	for (int i = 0; i < 16; i++) {
		block = new GaussianAutoRegressiveBlock();
		block->addLayer(2, 4, true, false);
		block->addLayer(4, 4, true, false);
		block->addLayer(4, 4, true, false);
		block->addLayer(4, 2, false, true);
		af.addBlock(block);
	}
	af.setMask();
	af.setCache(1);
	Optimizer optim(af);
	optim.setLearningRate(0.001);
	int n_data = 1000;
	Darr data = alloc(2);
	for (int ep = 0; ep < 1000; ep++) {
		double loss = 0.0;
		for (int i = 0; i < n_data; i++) {
			data[0] = km::STD(rEngine);
			data[1] = cos(data[0] * 3.0) + km::STD(rEngine) * 0.01;
			optim.zero_grad();
			loss += af.fit(data, 0);
			optim.step();
		}
		cout << loss / n_data << endl;
	}
	DarrList img = alloc(1, 300 * 300);
	for (int y = 0; y < 300; y++) {
		for (int x = 0; x < 300; x++) {
			data[0] = -2.0 + x / 75.0;
			data[1] = -2.0 + y / 75.0;
			img[0][y*300 + x] = af.fit(data, 0);
		}
	}
	Viewer plt(1, 1, 300, 300, img);
	plt.show();
}