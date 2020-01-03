#include "MixtureDensityNetwork.h"
#include "Optimizer.h"
#include "Viewer.h"
int main() {
	DiagCovMDN mdn; {
		mdn.push(new Dense(1, 8));
		mdn.push(new ReLU(8));
		mdn.push(new Dense(8, 8));
		mdn.push(new ReLU(8));
		mdn.push(new Dense(8, 8));
		mdn.push(new ReLU(8));
		mdn.push(new Dense(8, 8));
		mdn.push(new ReLU(8));
		mdn.push(new Dense(8, 2 + 2 + 2));
	}	mdn.setMDN(2, 1);	mdn.setCache(1);

	Optimizer optim(mdn);
	optim.setLearningRate(0.001);

	int n_data = 60000;
	DarrList x = alloc(n_data, 1);
	DarrList y = alloc(n_data, 1);


	for (int ep = 0; ep < 100; ep++) {
		for (int n = 0; n < n_data; n += 2) {
			x[n][0] = km::U(rEngine) * 4.0 - 2.0;
			y[n][0] = sin(3.0 * x[n][0]) + km::STD(rEngine) * 0.2 * sin(2.0 * x[n + 1][0]);

			x[n + 1][0] = km::U(rEngine) * 4.0 - 2.0;
			y[n + 1][0] = cos(5.0 * x[n + 1][0]) + km::STD(rEngine) * 0.1 * cos(x[n + 1][0]);
		}

		double lh = 0.0;

		for (int i = 0; i < n_data; i++) {
			optim.zero_grad();
			lh += mdn.fit(x[i], y[i]);
			optim.step();
		}
		cout << lh / double(n_data) << endl;
	}

}