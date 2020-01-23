#include "MixtureDensityNetwork.h"
#include "Optimizer.h"
#include "Viewer.h"
int main() {
	DiagCovMDN2 mdn; {
		mdn.push(new Dense(1, 4));	
		mdn.push(new ReLU(4));
		mdn.push(new Dense(4, 8));
		mdn.push(new ReLU(8));
		mdn.push(new Dense(8, 12));
		mdn.push(new ReLU(12));
		mdn.push(new Dense(12, 16));	
		mdn.push(new ReLU(16));
		mdn.push(new Dense(16, 18));	
		mdn.push(new ReLU(18));
		mdn.push(new Dense(18, 2 + 2 + 2));
	}	mdn.setMDN(2, 1);	mdn.setCache(1);

	Optimizer optim(mdn);
	optim.setLearningRate(0.0005);

	int n_data = 60000;
	DarrList x = alloc(n_data, 1);
	DarrList y = alloc(n_data, 1);

	for (int ep = 0; ep < 300; ep++) {
		for (int n = 0; n < n_data; n += 2) {
			x[n][0] = km::U(rEngine) * 2.0 - 1.0;
			y[n][0] = 0.3 * sin(8.0 * x[n][0]) -0.5 + km::STD(rEngine) * 0.2 * sin(9.0 * x[n][0]);

			x[n + 1][0] = km::U(rEngine) * 2.0 - 1.0;
			y[n + 1][0] = 0.5 * cos(8.0 * x[n + 1][0]) + 0.5 + km::STD(rEngine) * 0.05 * cos(8.0 * x[n+1][0]);
		}
		double lh = 0.0;

		for (int i = 0; i < n_data; i++) {
			optim.zero_grad();
			lh += mdn.fit(x[i], y[i]);
			optim.step();
		}
		cout << lh / double(n_data) << endl;
	}
	for (int n = 0; n < n_data; n++) {
		x[n][0] = km::U(rEngine) * 2.0 - 1.0;
		mdn.sampling(x[n], y[n]);
	}
	Viewer plt(x, y, n_data);
	plt.run();
}