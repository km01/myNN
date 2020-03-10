#include "CustomDataSet.h"
#include "Unit.h"
#include "Optim.h"
#include "LatentSpace.h"

int main() {

	CustomDataSet set;	
	int code_size = 2;
	double learning_rate = 0.000001;
	darr::v data = darr::alloc(2);
	darr::v field_grad = darr::alloc(code_size);
	LatentCodeSpace field(4, code_size);
	int n_batch = 40000;
	field.show_categorical_prior();
	double clustering_loss = 0.0;


	field.centroid_mean[0][0] = 0.5 + STD(rEngine) * 0.01;
	field.centroid_mean[0][1] = 0.5 + STD(rEngine) * 0.01;

	field.centroid_mean[1][0] = -0.5 + STD(rEngine) * 0.01;
	field.centroid_mean[1][1] = 0.5 + STD(rEngine) * 0.01;

	field.centroid_mean[2][0] = 0.5 + STD(rEngine) * 0.01;
	field.centroid_mean[2][1] = -0.5 + STD(rEngine) * 0.01;

	field.centroid_mean[3][0] = -0.5 + STD(rEngine) * 0.01;
	field.centroid_mean[3][1] = -0.5 + STD(rEngine) * 0.01;

	field.pre_categorical_prior[0] = 1.0;
	field.pre_categorical_prior[1] = 1.0;
	field.pre_categorical_prior[2] = 1.0;
	field.pre_categorical_prior[3] = 1.0;


	for (int i = 0; i < 2000; i++) {
		clustering_loss = 0.0;
		field.zero_grad();
		for (int j = 0; j < n_batch; j++) {
			set.getData(data);
			clustering_loss += field.calculate(data);
		}
		field.step(0.000001);
		cout << " loss : "<<clustering_loss / n_batch << endl;
		field.show_categorical_prior();
		cout << endl;
	}
	field.show_categorical_prior();
	darr::free(data);
	darr::free(field_grad);
}