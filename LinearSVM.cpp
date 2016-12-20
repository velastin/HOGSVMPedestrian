#include "LinearSVM.hpp"

#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

// SAV: I wonder why we need to do this?
vector<float> LinearSVM::getSupportVector() {

    int sv_count = get_support_vector_count();
    const CvSVMDecisionFunc* df = decision_func;
    const double* alphas = df[0].alpha;
    double rho = df[0].rho;
    int var_count = get_var_count();
    vector <float> support_vector(var_count, 0);
    for (unsigned int r = 0; r < (unsigned)sv_count; r++) {
    	float myalpha = alphas[r];
		const float* v = get_support_vector(r);
		for (int j = 0; j < var_count; j++,v++) {
		support_vector[j] += (-myalpha) * (*v);
		}
    }
    support_vector.push_back(rho);
    return support_vector;
}

