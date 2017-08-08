// Derived from an original by Miguel Angel Jara, Universidad de Santiago de Chile, 2015

#include "LinearSVM.hpp"

#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

#if CV_MAJOR_VERSION >= 3
// From https://github.com/Itseez/opencv/blob/master/samples/cpp/train_HOG.cpp

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector )
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);

    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );
    hog_detector.clear();

    hog_detector.resize(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
    hog_detector[sv.cols] = (float)-rho;
}

#else
// SAV: I wonder why we need to do this?
// This returns a set of support vectors that are apparently used by the OpenCV's hog detector
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
#endif

