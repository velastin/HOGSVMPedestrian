// Based on original version my Miguel Jara, USACH, 2015

#ifndef LINEARSVM_HPP
#define LINEARSVM_HPP

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>

#include <fstream>

#if CV_MAJOR_VERSION >= 3
#include <opencv2/ml.hpp>
#endif

#include <deque>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <dirent.h>

using namespace cv;
using namespace std;
#if CV_MAJOR_VERSION >= 3
using namespace cv::ml;
#endif

#if CV_MAJOR_VERSION >= 3
void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector );

#else
// This is an inherited class from the standard SVM Class
//TODO: to harmonise OpenCV 3.x and 2.x make this class implements a unified interface to SVM

class LinearSVM : public CvSVM{

	public:
  		vector<float> getSupportVector();
};
#endif

#endif
