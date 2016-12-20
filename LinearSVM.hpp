// Based on original version my Miguel Jara, USACH, 2015

#ifndef LINEARSVM_HPP
#define LINEARSVM_HPP

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <sstream>
#include <cstdio>

#include <fstream>

#include <deque>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <dirent.h>

using namespace cv;
using namespace std;


class LinearSVM : public CvSVM{

	public:
  		vector<float> getSupportVector();
};

#endif
