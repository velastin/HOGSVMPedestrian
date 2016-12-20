#ifndef CREATE_DESCRIPTORS_HPP
#define CREATE_DESCRIPTORS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <deque>
#include <vector>

using namespace cv;
using namespace std;



int createDescriptors(string positiveFolderPath,
						string negativeFolderPath,
						string descriptorFolderPath,
						string descriptorsFileName,
						Size resizeSize);


#endif
