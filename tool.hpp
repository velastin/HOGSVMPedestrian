// Based on original version my Miguel Jara, USACH, 2015

#ifndef TOOL_HPP
#define TOOL_HPP

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

typedef struct {
	Mat labels;
	Mat descriptors;
} DataDescriptors;


vector <string> arrayFilesName(string folderPath);
void ShowBar(int percentage,int sizeMax);
void pause(string msg);
void hard_pause(string msg);

// Appends data descriptors from a given file. See CreateDescriptors() for details on how file is created
int getDataDescriptors(string descriptorsFile, DataDescriptors &data);


#endif
