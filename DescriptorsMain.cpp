// (c) S.A. Velastin 2016 (UC3M)
// Derived from original by Miguel Jara, USACH, 2015
// Computes files with HOG descriptors for a set of negative and positive samples

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <deque>
#include <vector>
#include <fstream>

#include <sstream>

#include <algorithm>

#include <cmath>
#include "tool.hpp"
#include "createDescriptors.hpp"

using namespace cv;
using namespace std;

#define SVM_SUFFIX "-SVM.xml"
#define DATA_SUFFIX "-DESC.dat"

// Dimensions of normalised samples (for PAMELA-UANDES data)
#define WIDTH 56
#define HEIGHT 56

// Dimentions of normalised samples (for BOSS, data is already resized to this)
// #define WIDTH 64
// #define HEIGHT 128		// to be compatible with OpenCV Hog detector


void displayUsage(){
	cout << "./Descriptors -n path -p path -d path" << endl;
	cout << "-n path: path (no trailing /) where negative samples are" << endl;  
	cout << "-p path: path (no trailing /) where positive samples are" << endl;
	cout << "-d path: path (no trailing) /) where to store descriptors" << endl;
}

int main( int argc, char** argv ) {
	char opt;
	string neg_path, pos_path, data_path;
	Size data_size = Size(WIDTH,HEIGHT);  // this is the image dimensions to which the samples need resizing


	if(argc < 7){
		cerr << "Missing arguments" << endl;
		displayUsage();
		return -1;
	}

	// Deal with the command line, see http://linux.die.net/man/3/optarg for handling commands
	while((opt = getopt(argc, argv, ":n:p:d:")) != -1){
		switch(opt){
			case 'n':
			neg_path = optarg;
			break;
			case 'p':
			pos_path = optarg;
			break;
			case 'd':
			data_path = optarg;
			break;
			case '?':  // ***SAV not sure why "?" in particular
			cerr << "Invalid option:  '" << char(optopt) << "' doesn't exist." << endl << endl;
			displayUsage();
			exit(1);
			default:
			cerr << "Missing value for argument: '" << char(optopt) << "'" << endl << endl;
			displayUsage();
			exit(1); 
		}
	}

	cout << "Negatives on '" << neg_path << "', Positives on '" << pos_path << "' data on '" << data_path << "'\n";
	if (neg_path == "") { cout << "Hey! negatives path empty\n"; return 2; }
	if (pos_path == "") { cout << "Hey! positives path empty\n"; return 2; }
	if (data_path == "") { cout << "Hey! data path empty\n"; return 2; }

	// Will now try to access the list of video directories
	vector <string> Neg_Directories = arrayFilesName(neg_path);
	vector <string> Pos_Directories = arrayFilesName(pos_path);
	sort(Neg_Directories.begin(), Neg_Directories.end());
	sort(Pos_Directories.begin(), Pos_Directories.end());

	// and display the list of directories
	cout << "Found the following negative directories\n";
	for(int i=0; i < Neg_Directories.size(); i++){
		cout << Neg_Directories [i] << endl;
	}
	cout << "\nFound the following positive directories\n";
	for(int i=0; i < Pos_Directories.size(); i++){
		cout << Pos_Directories [i] << endl;
	}
	if (Neg_Directories != Pos_Directories) {
		cout << "Hey! neg and pos directories are different!\n"; // this restriction could be relaxed
		return 3;
	}
	cout << "\nOk, we have the same directory entries, so we proceed...\n";
	// To save computing descriptors multiple times, we take each of the directories and compute
	// descriptors, saving them on the data_path. They would also be useful for training in different
	// ways (SVM, CNN, ....)
	for (int dir=0; dir < Neg_Directories.size(); dir++) {
		string datafile_name = Neg_Directories[dir] + DATA_SUFFIX;
		createDescriptors (pos_path + "/" + Pos_Directories[dir], 
						   neg_path + "/" + Neg_Directories[dir], data_path, datafile_name, data_size);  
	} // each directory
	
	return 0;
}
