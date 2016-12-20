// (c) S.A. Velastin 2016 (UC3M)
// Based on original by Miguel Jara, USACH 2015
// Gets descriptors previously sent to files, so as to train SVMS on a leave-one-out basis

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <deque>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

#include "tool.hpp"
//#include "createDescriptors.hpp"

using namespace cv;
using namespace std;

#define SVM_SUFFIX "-SVM.xml"
#define DATA_SUFFIX "-DESC.dat"

// Obtained by grid optimisation with half the full (sit, sitting, stand) dataset
// (for RBF set C parameter to 2.0)
//#define SVM_CPARAMETER 0.03125
//#define SVM_GPARAMETER 0.03125

// Obtained by TrainGrid for 64x128 RBF
#define SVM_CPARAMETER 2.0
#define SVM_GPARAMETER 0.125


// Dimentions of normalised samples (for BOSS, data is already resized to this)
#define WIDTH 64
#define HEIGHT 128


void displayUsage(){
	cout << "./TrainLOO -d path -s path" << endl;
	cout << "-d path: path (no trailing /) where descriptor files are" << endl;  
	cout << "-s path: path (no trailing /) where to store SVM model file" << endl;  
}

//***************************************  SVM train ***************************************

/* Default constuctor
The default constructor initialize the structure with following values:

CvSVMParams::CvSVMParams() :
    svm_type(CvSVM::C_SVC), kernel_type(CvSVM::RBF), degree(0),
    gamma(1), coef0(0), C(1), nu(0), p(0), class_weights(0)
{
    term_crit = cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON );
}
*/


int SVMTrain (DataDescriptors &data,
				string modelFolderPath,
				string modelFileName,
				double CParameter, double GammaParameter) 
{	// If output path does not exist, then create it
    struct stat st = {0};
	if (stat(modelFolderPath.c_str(), &st) == -1) {
	    mkdir(modelFolderPath.c_str(), 0700);
	}

	// SVM parameters
	CvSVMParams params; // default constructor: RBF
    	params.C = CParameter;
	params.gamma = GammaParameter;

#ifndef TRAIN_SVM_RBF
   	params.kernel_type = CvSVM::LINEAR; // remove for RBF
	cout << "Training linear SVM, please wait ..." << endl;
#else
	cout << "Training RBF SVM, please wait ..." << endl;
#endif

	  Size oldlabelsSize = data.labels.size();
		Size olddescriptorsSize = data.descriptors.size();
		cout << "Result data labels " << oldlabelsSize.height << " x " << oldlabelsSize.width << endl;
		cout << "Result Descriptors " << olddescriptorsSize.height << " x " << olddescriptorsSize.width << endl;
/*
	pause("Now to squirt out the data on the screen!");
	for (int row=0; row < oldlabelsSize.height; row++) {
		cout << "Label: " << data.labels.at<float>(row,0) << ": ";
		for (int feature=0; feature < olddescriptorsSize.width; feature++)
			cout << data.descriptors.at<float>(row,feature) << " ";
		cout << endl;
	}
*/

    pause("Tis gonna take a loooong time ..");

    CvSVM SVM;
    SVM.train(data.descriptors, data.labels, Mat(), Mat(), params);

    cout << "Successful training" << endl;

    stringstream modelPath;
    modelPath << modelFolderPath << "/" << modelFileName;

    cout << "Saving SVM in " << modelPath.str() << endl;
    SVM.save(modelPath.str().c_str());
    cout << "COMPLETED SUCCESSFULLY" << endl;

    return 0;

}



//*******************************************************************************************
int main( int argc, char** argv ) {
	char opt;
	string data_path, svm_path;
	Size data_size = Size(WIDTH,HEIGHT);  // this is the image dimensions to which the samples need resizing


	if(argc < 5){
		cerr << "Missing arguments" << endl;
		displayUsage();
		return -1;
	}

	// Deal with the command line, see http://linux.die.net/man/3/optarg for handling commands
	while((opt = getopt(argc, argv, ":d:s:")) != -1){
		switch(opt){
			case 'd':
			data_path = optarg;
			break;
			case 's':
			svm_path = optarg;
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

	cout << "Data on '" << data_path << "'\n";
	cout << "SVM on '" << svm_path << "'\n";
	if (data_path == "") { cout << "Hey! data path empty\n"; return 2; }
	if (svm_path == "") { cout << "Hey! svm path empty\n"; return 3; }

	// Will now try to access the list of video directories
	vector <string> Data_Directories = arrayFilesName(data_path);
	vector <string> Data_Files;

	// and display the list of files while searching for the actual data files
	cout << "Found the following data entries \n";
	for (int i=0; i < Data_Directories.size(); i++){
		int pos = Data_Directories[i].find (DATA_SUFFIX); // NB does not check that it is at the end of name!
		string data_file;
		if (pos!= string::npos) {
			data_file = Data_Directories[i].substr(0, pos); // found the pattern so this will be considered
			Data_Files.push_back(data_file);
		}
		cout << Data_Directories [i] << "(" << pos << ") '" << data_file << "'" << endl;
	} // each directory
	sort (Data_Files.begin(), Data_Files.end()); // should not be necessary but just in case
	cout << "\nThese are the data files to be processed\n";

	for (int i=0; i<Data_Files.size(); i++) {
		cout << "'" << Data_Files[i] << "'\n";
	}
	pause ("Here we go");
	
	// Right. Now we create a bunch of descriptors/labels from the data files
	// NB. The gathering of data seems wasteful but SVM.train() takes so much more time that it does not matter
	for (int one_out=0; one_out < Data_Files.size(); one_out++) {
		DataDescriptors data;  // we will store the data here (should clear itself in each loop)

		// We will leave this one out and use its name for the model
		string modelFile=Data_Files[one_out] + string(SVM_SUFFIX);
		cout << "Leaving '" << Data_Files[one_out] << "' out\n";

		// we now gather data from all the other ones
		for (int one_in=0; one_in < Data_Files.size(); one_in++) {
			if (one_in == one_out) continue; // ignore the one we are leaving out
			cout << "         In: '" << Data_Files[one_in] << "'\n";
		    getDataDescriptors (data_path+"/"+Data_Files[one_in]+DATA_SUFFIX, data);
		} // one_in

		Size oldlabelsSize = data.labels.size();
		Size olddescriptorsSize = data.descriptors.size();
		cout << "Result data labels " << oldlabelsSize.height << " x " << oldlabelsSize.width
			 << endl;
		cout << "Result Descriptors " << olddescriptorsSize.height << " x "
			 << olddescriptorsSize.width << endl;

// We change the labels (hypothesis: Dalal used -1 to indicate positives?)
#ifdef CHANGELABELS
		cout << "**** changing response labels *****" << endl;
		for (int i=0; i<oldlabelsSize.height; i++)
			switch (data.labels.at<int>(i, 0)) {
				case 1:
					data.labels.at<int>(i, 0) = -1;
					break;
				case -1:
					data.labels.at<int>(i, 0) = 1;
					break;
				default:
					cout << "!!!! unexpected label " << data.labels.at<int>(i, 0) <<
						"at i: " << i << endl; 
			}
#endif


		// OK, we have everything else except one_out, now we train one SVM and wait wait ... 
	    if (SVMTrain (data, svm_path, modelFile, SVM_CPARAMETER, SVM_GPARAMETER)) {
			cout << "Something appeared to have gone wrong! \n";
		}
	} // for each file (one_out)
	return 0;

}
