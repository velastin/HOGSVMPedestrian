// (c) S.A. Velastin 2016 (UC3M)
// Based on original by Miguel Jara, USACH 2015
// Uses grid training (c, gamma) to obtain best RBF SVM for a given training set
// Gets descriptors previously sent to files

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
//#define SVM_CPARAMETER 0.002  // Diego 0.02  *** This is not needed

// Undefine this for linear in the cmakelist file
// #define TRAIN_SVM_RBF

// Dimensions of normalised samples (for BOSS, data is already resized to this)
//#define WIDTH 64		*** not needed
//#define HEIGHT 128

// These are settings for the parameter search grid
#define CGRID_START 1E-15				// pow(2,-30)
#define CGRID_END   1E-08				// pow(2,-20)
#define CGRID_STEP 2				
#define GGRID_START 0.000030518		// pow(2,-15)
#define GGRID_END 8					// pow(2,3)
#define GGRID_STEP 4				// pow(2,2)


void displayUsage(){
	cout << "./TrainGrid -d path -s path" << endl;
	cout << "-d path: path (no trailing /) where descriptor files are" << endl;  
	cout << "-s path: path (no suffix) full pathname of file where to store SVM model" 
		<< endl;  
}

//***************************************  SVM train ***************************************
// Finds optimal SVM for the training data passed in data
int SVMTrain (DataDescriptors &data,
				string modelFileName) 

{	string results_path=modelFileName.substr(0,modelFileName.find_last_of("/"));

    CvSVM SVM; // construct a default SVM (RBF ...)
	CvSVMParams params;  // a set of parameters
	params = SVM.get_params();	// we make sure we have a complete definition of all params

#ifndef TRAIN_SVM_RBF
	params.kernel_type = CvSVM::LINEAR; // The linear case
	cout << "Will train with a linear model\n";
#else
	cout << "Will train with an RBF model\n";
#endif

	CvParamGrid CvParamGrid_C(CGRID_START, CGRID_END, CGRID_STEP);
	cout << "Grid Start " << CvParamGrid_C.min_val << " End " << CvParamGrid_C.max_val << " Step " << CvParamGrid_C.step << endl;
	hard_pause("that is the C Grid");

	CvParamGrid CvParamGrid_gamma(GGRID_START, GGRID_END, GGRID_STEP); // if linear, it does not matter
	if (!CvParamGrid_C.check() || !CvParamGrid_gamma.check()) {
    	cout<<"The grid is NOT VALID."<<endl;
		return 1;
	}

	// If output path does not exist, then create it
    struct stat st = {0};
	if (stat(results_path.c_str(), &st) == -1) {
	    mkdir(results_path.c_str(), 0700);
	}

	// Now we are ready to find optimum SVM
	pause("Ready to start training");

	clock_t begin,end; // to time the process
	double elapsed;

	begin = clock();

#ifdef TRAIN_SVM_RBF
// RBF case
	SVM.train_auto(data.descriptors, data.labels, Mat(), Mat(), params,10, CvParamGrid_C,
	 CvParamGrid_gamma, CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU),
	 CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), true);
#else
// Linear case
	SVM.train_auto(data.descriptors, data.labels, Mat(), Mat(), params,10, CvParamGrid_C,
	 CvSVM::get_default_grid(CvSVM::GAMMA), CvSVM::get_default_grid(CvSVM::P),
	 CvSVM::get_default_grid(CvSVM::NU),
	 CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), true);
#endif

	end = clock();
	elapsed = double(end - begin) / CLOCKS_PER_SEC;

	params= SVM.get_params();
	cout<<"gamma: "<<params.gamma<<endl;
	cout<<"C: "<<params.C<<endl;
	cout<<"Time elapsed (secs): " << elapsed << " (" << int(elapsed/60) << " mins)\n";	
	string SVMFile=results_path+SVM_SUFFIX;
    cout << "Saving SVM in " << SVMFile << endl;
    SVM.save(SVMFile.c_str());
    cout << "COMPLETED SUCCESSFULLY" << endl;

    return 0;

}



//*******************************************************************************************
int main( int argc, char** argv ) {
	char opt;
	string data_path, svm_path;
//	Size data_size = Size(WIDTH,HEIGHT);  // this is the image dimensions to which the samples need resizing *** Not needed

#ifdef TRAIN_SVM_RBF
	hard_pause("I will be using SVM RBF");
#else
	hard_pause("I will be using SVM LINEAR");
#endif

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
	DataDescriptors data; // this is where we store the descriptors
	for (int datak=0; datak < Data_Files.size(); datak++) {
		// we get the descriptors from the files
	    getDataDescriptors (data_path+"/"+Data_Files[datak]+DATA_SUFFIX, data);
	} // for each file (datak)
	// OK. we have the descriptors
	Size oldlabelsSize = data.labels.size();
	Size olddescriptorsSize = data.descriptors.size();
	cout << "Result data labels " << oldlabelsSize.height << " x " << oldlabelsSize.width 
		<< endl;
	cout << "Result Descriptors " << olddescriptorsSize.height << " x " 
		<< olddescriptorsSize.width << endl;

	// OK, we have all the data in
    if (SVMTrain (data, svm_path))
		cout << "Something appeared to have gone wrong! \n";
	return 0;
}
