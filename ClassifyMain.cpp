// (c) S.A. Velastin 2016 (UC3M)
// Derived from originals by Miguel Jara and Diego Gomez, USACH, 2015
// This is run after TrainLOO
// We assume that there is a set of trained SVMs (using leave one out) and a corresponding set
// of negative and positive samples
// Classes are "sit", "sitting" and "standing". The SVMs have been trained separately
// for each class so they can be assessed separately or fused...
// TODO: Make this program more modular so that we can have variants


/*
 * 3.1	Added calculation of mean SVM distances to hyperplane
 */

#define VERSION 3.1

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <deque>
#include <vector>
#include <fstream>

#include <sstream>

#include <algorithm>

#include <cmath>
#include "tool.hpp"
#include "LinearSVM.hpp"

using namespace cv;
using namespace std;

// if we are using "negative logic" for positives and negatives
#ifdef CHANGELABELS
#define POS_LABEL -1
#define NEG_LABEL 1
#else
#define POS_LABEL 1
#define NEG_LABEL -1
#endif

#define SVM_SUFFIX "-SVM.xml"

// Dimensions of normalised samples (for BOSS, data is already resized to this)
#define WIDTH 64
#define HEIGHT 128


//******************* TMeanValues: to compute running averages **********
class TMeanValues {
public:
  double	mean;
  long		samples;
  bool		inited;
  
  void add_sample (double sample);
  TMeanValues (void);
};

TMeanValues::TMeanValues (void) {
  inited = false;
}

void TMeanValues::add_sample (double sample) {
  if (!inited) {
    mean = sample;  // this is the first sample
    samples = 1;
    inited = true;
    return;
  }
  samples++; // this is calculation of running average
  mean = sample/samples+ (samples-1)*mean/samples;
  return;
}

//**********************************************************************************
// Return value:
//	1: positive
//	-1: negative
// 	0: something did not quite work!

int Process_Image (LinearSVM &SVM, string folderpath, string image_file, int expected, Size resizeSize, double &confidence)
{	Mat image;
	Mat imageGray;
	Mat resizeImage;

	string imagePath = folderpath + "/" + image_file;
	cout << "***** Process_Image: ";

	// this is OpenCV reading the image
	image = imread(imagePath);
	if (image.data == NULL) {
		cout << "Something went wrong when reading the file\n";
		return 0;
	}
	cvtColor(image, imageGray, CV_BGR2GRAY);  // Convert to grey image
	Size ImSize = image.size();  // find out size
	cout << "Image read " << ImSize.width << "," << ImSize.height 
		<< " (want " << resizeSize.width << "," << resizeSize.height << "): ";
	if (ImSize == resizeSize)
		resizeImage = imageGray; // no need to call resize
	else
		resize(imageGray, resizeImage, resizeSize, CV_INTER_LINEAR); // standard size

	// Prepare things for HOG
	Size win_size=resizeSize; // input image size
	Size block_size=Size(16, 16); 
	Size block_stride=Size(8, 8); 
	Size cell_size=Size(8, 8);
	int nbins=9;
	HOGDescriptor hog(win_size, block_size, block_stride, cell_size, nbins);
	vector<float> descriptor;
		
	// Compute the HOG
	hog.compute(resizeImage,descriptor);

	// Convert to Mat
	Mat descriptorMat(1,descriptor.size(),CV_32FC1);
	for(int i =0; i < descriptor.size(); i++){
		descriptorMat.at<float>(0,i) = descriptor [i];
	}
	// Now see what the SVM makes of it
	int valueDF;	
	confidence=SVM.predict(descriptorMat, true);
	valueDF = SVM.predict(descriptorMat); // should return the class label
	cout << valueDF << "(" << confidence << ") Expected: " << expected;
	if (valueDF != expected) cout << " *** wrong!!!";
	cout << endl;
	return valueDF;
}

//**********************************************************************************
void Process_Data (string SVM_fullfile, string basename, string pos_path, 
					string neg_path, ofstream &results_file)
{	cout << "**** Process data, SVM: '" << SVM_fullfile << "' Basename: '" << basename
		<< "' Pos path: " << pos_path << "' Neg path: '" << neg_path << "'\n";
	results_file << basename << ",";	// start results line

	// we now load the SVM
	LinearSVM SVM;
	Size resize= Size(WIDTH, HEIGHT);
	vector<float> support_vector;
	
	SVM.load(SVM_fullfile.c_str());  // TODO: need to check if successful
	pause ("SVM model file loaded");
	support_vector = SVM.getSupportVector(); // TODO: need to check if successful
	cout << "Support vector size: " << support_vector.size() << endl;
	pause ("SVM support vector loaded");

	// Let's deal with the negatives first
	string Directory = neg_path + "/" + basename;
	vector <string> Files = arrayFilesName(Directory);
	int total_neg=0, total_pos=0, true_neg=0, false_pos=0, false_neg=0, true_pos=0;
	TMeanValues MeanTP,MeanFP,MeanFN,MeanTN; 
	cout << "Negatives Directory: '" << Directory << "' N files: " 
		<< Files.size() << endl;
	for (int i= 0; i<Files.size(); i++) {
	  double confidence;
		cout << "'" << Files[i] << "'\n";
		switch (Process_Image (SVM, Directory, Files[i], NEG_LABEL, resize, confidence)) {
			case 0: // there was a problem, ignore
			break;
			case NEG_LABEL: // a true negative
				total_neg++;	// a negative ground truth
				true_neg++;
				MeanTN.add_sample(confidence);
				cout << "Mean TN " << MeanTN.mean << endl;
			break;
			case POS_LABEL: // a false positive
				total_neg++;
				false_pos++;
				MeanFP.add_sample(confidence);
				cout << "Mean FP " << MeanFP.mean << endl;
		}
	}
	results_file << total_neg << "," << true_neg << "," << false_pos;
	pause("Those were the negative files");


	// Now the positives
	Files.clear();
	Directory = pos_path + "/" + basename;
	Files = arrayFilesName(Directory);
	int number_pos = Files.size();
	cout << "Positive Directory: '" << Directory << "' N files: " 
		<< Files.size() << endl;
	for (int i= 0; i<Files.size(); i++) {
	  double confidence;
		cout << "'" << Files[i] << "'\n";
		switch (Process_Image (SVM, Directory, Files[i], POS_LABEL, resize, confidence)) {
			case 0: // there was a problem, ignore
			break;
			case NEG_LABEL: // a false negative
				total_pos++;	// a positive ground truth
				false_neg++;
				MeanFN.add_sample(confidence);
				cout << "Mean FN " << MeanFN.mean << endl;
			break;
			case POS_LABEL: // a true positive
				total_pos++;
				true_pos++;
				MeanTP.add_sample(confidence);
				cout << "Mean TP " << MeanTP.mean << endl;
		}
	}
	results_file << "," << total_pos << "," << true_pos << "," << false_neg << "," 
		<< total_pos+total_neg << ",";
	// Now work out precision & recall
	if ((true_pos+false_pos)>0)
		results_file << (float) true_pos / (true_pos+false_pos);  // precision
	else results_file << "0";
	results_file << ",";
	if ((true_pos+false_neg)>0)
		results_file << (float) true_pos / (true_pos+false_neg);  // recall
	else results_file << "0";
	results_file << "," << MeanTP.mean << "," << MeanTN.mean << "," << MeanFP.mean << "," << MeanFN.mean;
	results_file << endl;
	pause("Those were the positive files");

	return;
}



// ********************************************************************
void displayUsage(){
	cout << "./Classify -s path -n path -p path -r path" << endl;
	cout << "-s path: path (no trailing /) to find SVM model files" << endl;  
	cout << "-n path: path (no trailing /) to find negative samples" << endl;
	cout << "-p path: path (no trailing /) to find positive samples" << endl;
	cout << "-r path: path (no trailing /) where to save output results (text, metrics)\n";
}

// *********************************************************************
int main( int argc, char** argv ) {
	char opt;
	string svm_path, pos_path, neg_path, results_file;
	Size data_size = Size(WIDTH,HEIGHT);  // this is the image dimensions to which the samples need resizing (but we will assume they are of this size!)


	if(argc < 9){
		cerr << "Missing arguments" << endl;
		displayUsage();
		return -1;
	}

	// Deal with the command line, see http://linux.die.net/man/3/optarg for handling commands
	while((opt = getopt(argc, argv, ":n:s:p:r:")) != -1){
		switch(opt){
			case 's':
			svm_path = optarg;
			break;
			case 'n':
			neg_path = optarg;
			break;
			case 'p':
			pos_path = optarg;
			break;
			case 'r':
			results_file = optarg;
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

	cout << "SVM models on '" << svm_path << "', Negatives on '" << neg_path 
		<< "Positives on '" << pos_path << "' Results on '" << results_file << "'\n";
	if (svm_path == "") { cout << "Hey! svm path empty\n"; return 2; }
	if (neg_path == "") { cout << "Hey! negatives path empty\n"; return 2; }
	if (pos_path == "") { cout << "Hey! positives path empty\n"; return 2; }
	if (results_file == "") { cout << "Hey! results path empty\n"; return 2; }

	// Will now try to access the lists of input files
	// Actually this is what determines other filenames so strictly, no need to find
	// files in the other directories
	vector <string> SVM_Directories = arrayFilesName(svm_path);
	sort (SVM_Directories.begin(), SVM_Directories.end());
	for (int i=0; i<SVM_Directories.size(); i++)
		cout << "'" << SVM_Directories[i] << "'\n";
	pause ("That was the SVM directories");

	vector <string> Neg_Directories = arrayFilesName(neg_path);
	sort (Neg_Directories.begin(), Neg_Directories.end());
	for (int i=0; i<Neg_Directories.size(); i++)
		cout << "'" << Neg_Directories[i] << "'\n";
	pause ("That was the negatives directories");

	vector <string> Pos_Directories = arrayFilesName(pos_path);
	sort (Pos_Directories.begin(), Pos_Directories.end());
	for (int i=0; i<Pos_Directories.size(); i++)
		cout << "'" << Pos_Directories[i] << "'\n";
	pause ("That was the positives directories");

	pause("\nOk, we have the directory entries, so we proceed");
	string results_path=results_file.substr(0,results_file.find_last_of("/"));

	struct stat st = {0};
	if (stat(results_path.c_str(), &st) == -1) { // if directory does not exist then create it
	    mkdir(results_path.c_str(), 0700);
	}
    ofstream data_results(results_file.c_str()); // TODO: check if opened ok
	data_results << pos_path << endl;
	data_results << "Set," << "Negs" << "," << "TN" << "," << "FP,";  // header
	data_results << "Pos" << "," << "TP" << "," << "FN" << "," 
		<< "Total" << ",Precision,Recall"; 
	data_results << ",MeanTP,MeanTN,MeanFP,MeanFP" << endl;


	// TODO: add -n1 -n2 to include as negatives the positives of the other classes
	// TODO: Really we should only test on Inter_100 set as that contains most (all) GT positives
	// ... here we do the processing
	for (int test_set=0; test_set<SVM_Directories.size(); test_set++) {
		// We will test classification performance for test_set
		string SVM_file = SVM_Directories[test_set];	// this is the SVM model file
		size_t pos= SVM_file.rfind(SVM_SUFFIX, string::npos);
		if (pos == string::npos) {
			cout << "mmmmm.... did not find the SVM suffix in '" << SVM_file << "'\n";
			continue; // jump to the next one (this is a fail case)
		}
		// Get basename: should correspond to positive and negative directories
		string basename = SVM_file.substr(0, pos);
		// this is the path qualified name of the SVM file:
		string SVM_fullfile= svm_path+"/"+SVM_file;		

		cout << "'" << SVM_file << "', Basename '" << basename << "'\n";

		Process_Data (SVM_fullfile, basename, pos_path, neg_path, data_results);
	} // for test_set
	data_results.close();
	return 0;
}
