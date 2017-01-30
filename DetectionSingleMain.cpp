// (c) S.A. Velastin 2016 (UC3M)
// Derived from originals by Miguel Jara and Diego Gomez, USACH, 2015
// This is run after TrainLOO
// This takes one pre-trained SVM and a video and displays detection results
// Mainly used for a quick check that detection works (or not!)

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
#include "BOSSHog.hpp"

using namespace cv;
using namespace std;

#define SVM_SUFFIX "-SVM.xml"
#define VIDEO_SUFFIX ".avi"
#define GT_SAMPLE_RATE 8	// one out of these many frames in the ground truth

//#define DATA_SUFFIX "-DESC.dat"

// Dimentions of normalised samples (for BOSS, data is already resized to this)
// Jan 2017: adapted for UANDES dataset
// TODO: make these command-line parameters
#define WIDTH 56
#define HEIGHT 56

// ***************  This is the detector, callig OpenCV's HOG detector
// hog:			has been previously setup with appropriate model and parameters
// imagegray:	the input gray-level image
// scale0:		this defines the range of scales in the detector e.g. 1.05
vector <Rect> detector(BOSSHog &hog, Mat imageGray, double scale0)
{
	vector <Rect> found;	// OpenCV's detector returns the results here

	// and do the detection
	cout << "Calling the detector\n";
//	hog.detectMultiScale(imageGray, found);  // other parameters by default
	hog.detectMultiScale(imageGray, found, 0, Size(8,8), Size(0,0), scale0, 2);

	// Here we could filter what has been found e.g. using size rules etc.
	return found;
}


// ******************************************************************************
// Processes a video file given an SVM model
// VideoFilename:	Full path for video file
// SVM_FullFile:	Full path of SVM model file
// VideoFile:		Name to be used for the video window
int Process_Video (string VideoFileName, string SVM_FullFile, string VideoFile)
{	LinearSVM SVM;		// will hold the model
	vector<float> support_vector;
	
	cout << "SVM model file on '" << SVM_FullFile << "'\n";
	SVM.load(SVM_FullFile.c_str());  // TODO: need to check if successful
	pause ("SVM model file loaded");
	support_vector = SVM.getSupportVector(); // TODO: need to check if successful
	pause ("SVM support vector loaded");

	// Define the HOG detector
	Size block_size=Size(16, 16); 
	Size block_stride=Size(8, 8); 
	Size cell_size=Size(8, 8);
	int nbins=9;
	Size Train_Size = Size(WIDTH, HEIGHT);  // the training size
	double scale0 = 1.05;	// needed by hog detector

	// construct a hog (detector) object
//	HOGDescriptor hog (Train_Size, block_size, block_stride, cell_size, nbins, 
//			-1, 0.2, true, 64);
//	HOGDescriptor hog;
	BOSSHog hog (Train_Size, block_size, block_stride, cell_size, nbins);

	// Mainly historical, we output different possible detector sizes on screen
	vector <float> s_vector = hog.getDefaultPeopleDetector();	// OpenCV's default
	cout << "size 64x128 default " << s_vector.size() << endl;
	s_vector.clear();
	s_vector = hog.getBOSSPeopleDetector();
	cout << "size 64x128 BOSS " << s_vector.size() << endl;		// BOSS default
	cout << "size support " << support_vector.size() << endl;	// the model we have read
	pause ("press me");
	
	// This uses OpenCV's default detector (for 64x128)
	 //hog.setSVMDetector(hog.getDefaultPeopleDetector());
	 //cout << "*** using default HOG detector 64x128 ****" << endl;

	// This uses our hard-coded detector (for 64x128)
	// hog.setSVMDetector(hog.getBOSSPeopleDetector());
	// cout << "**** using default BOSS detector *****" << endl;

	// This uses what we read from the XML file
	hog.setSVMDetector (support_vector);
	cout << "**** Using model from file *****" << endl;

	cout << "Process Video '" << VideoFileName << "'\n";
	// Try to open the file
	VideoCapture vc(VideoFileName.c_str() );
	if(!vc.isOpened()){
		cerr << VideoFileName <<" Video File Read Error." << endl;
		return 1;	
	}
    int ex = static_cast<int>(vc.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form

    // Transform from int to char via Bitwise operators
    char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};

    Size S = Size((int) vc.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) vc.get(CV_CAP_PROP_FRAME_HEIGHT));

	int total_frames=vc.get(CV_CAP_PROP_FRAME_COUNT);

    cout << "Input frame resolution: Width=" << S.width << "  Height=" << S.height
		<< " Train width=" << Train_Size.width << " Train height=" << Train_Size.height
         << " of nr#: " <<  total_frames << endl;
	int frame_ms = 1000/vc.get(CV_CAP_PROP_FPS);
 	cout << "FPS: " << vc.get(CV_CAP_PROP_FPS) << " ms: " << frame_ms << endl;
 	cout << "Input codec type: " << EXT << endl;
	hard_pause ("We will now start the video process");
	
	// ******************* Process the video ***********************************
	// create a window for display
	namedWindow(VideoFile.c_str() , CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
    for (int frame=0; frame < total_frames; frame++) {
		Mat imageColor;
		Mat imageGray;
		vector<Rect> found; // what the hog detector will return

		vc >> imageColor;	// neet way to read a frame from the video!
		if(imageColor.empty()) {
			cout << "*** warning, error when reading frame " << frame << endl;
			continue;  // but carry on with the next one
		}
		// frame read ok, carry on processing
		cvtColor(imageColor, imageGray, CV_BGR2GRAY);  // convert to gray for processing
//		ShowBar(fame*100/(total_frames-1),50);  
		found = detector(hog, imageGray, scale0); // the detector
		for(int i = 0; i < found.size(); i++ ){
			Rect r = found[i];
			cout << frame << "," << "1," << r.x << "," << r.y << "," << r.width
				<< "," << r.height << endl; 
			rectangle(imageColor, r, cv::Scalar(0,0,255), 1); // draw the rectagle on the image
		}
		imshow(VideoFile.c_str(), imageColor);  // show the image

		// TODO: metrics and generate output
		// ...
		// frame processed
		int c = waitKey(1);	// wait for the user or the end of the frame period
		if( c == 'q' || c == 'Q' ) break;  // user stops
	} // we loop until no more frames to process
	destroyWindow(VideoFile.c_str());
	// TODO output some statistics, write results somewhere...
	return 0; // success
}

// ********************************************************************
void displayUsage(){
	cout << "./Detection -s path -v path" << endl;
	cout << "-s path: path (no trailing /) to find SVM model file" << endl;  
	cout << "-i path: path (no trailing /) to find original input video" << endl;
}

// *********************************************************************
int main( int argc, char** argv ) {
	char opt;
	string svm_path, in_video_path;
	Size data_size = Size(WIDTH,HEIGHT);  // this is the image dimensions to which the samples need resizing


	if(argc < 5){
		cerr << "Missing arguments" << endl;
		displayUsage();
		return -1;
	}

	// Deal with the command line, see http://linux.die.net/man/3/optarg for handling commands
	while((opt = getopt(argc, argv, ":s:i:")) != -1){
		switch(opt){
			case 's':
			svm_path = optarg;
			break;
			case 'i':
			in_video_path = optarg;
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

	cout << "SVM model on '" << svm_path << "', Video on '" << in_video_path << endl;
	if (svm_path == "") { cout << "Hey! svm file name empty\n"; return 2; }
	if (in_video_path == "") { cout << "Hey! input video file name empty\n"; return 2; }


	pause("\nOk, we have the directory entries, so we proceed");
	Process_Video (in_video_path, svm_path, "Video");

	return 0;
}
