// (c) Sergio A Velastin 2016 UC3M

// This is a class derived from OpenCV's HOGDescriptor to allow variants
// We found that in OpenCV 2.4 detection does NOT work if we supply constructor parameters
// beyond win_sigma (i.e. from threshold_L2hys), even if we use what are supposed to be
// default values (e.g. just adding threshold_L2hys=0.2 makes detection fail (nothing
// is detected)

#ifndef BOSSHOG_HPP
#define BOSSHOG_HPP

#include <opencv2/opencv.hpp>

#include <vector>

#if CV_MAJOR_VERSION >= 3
#include <opencv2/objdetect.hpp>
#else
#include <opencv2/ocl/ocl.hpp>
#endif

using namespace cv;
using namespace std;


// Derived class from HOGDescriptor to add its own trained model
class BOSSHog : public HOGDescriptor{
	public:
	  BOSSHog(Size win_size, Size block_size, Size block_stride, Size cell_size, int nbins); 
//double win_sigma, double threshold_L2hys, bool gamma_correction, int nlevels);
//	  BOSSHog(Size win_size=Size(64, 128), Size block_size=Size(16, 16), Size block_stride=Size(8, 8), Size cell_size=Size(8, 8), int nbins=9, double win_sigma=DEFAULT_WIN_SIGMA, double threshold_L2hys=0.2, bool gamma_correction=true, int nlevels=DEFAULT_NLEVELS);

	  vector<float> getBOSSPeopleDetector(); // this is for a hard-coded detector instead of a model loaded from file
};

#endif
