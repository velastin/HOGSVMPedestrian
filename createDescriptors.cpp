#include "createDescriptors.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <sstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <dirent.h>

#include <cstdlib>
#include <cstdio>

#include <fstream>

#include "tool.hpp"
#include "descriptor.hpp"

using namespace cv;
using namespace std;


// Computes all the HOG descriptors from the path and the list of files
vector <Descriptor> hogAllDescriptorWithResize(string folderPath, vector<string> imagesName, Size resizeSize){

	vector<Descriptor> descriptors;
	descriptors.clear();

	cout << "***** hogAllDescriptorWithResize ***************\n";

	for(int i = 0; i < imagesName.size(); i++){
		string imagePath;
		string imageBaseName;
		Mat image;
		Mat imageGray;
		Mat resizeImage;

		imagePath = folderPath + "/" + imagesName [i];
		imageBaseName = imagesName [i].substr(0, imagesName [i].find_last_of("."));
//		cout << "'" << imagePath << "'(" << imageBaseName << ")\n";  // display but might mess up the progress bar


		// this is OpenCV reading the image
		image = imread(imagePath);
		cvtColor(image, imageGray, CV_BGR2GRAY);  // Convert to grey image
		Size ImSize = image.size();  // find out size
//		cout << "Image read " << ImSize.width << "," << ImSize.height << "\n";
		if (ImSize == resizeSize) {
			resizeImage = imageGray; // no need to call resize
		}
		else
			resize(imageGray, resizeImage, resizeSize, CV_INTER_LINEAR); // standard size

		Size win_size=resizeSize; // input image size
		Size block_size=Size(16, 16); 
		Size block_stride=Size(8, 8); 
		Size cell_size=Size(8, 8);
		int nbins=9;
		HOGDescriptor hog(win_size, block_size, block_stride, cell_size, nbins);
		vector<float> descriptor;
		
		//vector<Point> locs;
		//hog.compute(image,descriptor,Size(8,8),Size(0,0),locs);

		hog.compute(resizeImage,descriptor);

		Descriptor auxDescrip = {imageBaseName, descriptor};
		descriptors.push_back(auxDescrip);  // we build a list of descriptors

		ShowBar(i*100/(imagesName.size()-1),50);
	} // i (each filename) 

	return descriptors;
}

// This is to save the descriptors on a file
int saveDescriptors(vector<Descriptor> descriptorsPosImg, vector<Descriptor> descriptorsNegImg, string descriptorsPath){

	cout << "***** saveDescriptors  '" << descriptorsPath << "' *******\n";
	if(descriptorsPosImg.size() == 0 && descriptorsNegImg.size() == 0){
		return -1;
	}

	string descriptorFolderPath = descriptorsPath.substr(0,descriptorsPath.find_last_of("/"));
	cout << "Descriptors stored on: '" << descriptorFolderPath << "' directory\n";
	struct stat st = {0};
	if (stat(descriptorFolderPath.c_str(), &st) == -1) { // if directory does not exist then create it
	    mkdir(descriptorFolderPath.c_str(), 0700);
	}

	ofstream data(descriptorsPath.c_str());

	data << descriptorsPosImg.size() + descriptorsNegImg.size() << " ";

	if(descriptorsPosImg.size() == 0)
		data << descriptorsNegImg [0].descrip.size() << endl;
	else
		data << descriptorsPosImg [0].descrip.size() << endl;

	// Do the positives
	for(int i = 0; i < descriptorsPosImg.size(); i++){
		// data << "# " << descriptorsPosImg [i].baseName << endl; // This should not have been here, it generates extra line!
		data << "1 ";
		for(int j=0; j < descriptorsPosImg [i].descrip.size(); j++){
			data << descriptorsPosImg [i].descrip [j] << " ";
		}
		data << "# " << descriptorsPosImg [i].baseName << endl;

		ShowBar( (i*100/(descriptorsPosImg.size()-1)) / 2, 50);
	}

	// And now the negatives
	for (int i = 0; i < descriptorsNegImg.size(); i++){
		// data << "# " << descriptorsNegImg [i].baseName << endl; // See above
		data << "-1 ";
		for(int j=0; j < descriptorsNegImg [i].descrip.size(); j++){
			data << descriptorsNegImg [i].descrip [j] << " ";
		}
		data << "# " << descriptorsNegImg [i].baseName << endl;

		ShowBar( 50 + (i*98/(descriptorsNegImg.size()-1)) / 2, 50);
	}

	data.close();

	ShowBar( 100, 50);	
	return 0;
}

// Inputs are directories of positive and negative images and a path and filename of where to store the results
int createDescriptors(string positiveFolderPath,
						string negativeFolderPath,
						string descriptorFolderPath,
						string descriptorsFileName,
						Size resizeSize) 
{	vector<string> posImgsName;	// these will hold the file names within the directories
	vector<string> negImgsName;

	vector<Descriptor> descriptorsPosImg; // and the corresponding descriptors
	vector<Descriptor> descriptorsNegImg;

	cout << "\n****** Create Descriptors ***********\n";
	cout << "Positives: '" << positiveFolderPath << "'\nNegatives: '" << negativeFolderPath << "'\n";

	posImgsName = arrayFilesName(positiveFolderPath);	// get the filenames (positive and negative)
	sort(posImgsName.begin(), posImgsName.end());
	negImgsName = arrayFilesName(negativeFolderPath);
	sort(negImgsName.begin(), negImgsName.end());

	cout << "Descriptors Path '" << descriptorFolderPath << "/" << descriptorsFileName << "')\n";

	cout << "Number of Positives: " << posImgsName.size() << endl;
	cout << "Number of Negatives: " << negImgsName.size() << endl;

	if (posImgsName.size()) {
		cout << "Calculating descriptors of positive images" << endl;
		descriptorsPosImg = hogAllDescriptorWithResize(positiveFolderPath, posImgsName, resizeSize);
		cout << "Size descriptor 0: " << descriptorsPosImg [0].descrip.size() << endl;
		//for(int i=0; i< descriptorsPosImg.size(); i++){
			/*for(int j=0; j < descriptorsPosImg [0].size() ; j++ ){
				cout << "algo " << j << ": " << descriptorsPosImg [0][j] << endl;
			}*/
		//}
	}
cout << negImgsName.size() << endl;
pause("Now the negatives");

	if (negImgsName.size()) {
pause("we will do the negatives");
		cout << "Calculating descriptors of negative images" << endl;
		descriptorsNegImg = hogAllDescriptorWithResize(negativeFolderPath, negImgsName, resizeSize);
		cout << "size descriptor 0: " << descriptorsNegImg [0].descrip.size() << endl;
	}
pause("Now to save the descriptors");

	// Now save the descriptors
	string descriptorsFile = descriptorFolderPath + "/" + descriptorsFileName;
	cout << "Saving descriptors in '"<< descriptorsFile << "'" << endl;

	if(saveDescriptors(descriptorsPosImg, descriptorsNegImg, descriptorsFile)){
		cout << "Error: No Positive or Negative images" << endl;
		return -1;
	}

	cout << "COMPLETED SUCCESSFULLY" << endl;

	return 0;
}
