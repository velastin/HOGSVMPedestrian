#include "tool.hpp"


vector <string> arrayFilesName(string folderPath){
    
    vector<string> files;

    DIR *pDIR;
    struct dirent *enter;
    if( pDIR = opendir(folderPath.c_str()) ){
        while(enter = readdir(pDIR)){
                string fileName = enter->d_name;
                if( fileName != "." && fileName != ".." ){
                    files.push_back(fileName);
                }
        }
        closedir(pDIR);
    }

    return files;
}

void ShowBar(int percentage,int sizeMax){
    stringstream ss;
    ss << " " << percentage << "%";
    string Msg = ss.str();
    int i = 0,a = 0;
        for(i = sizeMax+Msg.size()+2; i>=0; --i) putchar('\b'); // vamos al principio de la barrita
    
    putchar('[');
        for(i = (percentage*sizeMax)/100; a<i; ++a) putchar('#');
        for(; a < sizeMax; ++a) putchar('-');
    putchar(']');
    
    if(percentage != 100){
    	cout << Msg;
    }
    else{
    	cout << Msg <<endl;
    }
    fflush(stdout);
}

void hard_pause(string msg)
{
    std::string dummy;
    std::cout << msg << "... (press RETURN to continue)";
    std::getline(std::cin, dummy);
}

void pause(string msg)
{
    std::string dummy;
    std::cout << msg << "... ";
#ifdef ENABLEPAUSE
	std::cout << "(press RETURN to continue)";
    std::getline(std::cin, dummy);
#else
	std::cout << endl;
#endif
}

// Appends data descriptors from a given file. See CreateDescriptors() for details on how file is created
int getDataDescriptors(string descriptorsFile, DataDescriptors &data){

	cout << "****** Get Data Descriptors '" << descriptorsFile << "'\n";
//    DataDescriptors data;
    int numDescrips;
    int numFeatures;

    string line;
    ifstream file(descriptorsFile.c_str());
	if (!file.failbit) {
		pause ("*** error: unable to open the file");
		return 1;
	}

	// ***SAV: why not use stream read?
    if(getline(file, line)){  // the first line has number_of_pos_plus_neg space number_of_features
        stringstream sline(line);
        string token;

        getline(sline, token, ' ' );
        numDescrips = atoi(token.c_str());
        getline(sline, token, ' ' );
        numFeatures = atoi(token.c_str());  
    }

    cout << numDescrips << " " << numFeatures << endl;

	// here we remember the size of the data so far (on entry to this function)
	Size oldlabelsSize = data.labels.size();
	Size olddescriptorsSize = data.descriptors.size();
	cout << "In current data labels " << oldlabelsSize.height << " x " << oldlabelsSize.width << endl;
	cout << "Descriptors " << olddescriptorsSize.height << " x " << olddescriptorsSize.width << endl;
	pause ("Now it will create space and read the file ...");

	// We create the data structure to hold the data
    Mat allLabels (numDescrips, 1, CV_32S);	// for the class labels (SAV: changed to int)
    Mat allDescriptors (numDescrips, numFeatures, CV_32FC1);	// for the features

    int i = 0;
    int j = 0;

    while(getline(file, line)){ // we now process each line in turn
        stringstream sline(line);
        string token;

        bool first = true;
        float label;
        float num;

        while(getline(sline, token, ' ' )){
            if(token == "#"){ // a "#" indicates the end of the data
                break;
            }

            if(first){ // this is the class value (1: pos, -1: neg)
                label = atoi(token.c_str());
                allLabels.at<int>(i, 0) = label;
                first = false;
            }
            else{
                num = atof(token.c_str());
                allDescriptors.at<float>(i, j) = num;
                j++;
            }
        }
/*
		// to check that we have read things correctly
		if (i==0) {
			cout << allLabels.at<float>(i,0) << endl;
			for (int k = 0; k < j; k++)
				cout << allDescriptors.at<float>(i,k) << " ";
			cout << ":" << i << endl;
			pause("that was the first descriptor");
		}
*/
        ShowBar(i*100/(numDescrips-1),50);

        i++;  // for next line
        j = 0; // reset
    }

    file.close();

    cout << "matrix size: " << allDescriptors.rows << " " << allDescriptors.cols << endl;
    cout << "labels size: " << allLabels.rows << " " << allLabels.cols << endl;
	pause ("Now to add newly read data ..");


	if (oldlabelsSize.height == 0) {
    	data.labels = allLabels;
    	data.descriptors = allDescriptors;
	} // when size was zero
	else
		for (int i=0; i<numDescrips; i++) {
			data.labels.push_back(allLabels.row(i));  // add these new rows to the data (would this work above?)
			data.descriptors.push_back(allDescriptors.row(i));
		} // for each data

	oldlabelsSize = data.labels.size();
	olddescriptorsSize = data.descriptors.size();
	cout << "In new data labels " << oldlabelsSize.height << " x " << oldlabelsSize.width << endl;
	cout << "Descriptors " << olddescriptorsSize.height << " x " << olddescriptorsSize.width << endl;
    return 0;
}

