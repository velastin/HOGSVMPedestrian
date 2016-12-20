// S.A. Velastin 2016, based on original by Diego Gomez (2015, USACH)


#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <deque>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <stdlib.h>

#include "functions.hpp"
#include "tool.hpp"

//This program gets positives and negatives samples from videos.

using namespace cv;
using namespace std;

#define PERCENTAGE 0.1 //Margin added to each direction

// ***SAV dimentions of normalised samples
#define WIDTH 128
#define HEIGHT 256

// ***SAV added this to define extension of output files
#define SAMPLEFILEEXT ".png"
#define OUTFILEEXT ".csv"

// and this defines the 1 in X frames chosen in the ground truth
#define FRAMEREDUCTION 8

//Information about how to use program.
// In this version we only output to the screen ... so -o path is ignored
void displayUsage(){
	cout << "How to use?: " << endl;
	cout << "./GetGT -g path -o path" << endl;
	cout << "-g path: path where to find ground truth files " << endl;  
	cout << "-o path: output path" << endl; 
}



// The main program!
int main(int argc, char** argv){

	srand(time(NULL));

	char opt;
	string input_path, output_path;

	if(argc < 5){
		cerr << "Missing arguments" << endl;
		displayUsage();
		return -1;
	}

// ***SAV, see http://linux.die.net/man/3/optarg for handling commands
	while((opt = getopt(argc, argv, ":g:o:")) != -1){
		switch(opt){
			case 'g':
			input_path = optarg;
			break;
			case 'o':
			output_path = optarg;
			break;
			case '?':  // ***SAV not sure why "?" in particular
			cerr << "Invalid option:  '" << char(optopt) << "' doesn't exist." << endl << endl;
			displayUsage();
			exit(0);
			default:
			cerr << "Missing value for argument: '" << char(optopt) << "'" << endl << endl;
			displayUsage();
			exit(0); 
		}
	}
	if (input_path == "") { cout << "Hey! input path empty\n"; return 2; }
	if (output_path == "") { cout << "Hey! output path empty\n"; return 2; }
	vector <string> GT_files = arrayFilesName(input_path);
	sort (GT_files.begin(), GT_files.end());
	for (int i=0; i < GT_files.size(); i++) 
		cout << "'" << GT_files[i] << "'\n";
	pause ("Those are the GT files. Now we will process each one");

	// Output folder
	cout << "Text GT stored on: '" << output_path << "' directory\n";
	struct stat st = {0};
	if (stat(output_path.c_str(), &st) == -1) { // if directory does not exist then create it
	    mkdir(output_path.c_str(), 0700);
	}


	for (int i=0; i < GT_files.size(); i++) {
		unsigned int total_frames;
		string basename = GT_files[i].substr(0,GT_files[i].find_last_of ("."));
		string outfile=output_path+"/"+basename+OUTFILEEXT;
		cout << "Basename '" << basename << "' Outfile '"<< outfile << "'\n";
		ofstream data(outfile.c_str());

		deque<deque<Cuerpo> > cuerpos; //deque with information about coordinates
		deque<deque<Estado> > estados; //deque with information about state of each people
		readXMLFile(input_path+"/"+GT_files[i], cuerpos, estados, total_frames);	
	
    	if(cuerpos.empty()){ 	//Theres are no people in the video
			cout << "No people have been found in the video." << endl;
			data.close();
			continue;
		}
		cout << "Read file '" << GT_files[i] << "'\n";
		cout << "Size cuerpos: " << cuerpos.size() << ", estados: " << estados.size() 
			<< ", Frames: " << total_frames << endl;
		// First the bodies (cuerpos)
		for (int j=0; j<cuerpos.size(); j++) {
			int cuerpo_size = cuerpos.at(j).size();
			cout << "Cuerpo " << j << " Sizej " << cuerpo_size << endl;
			for (int k=0; k <cuerpo_size; k++) {
				cout << "Id " << cuerpos.at(j).at(k).id << " Start "
					<< cuerpos.at(j).at(k).inicio << " End "
					<< cuerpos.at(j).at(k).fin << " x "
					<< cuerpos.at(j).at(k).x << " y "
					<< cuerpos.at(j).at(k).y << " w "
					<< cuerpos.at(j).at(k).w << " h "
					<< cuerpos.at(j).at(k).h << endl;
				data << cuerpos.at(j).at(k).id << " " 
					<< cuerpos.at(j).at(k).inicio*FRAMEREDUCTION << " " 
					<< cuerpos.at(j).at(k).fin*FRAMEREDUCTION << " "
					<< cuerpos.at(j).at(k).x << " "
					<< cuerpos.at(j).at(k).y << " "
					<< cuerpos.at(j).at(k).w << " "
					<< cuerpos.at(j).at(k).h << endl;
			}
		}
		// now the states (estado: classes)
		for (int j=0; j<estados.size(); j++) {
			int estado_size = estados.at(j).size();
			cout << "Class " << j << " Sizej " << estado_size << endl;
			for (int k=0; k <estado_size; k++) {
				cout << "Id " << estados.at(j).at(k).id << " Start "
					<< estados.at(j).at(k).inicio << " End "
					<< estados.at(j).at(k).fin << " Class "
					<< estados.at(j).at(k).estado << endl;
			}
		}

		pause("");
	} // for each file
	return 0;
}


