#ifndef FUNCTIONS_CPP
#define FUNCTIONS_CPP

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <deque>
#include <iostream>
#include <string>
#include <sstream>
#include "quick_tinyxml/tinyxml.h"

using namespace std;
using namespace cv;

struct Cuerpo{
    int id;
	int inicio;
	int fin;
	int x;
	int y;
	int h;
	int w;
};

struct Estado{
    int id;
	int inicio;
	int fin;
	int estado;
};

void readXMLFile(string input_arch, deque<deque<Cuerpo> >& cuerpos, deque<deque<Estado> >& estados, unsigned int &total_frames);

bool checkNear(float percentage, Rect rec, int pos, deque<deque<Cuerpo> >& cuerpos, int virtualFrame, int num_bodies);

bool checkNegative(Rect &neg_rec, int pos, deque<deque<Cuerpo> >& cuerpos, int virtualFrame, int num_bodies, int height, int width, Size size);


#endif
