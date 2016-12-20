(c) S A Velastin UC3M, 2016
This is code based on Diego Gomez's and Miguel Jara's (both USACH 2015) to process video files
(BOSS and LosAndes dataset).

This version is compiled/linked with OpenCV 2.4 (OpenCV 3.x changed the machine learning API, I have also
found that the SVM model files it generates are much larger! (suspect not compatible?) and that the
classification results are slightly worse (!) at least for the linear case. Maybe in 3.x they changed the
underlying implementation?).

Moved to cmake and it seems to work ..

Can changed values of positives and negatives (-1,1). Classification still the same but also same poor and slow detection.

Need to either pass some parameters via command line (e.g. image size, type of SVM, etc,) or through cmake compiler defines
to make this more flexible.

I normally something like #ifdef ENABLEPAUSE which is then defined in the CMakeLists.txt and for documentation when I do not want that
defined I use something like ENABLEPAUSENOT

*****************************************************

This is the CMakeLists.txt, binaries are explained later here

cmake_minimum_required(VERSION 2.8)
project( Descriptors  )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( Descriptors DescriptorsMain.cpp tool.cpp createDescriptors.cpp )
add_executable (TrainLOO TrainLOO_Main.cpp tool.cpp)
add_executable (TrainGrid TrainGrid_Main.cpp tool.cpp)
add_executable (Detection DetectionMain.cpp tool.cpp LinearSVM.cpp BOSSHog.cpp)
add_executable (DetectionSingle DetectionSingleMain.cpp tool.cpp LinearSVM.cpp BOSSHog.cpp)
add_executable (Classify ClassifyMain.cpp tool.cpp LinearSVM.cpp)

target_link_libraries( Descriptors ${OpenCV_LIBS} )
target_link_libraries (TrainLOO ${OpenCV_LIBS})
target_link_libraries (TrainGrid ${OpenCV_LIBS})
target_link_libraries (Detection ${OpenCV_LIBS})
target_link_libraries (DetectionSingle ${OpenCV_LIBS})
target_link_libraries (Classify ${OpenCV_LIBS})

set_target_properties (TrainLOO PROPERTIES COMPILE_DEFINITIONS "ENABLEPAUSENOT;CHANGELABELSNOT;TRAIN_SVM_RBF")
set_target_properties (Classify PROPERTIES COMPILE_DEFINITIONS "ENABLEPAUSENOT;CHANGELABELSNOT")
set_target_properties (DetectionSingle PROPERTIES COMPILE_DEFINITIONS "ENABLEPAUSE")
set_target_properties (TrainGrid PROPERTIES COMPILE_DEFINITIONS "ENABLEPAUSE;TRAIN_SVM_RBF")

****************************************************
Video files are stored in Video/OriginalCam1 and they are called Cell_phone_Spanish.Cam1.avi, Checkout_French.Cam1.avi ...
Ground truths are stored in the same folder above and are called Cell_phone_Spanish.Cam1.xgtf, Cell_phone_Spanish.Cam1.xgft ...
It is that structrure of video names that will be followed for Leave One Out experiments

Using the above data, code in code0_doSamples extracts "samples" (either 128x256 or 64x128, widthxheight). These are stored in BossSamples
Neg:	Negative samples in folders with the same name as the video files
Pos:	Inter_020	Those with overlap less than 20%
	  sit		These are directories with samples in each class (also with the structure of video names)
	  sitting
	  stand
	Inter_050	Those with overlap less than 50%
	  sit, sittig, stand
	Inter_100	Those with overlap less than 100% (ie all positives)
	  sit, sitting, stand
	Inter_full	Here there are no distinctions of classes (sit, sitting, standing) i.e. they refer to just "person"
	  Inter_020_full
	  Inter_050_full
	  Inter_100_full
	Training_full	This has half of the person samples and is used for TrainGrid (i.e. to find optimal SVM values)
new64x128		Correspond to samples at 64x128 size (same size as Dalal and OpenCV)
	neg		Negative samples in the same structure as the videos
	Pos
	  full_100	Positives in the structure of the videos, just "person" and all the samples
	sit		Positives, 100% sit
	sitting		id.
	stand		id.
NegTraining		negative samples with 1/2 the data to be used with TrainGrid (to find optimal SVM values)
	

The order of execution is like this

./Descriptors -n negatives_path -p positives_path -d descriptors_path
    Generates text descriptor files (in descriptors_path) using negatives found in negatives_path and positives in positive_path
    negatives_path and positive_path are assumed to have the same structure and be divided into video names folders
    descriptor_path is the populated by descriptor files maintaining the video names structure (names *-DESC.dat)
    
./TrainGrid -d descriptor_path -s name_svm_file
    Will find optimal set of SVM parameters (uses function train_grid) using the data descriptors found in descriptor_path
    (typically that is half the set of descriptors for the whole dataset)
    The output XML model file is stored in name_svm_file
    
./TrainLOO -d descriptor_path -s svm_path
    Trains a set of SVM's according to the stucture found in descriptor_path. eg if the descriptor path contains data descriptor files
    d1-DESC.dat, d2-DESC.dat ... dN-DESC.dat the program generates files svm_path/d1-SVM.xml, svm_path/d2-SVM.xml, ... svm_path/DN-SVM.xml
    where d1-SVM.xml is the results of training leaving d1-DESC-dat out, d2-SVM.xml training leaving d2-DESC.dat out and so on
    Typically, the results found with TrainGrid above are programmed into TrainLOO
    TODO: pass SVM parameters in command line
    
./Classify -s svm_path -n negative_path -p positive_path -r results_file
    Remember that TrainLOO generated a set of SVM files each inheriting the name of the descriptors (left out) and therefore of the video
    files corresponding to those (neg and pos) descriptors. So, this program takes each of those SVM classifiers (models) in turn and 
    picks the negative and positive samples corresponding to that video (as those were "left out" in the training of that SVM) and works
    out true positives, true negatives, false positives, false negatives, precision and recall for each classifier. The results are placed on
    a comma separated file results_file which can then be open by a spreadsheet.
    
./Detection and ./DetectionSimple
    I have not had too much luck with these. The idea is to use each SVM model and then the original videos to detect the people in those videos.
    The results are not good!
    TODO: find out why the results are poor
    
