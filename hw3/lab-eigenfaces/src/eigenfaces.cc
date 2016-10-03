
#include <iostream>
#include <map>
#include <limits>
#include <regex>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <libltdl/lt_system.h>

using namespace cv;
using namespace cv::face;
using namespace std;
using namespace boost;
using namespace boost::filesystem;

#define NUM_EIGENVECTORS 30




bool verify_folder(path& p){
	if (!exists(p)){
		cerr<< "Folder "<< p.c_str()<< " does not exist"<< endl;
		return false;
	}
	if (!(is_directory(p))){
		cerr<< p.c_str()<< " is not a folder."<< endl;
		return false;
	}
	return true;
}

void show_images(std::string needle, std::string first, std::string second){

	// load the images
    Mat imageA = imread(needle, CV_LOAD_IMAGE_COLOR);
		Mat imageB = imread(first, CV_LOAD_IMAGE_COLOR);
		Mat imageC = imread(second, CV_LOAD_IMAGE_COLOR);

		// show image A
    namedWindow( "Needle", WINDOW_AUTOSIZE );
    imshow( "Needle", imageA );

		// show image B
		namedWindow( "First", WINDOW_AUTOSIZE );
    imshow( "First", imageB );

		namedWindow( "Second", WINDOW_AUTOSIZE );
    imshow( "Second", imageB );

		// wait for things
    waitKey(0);
}

void trainFaceRecognizer(path& train_path, Ptr<FaceRecognizer> recognizer) {
    vector<Mat> images;
    vector<int> labels;

	directory_iterator it(train_path);
	directory_iterator end_it;

    regex re("([0-9]{2})");

	for(;it != end_it; ++it){
        if(!is_regular_file(it->path()))
            continue;

		string fname = it->path().string();
        vector<string> tokens;

        cerr << "Reading " << fname << endl;

        // Get subject number by regex
        sregex_token_iterator
            begin_tokens(fname.begin(), fname.end(), re),
            end_tokens;
        copy(begin_tokens, end_tokens, back_inserter(tokens));

        // Read the actual image
        Mat img = imread(fname, IMREAD_GRAYSCALE);

        if(img.data == NULL) {
            cerr << "Error reading " << fname << endl;
            exit(EXIT_FAILURE);
        } else {
            labels.push_back(std::stoi(tokens.front()));
            images.push_back(img);
        }
	}

    // The line below actually does all the work
    recognizer->train(images, labels);
}

int main(int argc, const char* argv[]) {

	if (argc < 3){
		cerr<< "Usage: \n"<< argv[0]<< " [train dataset] [test image]"<< endl;
		return -1;
	}

	// Check that folders and files exist
	path train_path (argv[1]);
	if (!verify_folder(train_path))  return -1;

	path test_path (argv[2]);
	if (!exists(test_path))  return -1;


	Ptr<FaceRecognizer> facePtr = createEigenFaceRecognizer(NUM_EIGENVECTORS);

	// Load training samples
    trainFaceRecognizer(train_path, facePtr);

	cout << "*** loaded training samples ***" << endl;

    Mat testImage = imread(test_path.string(), IMREAD_GRAYSCALE);

    if(testImage.data == NULL) {
        cerr << "Error reading " << test_path.string() << endl;
        exit(EXIT_FAILURE);
    }

    int label;
    double confidence;

    facePtr->predict(testImage, label, confidence);

    cout << "Prediction for " << test_path << ": Subject " << label << "(" << confidence << ")" << endl;

	cout<< endl<< endl;
}
