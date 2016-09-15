
#include <iostream>
#include <limits>
#include <string>
#include <boost/filesystem.hpp>
#include "luv_color_histogram.hh"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
using namespace jir;
using namespace boost;
using namespace boost::filesystem;

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

void load(path& p, vector<LuvColorHistogram>& hist_vector, vector<std::string>& filenames){
	directory_iterator it(p);
	directory_iterator end_it;

	for(;it != end_it; ++it){
		// create a new histogram
		LuvColorHistogram * image = new LuvColorHistogram();
		image->load(it->path().string());

		// store it and the filename
		hist_vector.push_back(*image);
		filenames.push_back(it->path().string());
	}
}

void show_images(std::string fileA, std::string fileB){

	// load the images
    Mat imageA = imread(fileA, CV_LOAD_IMAGE_COLOR);
		Mat imageB = imread(fileB, CV_LOAD_IMAGE_COLOR);

		// show image A
    namedWindow( "Image A", WINDOW_AUTOSIZE );
    imshow( "Image A", imageA );

		// show image B
		namedWindow( "Image B", WINDOW_AUTOSIZE );
    imshow( "Image B", imageB );

		// wait for things
    waitKey(0);
}

void compare_hist_vectors(const vector<LuvColorHistogram>& h1, const vector<LuvColorHistogram>& h2, vector<std::string>& filenames){

	vector<LuvColorHistogram>::const_iterator it1= h1.begin();
	vector<std::string>::const_iterator fn1 = filenames.begin();

	for (; it1 != h1.end(); ++it1, ++fn1){
		vector<LuvColorHistogram>::const_iterator it2= h2.begin();
		vector<std::string>::const_iterator fn2 = filenames.begin();

		// metric for the distance between two images
		double metric(0);

		// minimum
		double min_metric(std::numeric_limits<double>::max());
		std::string min_fn;
		LuvColorHistogram min_value;

		for (; it2 != h2.end(); ++it2, ++fn2){

			// Compare metric between h1 and h2
			metric = it1->compare(*it2);

			// if the metric is smaller, we can store that.
			if(metric <= min_metric && it1 != it2){
				min_metric = metric;
				min_value = *it2;
				min_fn = *fn2;
			}
		}

		// Print filenames and show images
		cout << *fn1 << " has closest image " << min_fn << endl;
		show_images(*fn1, min_fn);
	}
}

int main(int argc, const char* argv[]) {

	if (argc < 2){
		cerr<< "Usage: \n"<< argv[0]<< " [folder name]"<< endl;
		return -1;
	}

	path p (argv[1]);
	if (!verify_folder(p)) return -1;


	vector<LuvColorHistogram> histograms;
	vector<std::string> filenames;

	load(p, histograms, filenames);
	cout<< "*** loaded "<< histograms.size()<< " samples."<< endl;

	cout<< "Comparing histograms"<< endl;
	compare_hist_vectors(histograms, histograms, filenames);

	cout<< endl<< endl;
}
