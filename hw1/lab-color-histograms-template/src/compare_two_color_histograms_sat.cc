
#include <iostream>
#include <limits>
#include <string>
#include <boost/filesystem.hpp>
#include "sat_color_histogram.hh"
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

void load(path& p, vector<SaturatedColorHistogram>& hist_vector, vector<std::string>& filenames){
	directory_iterator it(p);
	directory_iterator end_it;

	for(;it != end_it; ++it){
		// create a new histogram
		SaturatedColorHistogram * image = new SaturatedColorHistogram();
		image->load(it->path().string());

		// store it and the filename
		hist_vector.push_back(*image);
		filenames.push_back(it->path().string());
	}
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

void find_two_closest(const SaturatedColorHistogram& needle, const vector<SaturatedColorHistogram>& haystack, const vector<std::string>& hayfiles, SaturatedColorHistogram& first, std::string& first_file, SaturatedColorHistogram& second, std::string& second_file) {

	// iterators which search
	vector<SaturatedColorHistogram>::const_iterator searcher = haystack.begin();
	vector<std::string>::const_iterator files = hayfiles.begin();

	// the maximal distance
	const double distance_max(std::numeric_limits<double>::max());

	// distance to first or second.
	double metric;
	double distance_first = distance_max;
	double distance_second = distance_max;

	// crawl through the haystack
	for (; searcher != haystack.end(); ++searcher, ++files){

		// current distance
		metric = needle.compare(*searcher);

		// if we are best than the best distance, shift everything back
		if(metric < distance_first){
			// shift 1 -> 2
			distance_second = distance_first;
			second = first;
			second_file = first_file;

			// and make a new 1
			distance_first = metric;
			first = *searcher;
			first_file = *files;

		// if it is still less than the second
		} else if (metric < distance_second){
			// make a new 2
			distance_second = metric;
			second = *searcher;
			second_file = *files;
		}
	}
}

void find_best_grams(const vector<SaturatedColorHistogram>& train_grams, vector<std::string>& train_files, const vector<SaturatedColorHistogram>& test_grams, const vector<std::string>& test_files){

	// iterate through the haystack
	vector<SaturatedColorHistogram>::const_iterator searcher = test_grams.begin();
	vector<std::string>::const_iterator files = test_files.begin();

	for (; searcher != test_grams.end(); ++searcher, ++files){

		// the current needle we find the best testing images for
		SaturatedColorHistogram needle = *searcher;

		// all the output
		SaturatedColorHistogram first;
		std::string first_file;
		SaturatedColorHistogram second;
		std::string second_file;

		find_two_closest(needle, train_grams, train_files, first, first_file, second, second_file);

		cout << "Image <" << (*files) << "> followed by <" << first_file << "> and <" << second_file << ">" << endl;

		// SHOW IMAGES woo
		show_images(*files, first_file, second_file);
	}
}



int main(int argc, const char* argv[]) {

	if (argc < 3){
		cerr<< "Usage: \n"<< argv[0]<< " [train] [test]"<< endl;
		return -1;
	}

	// load train path
	path train_path (argv[1]);

	if (!verify_folder(train_path))  return -1;

	// load test path
	path test_path (argv[2]);

	if (!verify_folder(test_path))  return -1;


	// Load training samples
	vector<SaturatedColorHistogram> train_grams;
	vector<std::string> train_files;
	load(train_path, train_grams, train_files);

	cout<< "*** loaded "<< train_grams.size()<< " training samples."<< endl;

	// Load testing samples
	vector<SaturatedColorHistogram> test_grams;
	vector<std::string> test_files;
	load(test_path, test_grams, test_files);

	cout<< "*** loaded "<< train_grams.size()<< " testing samples."<< endl;

	find_best_grams(train_grams, train_files, test_grams, test_files);
	// compare_hist_vectors(histograms, histograms, filenames);

	cout<< endl<< endl;
}
