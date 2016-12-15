#include <stdio.h>
#include <iostream>

#define _USE_MATH_DEFINES
#include <cmath>

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"


using namespace cv;
using namespace std;

Mat rotZ(float q) {
	Mat R(3, 3, CV_64FC1, Scalar::all(0));
	R.at<double>(0, 0) = R.at<double>(1, 1) = cos(q);
	R.at<double>(2, 2) = 1.;
	R.at<double>(1, 0) = sin(q);
	R.at<double>(0, 1) = -R.at<double>(1, 0);
	return R;
}


void find_transform(Mat& F_left_right,
	std::vector<Point2f>& left_points, std::vector<Point2f>& right_points,
	Mat& mask) {
	cout << "Fundamental Matrix:" << endl;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			cout << F_left_right.at<double>(i, j) << ", ";
		}
		cout << endl;
	}
	cout << "Type: " << F_left_right.type() << endl;

	Mat A_left(3, 3, CV_64FC1, Scalar(0.)), A_right(3, 3, CV_64FC1, Scalar(0.));

	A_left.at<double>(0, 0) = 4.2836376527183329e+03; // fx
	A_left.at<double>(1, 1) = 4.3506039269027260e+03; // fy
	A_left.at<double>(2, 2) = 1.;
	A_left.at<double>(0, 2) = 1.6830912580592130e+03; // cx
	A_left.at<double>(1, 2) = 1.1552030153299565e+03; // cy

	A_right.at<double>(0, 0) = 4.2862740837050678e+03; // fx
	A_right.at<double>(1, 1) = 4.3517765025412155e+03; // fy
	A_right.at<double>(2, 2) = 1.;
	A_right.at<double>(0, 2) = 1.7955575629771263e+03; // cx
	A_right.at<double>(1, 2) = 1.1763845185382149e+03; // cy

	// Compute the Essential Matrix from the Fundamental Matrix
	Mat E_left_right = A_right.t()*F_left_right*A_left;

	cout << "Computed Essential Matrix." << endl;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			cout << E_left_right.at<double>(i, j) << ", ";
		}
		cout << endl;
	}
	cout << endl;

	Mat Sigma, U, Vt;
	SVD::compute(E_left_right, Sigma, U, Vt);

	Mat t1(3, 1, CV_64F, Scalar(0.)), t2(3, 1, CV_64F, Scalar(0));
	// We ignore Sigma and force it to be of the form diag(1,1,0).

	// fill-in: compute t1, t2

	cout << "Translation: First Solution:" << endl;
	cout << t1.at<double>(0, 0) << ", " << t1.at<double>(1, 0) << ", " << t1.at<double>(2, 0) << endl;
	cout << endl;

	Mat R1, R2;
	// fill-in: compute R1, R2

	cout << "Rotation: First Solution: " << endl;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			cout << R1.at<double>(i, j) << ", ";
		}
		cout << endl;
	}
	cout << endl<< endl;

	cout << "Translation: Second Solution:" << endl;
	cout << t2.at<double>(0, 0) << ", " << t2.at<double>(1, 0) << ", " << t2.at<double>(2, 0) << endl;
	cout << endl;

	cout << "Rotation: Second Solution: " << endl;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			cout << R2.at<double>(i, j) << ", ";
		}
		cout << endl;
	}

	// The third and the fourth solutions are just (R1, t2) and (R2, t1).

}

/** @function main */
int main(int argc, char** argv)
{
	// Call it with arguments:
	// imgs/jacobs1/Left-Right/DSCF3948-Left.jpg imgs/jacobs1/Left-Right/DSCF3948-Right.jpg

	if (argc != 3)
	{
		cerr<< "Two arguments (left and right images) expected."; return -1;
	}

	Mat img_left = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_right = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);


	if (!img_left.data || !img_right.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	//-- Step 1: Detect the keypoints using SIFT Detector
	SiftFeatureDetector detector;

	std::vector<KeyPoint> keypoints_left, keypoints_right;

	detector.detect(img_left, keypoints_left);
	cout << "Finished key-point detection for left-image." << endl;
	detector.detect(img_right, keypoints_right);
	cout << "Finished key-point detection for right-image." << endl;


	//-- Step 2: Calculate descriptors (feature vectors)
	SiftDescriptorExtractor extractor;

	Mat descriptors_left, descriptors_right;
	extractor.compute(img_left, keypoints_left, descriptors_left);
	cout << "Finished key-point descriptor extractions for left-image." << endl;
	extractor.compute(img_right, keypoints_right, descriptors_right);
	cout << "Finished key-point descriptor extractions for right-image." << endl;


	//-- Step 3: Matching descriptor vectors using FLANN or BFMatcher matcher
	FlannBasedMatcher matcher; // FlannBasedMatcher or BFMatcher
	std::vector< DMatch > matches;
	matcher.match(descriptors_left, descriptors_right, matches);

	cout << "Finished matching using FLANN." << endl;

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_left.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than a*min_dist )
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_left.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_matches; // The result will be drawn on this concatenated image


	//-- Localize the object
	std::vector<Point2f> left_points;
	std::vector<Point2f> right_points;

	for (unsigned int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		left_points.push_back(keypoints_left[good_matches[i].queryIdx].pt);
		right_points.push_back(keypoints_right[good_matches[i].trainIdx].pt);
	}

	Mat mask;

	// Since these points do not come from a planar object, a homography does not exist,
	// so the next level of filtering needs to be done by fitting a Fundamental-Matrix, and
	// embedding that fitting in RANSAC.
	// Mat H = findHomography(obj, scene, CV_RANSAC, 1, mask);

	Mat F_left_right= findFundamentalMat(left_points, right_points, CV_FM_RANSAC, 1, 0.95, mask);
	cout << "Finished RANSAC. Found Fundamental Matrix." << endl;
	find_transform(F_left_right, left_points, right_points, mask);

	drawMatches(img_left, keypoints_left, img_right, keypoints_right,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		/*vector<char>()*/ mask, DrawMatchesFlags::DRAW_RICH_KEYPOINTS); // DRAW_RICH_KEYPOINTS

	imwrite("image_matches.png", img_matches);
	cout << "Written output-image" << endl;
	imshow("Good Matches for Fundamental Matrix Estimation", img_matches);
	waitKey(0);

	return 0;
}
