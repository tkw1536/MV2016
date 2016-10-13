// Lab-5 template file provided by Kaustubh Pathak
// Add your code in all places where it says "Fill-in".

#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace std;
using namespace cv;

int main()
{
	// ========================================================
	// START OF SETTING POINTS
	// ========================================================

	// SETUP the destination points.
	// These are sort of regular
	std::vector<Point2f> dstPoints;

	// width and height of a square are constant.
	// so we use these variables here.
	// It's really both 60 (see assignment)
	const int SQ_WIDTH = 60;
	const int SQ_HEIGHT = 60;

	// lower side
	dstPoints.push_back(Point2f(1*SQ_WIDTH, 0*SQ_HEIGHT)); // A
	dstPoints.push_back(Point2f(2*SQ_WIDTH, 0*SQ_HEIGHT)); // B
	dstPoints.push_back(Point2f(3*SQ_WIDTH, 0*SQ_HEIGHT)); // C
	dstPoints.push_back(Point2f(4*SQ_WIDTH, 0*SQ_HEIGHT)); // D
	dstPoints.push_back(Point2f(5*SQ_WIDTH, 0*SQ_HEIGHT)); // E

	dstPoints.push_back(Point2f(6*SQ_WIDTH, 0*SQ_HEIGHT)); // F

	// right side
	dstPoints.push_back(Point2f(6*SQ_WIDTH, 1*SQ_HEIGHT)); // G
	dstPoints.push_back(Point2f(6*SQ_WIDTH, 2*SQ_HEIGHT)); // H
	dstPoints.push_back(Point2f(6*SQ_WIDTH, 3*SQ_HEIGHT)); // I
	dstPoints.push_back(Point2f(6*SQ_WIDTH, 4*SQ_HEIGHT)); // J

	dstPoints.push_back(Point2f(6*SQ_WIDTH, 5*SQ_HEIGHT)); // K

	// upper side
	dstPoints.push_back(Point2f(5*SQ_WIDTH, 5*SQ_HEIGHT)); // L
	dstPoints.push_back(Point2f(4*SQ_WIDTH, 5*SQ_HEIGHT)); // M
	dstPoints.push_back(Point2f(3*SQ_WIDTH, 5*SQ_HEIGHT)); // N
	dstPoints.push_back(Point2f(2*SQ_WIDTH, 5*SQ_HEIGHT)); // O
	dstPoints.push_back(Point2f(1*SQ_WIDTH, 5*SQ_HEIGHT)); // P

	dstPoints.push_back(Point2f(0*SQ_WIDTH, 5*SQ_HEIGHT)); // Q

	// left side
	dstPoints.push_back(Point2f(0*SQ_WIDTH, 4*SQ_HEIGHT)); // R
	dstPoints.push_back(Point2f(0*SQ_WIDTH, 3*SQ_HEIGHT)); // S
	dstPoints.push_back(Point2f(0*SQ_WIDTH, 2*SQ_HEIGHT)); // T
	dstPoints.push_back(Point2f(0*SQ_WIDTH, 1*SQ_HEIGHT)); // U

	dstPoints.push_back(Point2f(0*SQ_WIDTH, 0*SQ_HEIGHT)); // V

	// inside points
	dstPoints.push_back(Point2f(1*SQ_WIDTH, 1*SQ_HEIGHT)); // W
	dstPoints.push_back(Point2f(5*SQ_WIDTH, 1*SQ_HEIGHT)); // X
	dstPoints.push_back(Point2f(5*SQ_WIDTH, 4*SQ_HEIGHT)); // Y
	dstPoints.push_back(Point2f(1*SQ_WIDTH, 4*SQ_HEIGHT)); // Z

	// SETUP the source points.
	// These were taken by looking at the image.
	std::vector<Point2f> srcPoints;

	// lower side
	srcPoints.push_back(Point2f(158, 358)); // A
	srcPoints.push_back(Point2f(248, 364)); // B
	srcPoints.push_back(Point2f(336, 370)); // C
	srcPoints.push_back(Point2f(426, 376)); // D
	srcPoints.push_back(Point2f(516, 382)); // E

	srcPoints.push_back(Point2f(609, 388)); // F

	// right side
	srcPoints.push_back(Point2f(606, 318)); // G
	srcPoints.push_back(Point2f(598, 256)); // H
	srcPoints.push_back(Point2f(596, 202)); // I
	srcPoints.push_back(Point2f(592, 194)); // J

	srcPoints.push_back(Point2f(590, 112)); // K

	// upper side
	srcPoints.push_back(Point2f(522, 108)); // L
	srcPoints.push_back(Point2f(453, 106)); // M
	srcPoints.push_back(Point2f(386, 102)); // N
	srcPoints.push_back(Point2f(324, 99)); // O
	srcPoints.push_back(Point2f(256, 96)); // P

	srcPoints.push_back(Point2f(196, 92)); // Q

	// left side
	srcPoints.push_back(Point2f(172, 134)); // R
	srcPoints.push_back(Point2f(153, 177)); // S
	srcPoints.push_back(Point2f(128, 230)); // T
	srcPoints.push_back(Point2f(99, 286)); // U

	srcPoints.push_back(Point2f(69, 352)); // V

	// inside points
	srcPoints.push_back(Point2f(183, 291)); // W
	srcPoints.push_back(Point2f(518, 312)); // X
	srcPoints.push_back(Point2f(520, 152)); // Y
	srcPoints.push_back(Point2f(243,  136)); // Z

	// ========================================================
	// END OF SETTING POINTS
	// ========================================================

	// Use the OpenCV findHomography to compute the homography matrix H.
	Mat H;
	H = findHomography( srcPoints, dstPoints, 0 );


	cout << "The computed homography matrix size is: " << endl;
	cout << "rows: "<< H.rows << ", cols= "<< H.cols << endl;
	for (int i = 0; i < H.rows; ++i) {
		for (int j = 0; j < H.cols; ++j) {
			cout << H.at<float>(i, j) << ",\t";
		}
		cout << endl;
	}

	Mat Img = imread("donald_annotated_points.png" , 1);
	Mat Out_Img;

	// Use the warpPerspective method of OpenCV to convert the input image Img to
	// the corrected image Out_Img of size 300x300 using H.
	warpPerspective(Img, Out_Img, H, *(new Size(300, 300)));

	imshow("Input", Img);
	imshow("Output", Out_Img);
	imwrite("donald_annotated_points_out.png", Out_Img);
	waitKey(0);

}
