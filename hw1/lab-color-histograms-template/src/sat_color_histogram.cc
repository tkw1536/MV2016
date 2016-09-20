/*
* Copyright (c) 2012 Jacobs University Robotics Group
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without modification,
* are strictly limited to non-commercial academic purposes and are only permitted
* provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of Jacobs University Bremen nor the name jacobs_robotics nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* If you are interested in using this code for a commercial or non-academic purpose,
* please contact us.
*
* THIS SOFTWARE IS PROVIDED BY Jacobs Robotics ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL Jacobs Robotics BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Contact: Andreas Birk, a.birk@jacobs-university.de
*	   Kaustubh Pathak, k.pathak@jacobs-university.de
*
* Paper Mail:
*
*   Jacobs University Bremen gGmbH
*   Andreas Birk
*   Campus Ring 1
*   D-28759 Bremen
*   Germany
*/
/*
 * color_histogram_attribute_extractor.cc
 *
 *  Created on: Mar 25, 2012
 *      Author: francisc, kaustubh
 */

#include <iostream>
#include <cmath>
#include <algorithm>
//#include "color.hh"
#include "sat_color_histogram.hh"

using namespace jir;
using namespace boost;
using namespace std;
using namespace cv;

SaturatedColorHistogram::SaturatedColorHistogram(unsigned int nr_bins): _nr_samples(0){
	_histSize[0] = int(nr_bins);

	_ranges[0] = 0.0;
	_ranges[1] = 255.0 + 0.01;
}

void SaturatedColorHistogram::normalize(void){
	if (_nr_samples == 0) return;

	/* TODO
	for (int i_l= 0; i_l < get_nr_bins_l(); ++i_l){
		for (int i_u= 0; i_u < get_nr_bins_u(); ++i_u){
			for (int i_v= 0; i_v < get_nr_bins_v(); ++i_v){
				_hist.at<float>(i_l, i_u, i_v) /= float(_nr_samples);
			}
		}
	}*/
}

bool SaturatedColorHistogram::load(const Mat& color_img, const Mat& mask, bool accumulate){
	if (!accumulate){
		_nr_samples= 0;
	}

	_nr_samples += color_img.total();

	Mat color_img_float(color_img.size(), CV_32FC3);
	color_img.convertTo(color_img_float, CV_32F, 1.0/255.0); // Each channel BGR is between 0.0 and 1.0 now

	_color_hsv_.create(color_img_float.size(), CV_32FC3); // The destination should be preallocated.

	// Read the documentation of cvtColor:
	// http://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#cvtcolor
	cvtColor(color_img_float, _color_hsv_, CV_BGR2HSV);

	int channels[] = {0};
	const float* ranges[]= {_ranges};

	// Read the documentation at:
    // http://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html
	calcHist( &_color_hsv_, 1, channels,  mask /* Mat() if no mask */, _hist, 1, _histSize, ranges, true, accumulate);

	// TODO: Add bins for outside of these ranges

	// cv::normalize() does not work for dim > 3.
	// cv::normalize( _hist, _normailzed_hist, 0, 1, cv::NORM_MINMAX, -1, Mat());
	// Look at: http://opencv.willowgarage.com/documentation/cpp/basic_structures.html
	// void normalizeColorHist(MatND& hist)
	// cout<< "\t normalized color histogram"<< endl;
	// For CV_COMP_BHATTACHARYYA, normalization is not needed.

	// Explicitly normalize, so that the hist bins add to unity.
	if (!accumulate){
		normalize();
	}

	return true;
}

bool SaturatedColorHistogram::load(const std::string& image_path_name, bool accumulate){
	Mat color_img= imread(image_path_name.c_str());
	if(!color_img.data){
		std::cerr<< "could not open "<< image_path_name<< endl;
		return false;
	}

	return load(color_img, Mat(), accumulate);
}

double SaturatedColorHistogram::compare(const SaturatedColorHistogram& other) const{
	double result= 0.0;
	result= cv::compareHist(_hist, other._hist, CV_COMP_BHATTACHARYYA);
	return result;
}
