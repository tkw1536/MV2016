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
 *  Created on: Mar 25, 2013
 *      Author: kaustubh
 */

#ifndef __LUV_COLOR_HISTOGRAM_HH__
#define __LUV_COLOR_HISTOGRAM_HH__

#include <vector>
#include <ostream>
#include <string>
//#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace jir {

/** A histogram based on binned L*u*v* space.
 */
class LuvColorHistogram {
	public:

		/**
		 * @param nr_bins_l L range is [0, 100]
		 * @param nr_bins_u u range is [-134, 220]
		 * @param nr_bins_v v range is [-140, 122]
		 */
		LuvColorHistogram(
				unsigned int nr_bins_l= 10,
				unsigned int nr_bins_u= 35,
				unsigned int nr_bins_v= 26);

		/**
		 * The normalization constant is nr_samples().
		 */
		void normalize(void);

		/** loads an image and computes its (normalized) histogram.
		 * Normalization is only done if accumulate= false
		 * @param image_path_name fully qualified path.
		 * @param accumulate if true, the current histogram will not be reset but updated.
		 * You should not load with accumulate=true after normalizing.
		 * @return if successful.
		 */
		bool load(const std::string& image_path_name, bool accumulate= false);

		/**
		 * Loads from the image using the mask. Normalization is only done if accumulate= false.
		 * @param image of encoding CV_8UC3
		 * @param mask
		 * @param accumulate
		 * @return
		 */
		bool load(const cv::Mat& image, const cv::Mat& mask, bool accumulate= false);

		/**
		 * This is the normalization constant.
		 */
		int nr_samples(void){
			return _nr_samples;
		}

		/** Compare two histograms by using the Bhattacharyya distance.
		 * Note: the object other should have been created using the same constructor parameter.
		 */
		double compare(const LuvColorHistogram& other) const;

		/**
		 * @return the histogram is a 3D array of size nr_bins_l*nr_bins_u*nr_bins_v.
		 * In OpenCV MatND is typedefed to just Mat.
		 * Its elements can be accessed as hist(l_bin,u_bin,v_bin).
		 */
		const cv::MatND& get_histogram(void) const{
			return _hist;
		}

		int get_nr_bins_l(void) const{
			return _histSize[0];
		}

		int get_nr_bins_u(void) const{
			return _histSize[1];
		}

		int get_nr_bins_v(void) const{
			return _histSize[2];
		}

		int get_nr_of_bins(void) const{
			return (_histSize[0]*_histSize[1]*_histSize[2]);
		}

		/**
		 * @param v will be cleared and filled with the histogram values in the nested order L(u(v))
		 */
		// void get_histogram_as_vector(Eigen::VectorXf& v) const;

		/**
		 * The opposite of get_histogram_as_vector
		 */
		//void set_histogram_from_vector(const Eigen::VectorXf& v,
		//		unsigned int nr_bins_l, unsigned int nr_bins_u, unsigned int nr_bins_v);

	protected:
		int _histSize[3];
		float _l_ranges[2];
		float _u_ranges[2];
		float _v_ranges[2];
		cv::MatND _hist; ///< This will be normalized (sum=1) in the load function.
		int _nr_samples;
		cv::Mat _color_luv; // cached from the last computation
};
}
#endif
