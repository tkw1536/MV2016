#include <string>
#include <stdio.h>
#include <math.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <opencv2/opencv.hpp>

#include "lm_circle.hpp"

using namespace std;
using namespace cv;

CircleFunctor::CircleFunctor(istream& datastream): DenseFunctor<double>(CIRCLE_PARAMS), _x(), _y(){
	string line;
	int i= 0;
	while (getline(datastream, line)){
		if (line.at(0)=='#') continue;
		std::istringstream li(line);
		double y(0), x(0);
		li >> x >> y;
		// cout<< "read "<< (i+1)<< ": "<< y<< ", "<< x<< endl;
		_x.push_back(x);
		_y.push_back(y);
		++i;
	}

	m_values= _x.size();
}


int CircleFunctor::operator()(const Eigen::VectorXd &b, Eigen::VectorXd &fvec) const{

	// extract the current parameters
	double x0 = b[0];
	double y0 = b[1];
	double aR = abs(b[2]);

	// iterate over all points
	for (int i= 0; i< m_values; ++i){
		double x= _x[i];
		double y= _y[i];

		// error = distance from circle boundary
		// sqrt((x - x0)^2 + (y - y0)^2) - |R|
		fvec(i) = sqrt(pow(x - x0, 2) + pow(y - y0, 2)) - aR; // don't square this
	}
	return 0;
}


void print_status(Eigen::LevenbergMarquardtSpace::Status r){
	switch(r){
	case Eigen::LevenbergMarquardtSpace::ImproperInputParameters:
		cout<< " ImproperInputParameters"<< endl;
		break;
	case Eigen::LevenbergMarquardtSpace::RelativeReductionTooSmall:
		cout<< " RelativeReductionTooSmall"<< endl;
		break;
	case Eigen::LevenbergMarquardtSpace::RelativeErrorTooSmall:
		cout<< " RelativeErrorTooSmall"<< endl;
		break;
	case Eigen::LevenbergMarquardtSpace::RelativeErrorAndReductionTooSmall:
		cout<< " RelativeErrorAndReductionTooSmall"<< endl;
		break;
	case Eigen::LevenbergMarquardtSpace::CosinusTooSmall:
		cout<< " CosinusTooSmall"<< endl;
		break;
	case Eigen::LevenbergMarquardtSpace::TooManyFunctionEvaluation:
		cout<< " TooManyFunctionEvaluation"<< endl;
		break;
	case Eigen::LevenbergMarquardtSpace::FtolTooSmall:
		cout<< " FtolTooSmall"<< endl;
		break;
	case Eigen::LevenbergMarquardtSpace::XtolTooSmall:
		cout<< " XtolTooSmall"<< endl;
		break;
	case Eigen::LevenbergMarquardtSpace::GtolTooSmall:
		cout<< " GtolTooSmall"<< endl;
		break;
	default:
		cout<< " Unknown status: "<< int(r)<<  endl;
		break;
	}
}

int main(int argc, char *argv[]){

	string file_name("cv_circle_pts.txt");
	ifstream data_file(file_name.c_str());
	if (!data_file){
		cerr<< "Data file "<< file_name<< " could not be opened!"<< endl;
		return -1;
	}

	cout.precision(6);
	cout <<  std::scientific<< std::setw(20);

	Eigen::VectorXd b(CIRCLE_PARAMS);

	// intial value
	// circle at the origin with radius 1
	b[0]= 0.0;
	b[1]= 0.0;
	b[2]= 1.0;

	std::cout << "Initial guess parameters b: \n" << b << std::endl;

	CircleFunctor functor(data_file);
	Eigen::NumericalDiff<CircleFunctor> numDiff(functor);
	Eigen::LevenbergMarquardt<Eigen::NumericalDiff<CircleFunctor>,double> lm(numDiff);

	lm.parameters.maxfev = 2000;
	lm.parameters.xtol = 1.0e-10;
	// lm.parameters.ftol

	std::cout << "Max function evaluations: "<< lm.parameters.maxfev << std::endl;

	Eigen::LevenbergMarquardtSpace::Status ret = lm.minimize(b);

	cout << "Return value: "; print_status(ret);
	cout << "Nr. of iterations: "<< lm.iter << endl;
	cout << "Nr. of function evaluations: "<< lm.nfev << endl;

	cout<< "Residual fval: "<< lm.fnorm<< endl;
	cout<< "Residual gval: "<< lm.gnorm<< endl;

	std::cout << "x that minimizes the function: \n" << b << endl;

	return 0;
}
