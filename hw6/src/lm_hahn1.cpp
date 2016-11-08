#include <string>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <opencv2/opencv.hpp>

#include "lm_hahn1.hpp"

using namespace std;
using namespace cv;

Hahn1Functor::Hahn1Functor(istream& datastream): DenseFunctor<double>(HAHN1_PARAMS), _x(), _y(){
	string line;
	int i= 0;
	while (getline(datastream, line)){
		if (line.at(0)=='#') continue;
		std::istringstream li(line);
		double y(0), x(0);
		li >> y >> x;
		// cout<< "read "<< (i+1)<< ": "<< y<< ", "<< x<< endl;
		_x.push_back(x);
		_y.push_back(y);
		++i;
	}
	
	m_values= _x.size();
}


int Hahn1Functor::operator()(const Eigen::VectorXd &b, Eigen::VectorXd &fvec) const{
	//  y = (b1+b2*x+b3*x**2+b4*x**3) / (1+b5*x+b6*x**2+b7*x**3)  +  e
	for (int i= 0; i< m_values; ++i){
		double x= _x[i];
		double x2= x*x;
		double x3= x2*x;
		fvec(i)= _y[i] - (b[0]+b[1]*x+b[2]*x2+b[3]*x3)/(1+b[4]*x+b[5]*x2+b[6]*x3); // don't square it
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

	string file_name("Hahn1.dat");
	ifstream data_file(file_name.c_str());
	if (!data_file){
		cerr<< "Data file "<< file_name<< " could not be opened!"<< endl;
		return -1;
	}
	
	cout.precision(6);
	cout <<  std::scientific<< std::setw(20);
	
	Eigen::VectorXd b(HAHN1_PARAMS);
	// Initial values given in the dat file.
	b[0]= 10;
	b[1]= -1;	
	b[2]= 0.05;
	b[3]= -0.00001;
	b[4]= -0.05;
	b[5]= 0.001;
	b[6]= -0.000001;
	
	std::cout << "Initial guess for b: \n" << b << std::endl;

	Hahn1Functor functor(data_file);
	Eigen::NumericalDiff<Hahn1Functor> numDiff(functor);
	Eigen::LevenbergMarquardt<Eigen::NumericalDiff<Hahn1Functor>,double> lm(numDiff);

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
