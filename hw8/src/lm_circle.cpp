#include <string>
#include <vector>
#include <math.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <random>
#include <unordered_set>
#include <limits>


#include "lm_circle.hpp"

using namespace std;
using namespace cv;

CircleFunctor::CircleFunctor(istream& datastream): DenseFunctor<double>(CIRCLE_PARAMS), _x(), _y(){

	// create empty vectors X and Y
	std::vector<double> X = std::vector<double>();
	std::vector<double> Y = std::vector<double>();


	// read in the file
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

	// and make the size properly
	m_values= _x.size();
}

CircleFunctor::CircleFunctor(std::vector<double>& X, std::vector<double>& Y): DenseFunctor<double>(CIRCLE_PARAMS), _x(), _y(){

	// if the size is not the same, throw an error.
	if(X.size() != Y.size()){
		throw 1;
	}

	// add all the elements
	for(std::vector<int>::size_type i = 0; i != X.size(); i++) {
		_x.push_back(X[i]);
		_y.push_back(Y[i]);
	}

	// and get the size
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

std::tuple<double, double, double> find_circle(CircleFunctor * functor) {

	// initial vector for parameters.
	// circle at the origin with radius 1
	Eigen::VectorXd b(CIRCLE_PARAMS);
	b[0]= 0.0;
	b[1]= 0.0;
	b[2]= 1.0;

	// prepare LevenbergMarquardt iterator
	Eigen::NumericalDiff<CircleFunctor> numDiff(*functor);
	Eigen::LevenbergMarquardt<Eigen::NumericalDiff<CircleFunctor>,double> lm(numDiff);

	lm.parameters.maxfev = 2000;
	lm.parameters.xtol = 1.0e-10;

	// run the minimisation
	lm.minimize(b);

	// and return the value
	return std::tuple<double, double, double>(b[0], b[1], b[2]);
}

std::tuple<std::vector<double>, std::vector<int>, double> RANSAC(CircleFunctor * functor, double tau, int N_min, int iter){

	// TODO: Finish this code

	// create a model
	std::vector<double> model_ = std::vector<double>();
	std::vector<int> I_ = std::vector<int>();
	double E_ = std::numeric_limits<double>::max();

	for(int i = 0; i < iter; i++){



	}

	return std::tuple<std::vector<double>, std::vector<int>, double>(model_, I_, E_);
}

// <shuffling>
// adapted from http://stackoverflow.com/a/28287865
std::vector<int> pick(int N, int k) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::unordered_set<int> vals = pickSet(N, k, gen);

    std::vector<int> result(vals.begin(), vals.end());
    std::shuffle(result.begin(), result.end(), gen);
    return result;
}

std::unordered_set<int> pickSet(int N, int k, std::mt19937& gen)
{
    std::unordered_set<int> elems;
    for (int r = N - k; r < N; ++r) {
        int v = std::uniform_int_distribution<>(1, r)(gen);

        if (!elems.insert(v).second) {
            elems.insert(r);
        }
    }
    return elems;
}

// </shuffling>


int main(int argc, char *argv[]){

	string file_name("pts_to_ransac.txt");
	ifstream data_file(file_name.c_str());
	if (!data_file){
		cerr<< "Data file "<< file_name<< " could not be opened!"<< endl;
		return -1;
	}

	cout.precision(6);
	cout <<  std::scientific<< std::setw(20);

	// create a functor
	CircleFunctor functor(data_file);

	// and run the minimisation
	// TODO: Make this use RANSAC
	std::tuple<double, double, double> optimal_params = find_circle(&functor);

	std::cout << "(x0, y0) = (" << std::get<0>(optimal_params) << "," << std::get<1>(optimal_params)  << ")" << endl;
	return 0;
}
