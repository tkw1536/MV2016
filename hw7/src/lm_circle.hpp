/*
 * lm_distortion_params.hpp
 *
 * Solving the Hahn1 benchmark from:
 *  http://www.itl.nist.gov/div898/strd/nls/data/hahn1.shtml
 *
 *  Created on: Feb 27, 2014
 *      Author: kaust
 */

#ifndef LM_DISTORTION_PARAMS_HPP_
#define LM_DISTORTION_PARAMS_HPP_

#include <iostream>
#include <fstream>

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

/** A generic functor which should be inherited from.
 *
 */
template <typename _Scalar, int NX=Eigen::Dynamic, int NY=Eigen::Dynamic>
struct DenseFunctor{
	typedef _Scalar Scalar;
	enum {
		InputsAtCompileTime = NX,
		ValuesAtCompileTime = NY
	};
	typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
	typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
	typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

	int m_inputs, m_values;

	DenseFunctor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
	DenseFunctor(int inputs, int values=ValuesAtCompileTime) : m_inputs(inputs), m_values(values) {}

	/** Number of parameters.
	 *
	 */
	int inputs() const { return m_inputs; }

	/** Number of observations.
	 *
	 */
	int values() const { return m_values; }

	//int operator()(const InputType &x, ValueType& fvec) { }
	// should be defined in derived classes

	//int df(const InputType &x, JacobianType& fjac) { }
	// should be defined in derived classes
};

const int CIRCLE_PARAMS= 3;

/**
 * Solving the Hahn1 benchmark from:
 *  http://www.itl.nist.gov/div898/strd/nls/data/hahn1.shtml
 */
class CircleFunctor : public DenseFunctor<double>{
public:
	CircleFunctor(std::istream& datastream);
	int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const;

protected:
	std::vector<double> _x; ///< predictor variable
	std::vector<double> _y; ///< response variable

};

#endif /* LM_DISTORTION_PARAMS_HPP_ */
