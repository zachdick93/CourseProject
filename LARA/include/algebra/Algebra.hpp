#ifndef ALGEBRA_H
#define ALGEBRA_H

#include <vector>
#include <cmath>

using std::vector;

/*
	This code is a quick filler to replace a linear algebra library used in the original project. Sources of this code below.
	    Source of the determinant function logic: https://www.tutorialspoint.com/cplusplus-program-to-compute-determinant-of-a-matrix
	    Source of the logic for the other functions: https://www.geeksforgeeks.org/adjoint-inverse-matrix/
*/

namespace algebra {
class Algebra {
public:
    static double determinant(const vector<vector<double>> &matrix, size_t n);
    static vector<vector<double>> inverse(const vector<vector<double>> &matrix, size_t n);
private:
    static vector<vector<double>> getCofactor(const vector<vector<double>> &matrix, size_t p, size_t q, size_t n);
    static vector<vector<double>> adjoint(const vector<vector<double>> &matrix, size_t n);
};
}

#endif