#ifndef ALGEBRA_H
#define ALGEBRA_H

#include <vector>
#include <cmath>

using std::vector;

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