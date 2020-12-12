#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>

using std::vector;

namespace utilities {
class Utilities {
public:
    static void randomize(vector<double> &v);
    static double expSum(const vector<double> &values);
    static double sum(const vector<double> &values);
    static double MSE(const vector<double> &pred, const vector<double> &answer, size_t offset);
    static double correlation(const vector<double> &pred, const vector<double> &answer, size_t offset);
};
}

#endif