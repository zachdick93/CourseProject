#ifndef SPA_VECTOR_H
#define SPA_VECTOR_H

#include <vector>
#include <string>

using std::vector;
using std::string;

namespace lara {
class SpaVector {
public:
    vector<int> index;
    vector<double> value;

    SpaVector() { ; }
    SpaVector(const vector<string> &container);
    double l1Norm();
    void normalize(double norm);
    int getLength();
    double dotProduct(const vector<double> &weight);
    double dotProduct(const vector<double> &weight, int offset);
};
}

#endif