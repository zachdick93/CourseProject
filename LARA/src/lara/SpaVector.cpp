#include "../../include/lara/SpaVector.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>

using std::stoi;
using std::stod;
using std::abs;
using std::cout;
using std::endl;

namespace lara {
SpaVector::SpaVector(const vector<string> &container) {
    index.resize(container.size());
    value.resize(container.size());

    size_t pos;
    for (size_t i = 0; i < container.size(); i++) {
        pos = container[i].find(':');
        index[i] = 1 + stoi(container[i].substr(0, pos));
        value[i] = stod(container[i].substr(pos+1));
    }
}

double SpaVector::l1Norm() {
    double sum = 0;
    for (double v : value) {
        sum += abs(v);
    }
    
    return sum;
}

void SpaVector::normalize(double norm) {
    for (size_t i = 0; i < value.size(); i++) {
        value[i] /= norm;
    }
}

int SpaVector::getLength() {
    size_t i = index.size();
    return index[i-1];
}

double SpaVector::dotProduct(const vector<double> &weight) {
    double sum = weight[0];
    for (size_t i = 0; i < index.size(); i++) {
        sum += value[i] * weight[index[i]];
    }
    return sum;
}

double SpaVector::dotProduct(const vector<double> &weight, int offset) {
    double sum = weight[offset];
    for (size_t i = 0; i < index.size(); i++) {
        sum += value[i] * weight[offset + index[i]];
    }
    return sum;
}
}
