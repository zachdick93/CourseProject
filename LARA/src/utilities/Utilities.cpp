#include <cmath>
#include <vector>
#include "../../include/utilities/Utilities.hpp"

using std::vector;

namespace utilities {
void Utilities::randomize(vector<double> &v) {
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = (2.0 * ((double) rand()/ (double) RAND_MAX) - 1.0)/10.0;
    }
}

double Utilities::expSum(const vector<double> &values) {
    double sum = 0; 
    for (double v : values) {
        sum += exp(v);
    }
    return sum;
}

double Utilities::sum(const vector<double> &values) {
    double sum = 0;
    for (double v : values) {
        sum += v;
    }
    return sum;
}

double Utilities::MSE(const vector<double> &pred, const vector<double> &answer, size_t offset) {
    double mse = 0;
    for (size_t i = 0; i < pred.size(); i++) {
        mse += (pred[i] - answer[i + offset]) * (pred[i] - answer[i + offset]);
    }
    return mse / pred.size();
}

double Utilities::correlation(const vector<double> &pred, const vector<double> &answer, size_t offset) {
    double mx = 0.0, my = 0.0, sx = 0.0, sy = 0.0;

    // first order moment
    for (size_t i = 0; i < pred.size(); i++) {
        mx += pred[i];
        my += answer[i+offset];
    }
    mx /= pred.size();
    my /= pred.size();

    // second order moment
    for (size_t i = 0; i < pred.size(); i++){
        sx += (pred[i] - mx) * (pred[i] - mx);
        sy += (answer[offset + i]- my) * (answer[offset + i]- my);
    }

    // handle special cases
    if (sx == 0 && sy == 0) {
        return 1;
    }
    else if (sx == 0 || sy == 0) {
        return 0;
    }

    sx = sqrt(sx / (pred.size() - 1));
    sy = sqrt(sy / (pred.size() - 1));

    // Pearson correlation
    double correlation = 0;
    for (size_t i = 0; i < pred.size(); i++) {
        correlation += (pred[i] - mx) / sx * (answer[i + offset] - my) / sy;
    }

    return correlation / (pred.size() - 1.0);
}
}