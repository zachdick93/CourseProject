
#ifndef VECTOR_4_REVIEW_H
#define VECTOR_4_REVIEW_H

#include <vector>
#include <string>
#include "SpaVector.hpp"

using std::vector;
using std::string;
namespace lara {
class Vector4Review {
public:
    string id;
    bool forTrain;
    vector<SpaVector> aspectV;
    vector<double> ratings;

    vector<double> pred;
    vector<double> predCache;
    vector<double> alpha;
    vector<double> alphaHat;

    Vector4Review(string id, const vector<string> &ratings, bool isTrain);
    size_t getAspectSize();
    void setAspect(int i, const vector<string> &features);
    double getDocLength();
    void normalize();
    int getLength();
    double getAspectSize(int k);
    void getAspectRating(const vector<vector<double>> &beta);
    void getAspectRating(const vector<double> &beta, int v);
    double dotProduct(const vector<double> &beta, int k);
};
}

#endif
