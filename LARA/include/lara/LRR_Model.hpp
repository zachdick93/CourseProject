#ifndef LRR_MODEL_H
#define LRR_MODEL_H

#include <vector>
#include <string>

using std::vector;
using std::string;

namespace lara {
class LRR_Model {
public:
    size_t k; // # of aspects
    size_t v; // # of words
    vector<double> mu; // prior for \alpha in each review
    vector<vector<double>> sigmaInv; // precision matrix (NOT covariance!)
    vector<vector<double>> sigma; // only used for calculating inverse (\Sigma)
    vector<vector<double>> beta; // word sentiment polarity matrix should have one bias term!
    double delta; // variance of overall rating prediction (\sigma in the manual)

    LRR_Model(size_t k, size_t v);
    LRR_Model(string filename);
    double calcCovariance(const vector<double> &vct);
    double calcDet();
    void calcSigmaInv(double scale);
    void saveToFile(string filename);

private:
    void create();
    void init();
    void loadFromFile(string filename);
};
}

#endif