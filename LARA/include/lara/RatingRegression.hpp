#ifndef RATING_REGRESSION_H
#define RATING_REGRESSION_H

#include <vector>
#include <string>
#include "Vector4Review.hpp"

using std::vector;
using std::string;
namespace lara {
class RatingRegression {
private:

    vector<double> alpha; // cached for difference vector

    double getRatingObjGradient();
    double getAspectObjGradient(int k, const vector<double> &beta);

protected:
    vector<Vector4Review> collection;
    vector<double> diagBeta; // cached diagonal for beta inference
    vector<double> gBeta; // cached gradient for beta inference
    vector<double> beta; // long vector for the matrix of beta
    
    vector<double> diagAlpha; // cached diagonal for alpha inference
    vector<double> gAlpha; // cached gradient for alpha inference
    vector<double> alphaCache; // to map alpha into a simplex by logistic functions

    int alphaStep;
    double alphaTol;
    int betaStep;
    double betaTol;
    double lambda;
    int v, k;
    int trainSize, testSize;

    void evaluateAspect();
    virtual double init(size_t v);
    virtual double prediction(Vector4Review &vct);


public:
    static const bool SCORE_SQUARE = false; //rating will be map by s^2 or exp(s)
    static const bool BY_OVERALL = false; //train aspect rating predictor by overall rating

    RatingRegression(int aStep, double aTol, int bStep, double bTol, double lmda);
    int loadVectors(string filename);
    int loadVectors(string filename, int size);
    void estimateAspectModel(string filename);
    void savePrediction(string filename);
    virtual void saveModel(string filename);
};
}

#endif