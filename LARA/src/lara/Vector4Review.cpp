#include "../../include/lara/Vector4Review.hpp"
#include "../../include/lara/LRR.hpp"
#include "../../include/lara/RatingRegression.hpp"
#include "../../include/utilities/Utilities.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

using std::cout;
using std::endl;
using std::min;
using std::max;
using std::stod;
using utilities::Utilities;
namespace lara {
Vector4Review::Vector4Review(string inputId, const vector<string> &inputRatings, bool isTrain) {
    id = inputId;
    forTrain = isTrain;
    ratings.resize(inputRatings.size());
    for (size_t i = 0; i < inputRatings.size(); i++) {
        ratings[i] = stod(inputRatings[i]);
    }
    aspectV.resize(ratings.size() - 1);
    alphaHat.resize(aspectV.size());
    alpha.resize(aspectV.size());
    pred.resize(aspectV.size());
    predCache.resize(aspectV.size());
}

size_t Vector4Review::getAspectSize() {
    if (aspectV.size() == 1) {
        return LRR::K;
    }
    else {
        return aspectV.size();
    }
}

void Vector4Review::setAspect(int i, const vector<string> &features) {
    aspectV[i] = SpaVector(features);
}

double Vector4Review::getDocLength() {
    double sum = 0;
    for (SpaVector vct : aspectV) {
        sum += vct.l1Norm();
    }
    return sum;
}

void Vector4Review::normalize() {
    double norm = getDocLength();
    double aSize;
    for (size_t i = 0; i < aspectV.size(); i++) {
        SpaVector vct = aspectV[i];

        aSize = vct.l1Norm();
        vct.normalize(aSize);

        // an estimate of aspect weight
        alphaHat[i] = ((double) rand() / (double) RAND_MAX) + log(aSize / norm);
    }

    norm = Utilities::expSum(alphaHat);
    
    for (size_t i = 0; i < aspectV.size(); i++) {
        alpha[i] = exp(alphaHat[i]) / norm;
    }
}

int Vector4Review::getLength() {
    int len = 0;
    for (SpaVector vct : aspectV) {
        len = max(len, vct.getLength());
    }
    return len;
}

double Vector4Review::getAspectSize(int k) {
    return aspectV[k].l1Norm();
}

void Vector4Review::getAspectRating(const vector<vector<double>> &beta) {
    for (size_t i = 0; i < aspectV.size(); i++) {
        predCache[i] = aspectV[i].dotProduct(beta[i]);
        if (RatingRegression::SCORE_SQUARE) {
            pred[i] = 0.5 * predCache[i] * predCache[i];
        }
        else {
            pred[i] = exp(predCache[i]);
        }
    }
}

void Vector4Review::getAspectRating(const vector<double> &beta, int v) {
    for (size_t i = 0; i < aspectV.size(); i++) {
        predCache[i] = aspectV[i].dotProduct(beta, v*i);
        if (RatingRegression::SCORE_SQUARE) {
            pred[i] = 0.5 * predCache[i] * predCache[i];
        }
        else {
            pred[i] = exp(predCache[i]);
        }
    }
}

double Vector4Review::dotProduct(const vector<double> &beta, int k) {
    return aspectV[k].dotProduct(beta);
}
}
