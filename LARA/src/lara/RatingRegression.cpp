#include "../../include/lara/RatingRegression.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>
#include "../../include/utilities/Utilities.hpp"
#include "../../include/lara/SpaVector.hpp"
#include "../../include/optimization/LBFGS.hpp"

using optimizer::LBFGS;
using utilities::Utilities;
using std::ofstream;
using std::endl;
using std::cout;
using std::vector;
using std::setprecision;
using std::max;
using std::min;

namespace lara {
RatingRegression::RatingRegression(int aStep, double aTol, int bStep, double bTol, double lmda) {
    alphaStep = aStep;
    alphaTol = aTol;

    betaTol = bTol;
    betaStep = bStep;
    lambda = lmda;

    std::srand(0); // with fixed random seed in order to get the same train/test split
}

int RatingRegression::loadVectors(std::string filename) {
    return loadVectors(filename, -1);
}

int RatingRegression::loadVectors(std::string filename, int size) {
    std::ifstream ifs (filename, std::ifstream::in);
    trainSize = 0;
    testSize = 0;
    
    int pos, len = 0;
    bool isTrain;
    std::vector<double> aspectSize;
    std::string tmpTxt;
    vector<string> ratings;
    while(ifs.good()) {
        std::getline(ifs, tmpTxt);
        pos = tmpTxt.find_first_of("\t ", 0);
        double prob = ( (double) rand() / (double) RAND_MAX);
        isTrain = prob < 0.75;
        if (isTrain) {
            trainSize++;
        }
        else {
            testSize++;
        }
        string id = tmpTxt.substr(0, pos);
        ratings.resize(0);
        int lastPos = pos;
        while (pos != string::npos) {
            lastPos = pos;
            pos = tmpTxt.find_first_of("\t ", pos+1);
            string rating = tmpTxt.substr(lastPos, pos - lastPos);
            ratings.push_back(rating);
        }
        Vector4Review vct (id, ratings, isTrain);
        if (aspectSize.empty()) {
            aspectSize.resize(vct.getAspectSize());
            k = aspectSize.size();
        } 

        for (size_t i = 0; i < vct.getAspectSize() ; i++) {
            vector<string> features;
            std::getline(ifs, tmpTxt);
            pos = tmpTxt.find_first_of(" ", 0); // sometimes may need to be "\t"
            string feature = tmpTxt.substr(0, pos);
            features.push_back(feature);
            while (pos != string::npos) {
                lastPos = pos;
                pos = tmpTxt.find_first_of(" ", pos+1); // sometimes may need to be "\t"
                if (pos == string::npos) break;
                feature = tmpTxt.substr(lastPos, pos-lastPos);
                features.push_back(feature);
            }
            vct.setAspect(i, features);
            aspectSize[i] += vct.getAspectSize(i);
        }
        vct.normalize();

        collection.push_back(vct);
        len = max(len, vct.getLength()); // max index word

        if (size > 0 && collection.size() >= size) break;
    }
    double sum = Utilities::sum(aspectSize);

    std::cout << "[Info]Aspect length proportion:\n";
    for (double v: aspectSize) {
        std::cout << std::fixed << "\t" << std::setprecision(3) << v/sum;
    }
    std::cout << "\n[Info]Load " << trainSize << "/" << testSize 
                  << " instance from " << filename 
                  << " with feature size " << len << std::endl;;
    ifs.close();
    return len;
}

void RatingRegression::evaluateAspect() {
    double aMSE = 0, oMSE = 0, icorr = 0, acorr = 0, corr, diff;
    int i = -1;
	bool iError = false, aError = false;

    vector<vector<double>> pred(k), ans(k);
    for (Vector4Review vct : collection) {
        if (vct.forTrain) continue; // only evaluating in testing cases
        i++;
        diff = prediction(vct) - vct.ratings[0];
        oMSE += diff * diff;
        for (int j = 0; j < k; j++) {
            pred[j].push_back(vct.pred[j]);
            ans[j].push_back(vct.ratings[j+1]);
        }

        // 1. Aspect evaluation: to skip overall rating in ground-truth
        aMSE += Utilities::MSE(vct.pred, vct.ratings, 1);

        try {
            corr = Utilities::correlation(vct.pred, vct.ratings, 1);
            icorr += corr;
        } 
        catch (int) {
            iError = true;
        }
    }

    // 2. entity level evaluation
    for (size_t j = 0; j < k; j++) {
        try {
            corr = Utilities::correlation(pred[j], ans[j], 0);
            acorr += corr;
        }
        catch (int) {
            aError = true;
        }
    }

    // MSE for overall rating, MSE for aspect rating, item level correlation, aspect level correlation
    if (iError) {
        std::cout << 'x';
    }
    else {
        std::cout << 'o';
    }
    if (aError) {
        std::cout << 'x';
    }
    else {
        std::cout << 'o';
    }

    std::cout << std::fixed << std::setprecision(3) 
                << "    " << sqrt(oMSE / (double) testSize)
                << "    " << sqrt(aMSE / (double)testSize)
                << "    " << icorr / (double) testSize
                << "    " << acorr / (double)k;
}

double RatingRegression::init(size_t v_) {
    if (collection.empty()) {
        std::cerr << "RatingRegression::init[Error]Load training data first!";
        return -1;
    }
    Vector4Review vct = collection[0];
    v = v_;
    k = vct.aspectV.size();

    diagBeta.resize(k * (v+1));
    gBeta.resize(diagBeta.size());
    beta.resize(gBeta.size());

    diagAlpha.resize(k);
    gAlpha.resize(k);
    alpha.resize(k);
    alphaCache.resize(k);

    return 0;
}

double RatingRegression::getRatingObjGradient() {
    double f = 0, orating, sum = Utilities::expSum(alpha);

    for (size_t i = 0; i < k; i++) {
        alphaCache[i] = exp(alpha[i]) / sum;
        gAlpha[i] = lambda * alpha[i];
        f += lambda * alpha[i] * alpha[i];
    }

    for (Vector4Review vct : collection) {
        if (vct.forTrain == false) continue;

        orating = -vct.ratings[0];
        for (size_t i = 0; i < k; i++) {
            orating += alphaCache[i] *vct.ratings[i + 1];
        }
        f += orating * orating;
        for (size_t i = 0; i < k; i++) {
            for (size_t j = 0; j < k; j++) {
                if (j == i) {
                    gAlpha[i] += orating * vct.ratings[i + 1] * alphaCache[i] * (1 - alphaCache[i]);
                }
                else {
                    gAlpha[i] -= orating * vct.ratings[j + 1] * alphaCache[i] * alphaCache[j];
                }
            }
        }
    }
    return f / 2;
}

double RatingRegression::getAspectObjGradient(int inK, const vector<double> &inBeta) {
    double f = 0, s, diff, rd;

    for (size_t j = 0; j < v; j++) {
        gBeta[j] = lambda * inBeta[j];
        f += lambda * inBeta[j] * inBeta[j];
    }

    for (Vector4Review vct : collection) {
        if (vct.forTrain == false) continue;

        s = vct.dotProduct(inBeta, inK);
        rd = BY_OVERALL ? vct.ratings[0] : vct.ratings[inK + 1];

        if (RatingRegression::SCORE_SQUARE) {
            diff = 0.5 * s * s - rd;
        }
        else {
            s = exp(s);
            diff = s - rd;
        }
        f += diff * diff;
        diff *= s;

        SpaVector sVct = vct.aspectV[inK];
        gBeta[0] += diff;
        for (size_t j = 0; j < sVct.index.size(); j++) {
            gBeta[sVct.index[j]] += diff * sVct.value[j];
        }
    }

    return f / 2;
}

double RatingRegression::prediction(Vector4Review &vct) {
    vct.getAspectRating(beta, v + 1);
    double orating = 0;
    for (size_t i = 0; i < k; i++) {
        orating += alphaCache[i] * vct.pred[i];
    }
    return orating;
}

void RatingRegression::estimateAspectModel(string filename) {
    init(loadVectors(filename));

    double f = 0;
    int n = 1 + v, m = 5, icall = 0;
    vector<int> iflag = {0};
    vector<int> iprint = {-1, 0};
    vector<double> inBeta(n);

    // training phase
    try {
        // Step 1: estimate rating regression model for overall rating with ground-truth aspect rating
        std::fill(std::begin(alpha), std::end(alpha), 0.0);
        std::fill(std::begin(diagAlpha), std::end(diagAlpha), 0.0);
        do {
            f = getRatingObjGradient();
            LBFGS::lbfgs(k, m, alpha, f, gAlpha, false, diagAlpha, iprint, alphaTol, 1e-20, iflag);
        } while (iflag[0] != 0 && ++icall <= alphaStep);
        std::cout << "[Info]Model for overall rating converge to " 
                  << f << ", with the learnt weights:";
        f = Utilities::expSum(alpha);
        for (size_t i = 0; i < k; i++) {
            alphaCache[i] = exp(alpha[i]) / f;
            std::cout << std::fixed << std::setprecision(3) << "    " << alphaCache[i];
        } 
        std::cout << std::endl;

        // Step 2: estimate rating regression model for each aspect with ground-truth aspect rating
        for (size_t i = 0; i < k; i++) {
            icall = 0;
            iflag[0] = 0;

            Utilities::randomize(inBeta);
            std::fill(std::begin(diagBeta), std::end(diagBeta), 0.0);
            do {
                f = getAspectObjGradient(i, inBeta);
                LBFGS::lbfgs(n, m, inBeta, f, gBeta, false, diagBeta, iprint, betaTol, 1e-20, iflag);
            } while (iflag[0] != 0 && ++icall <= betaStep);
            std::cout << "[Info]Model for aspect_" << i
                      << " converge to " << f;
            std::copy(std::begin(inBeta), std::end(inBeta), std::begin(beta) + (i*n));
        }

        // testing phase
        std::cout << "[Info]oMSE    aMSE    aCorr    iCorr";
        evaluateAspect();
    } catch (int e) {
        std::cerr << "RatingRegression::estimateAspectModel[Error] " << e << std::endl;
    }
}

void RatingRegression::savePrediction(string filename) {
    ofstream writer;
    writer.open(filename);

    for (Vector4Review vct : collection) {
        writer << vct.id;

        // all the ground-truth ratings
        for (size_t i = 0; i < vct.ratings.size(); i++) {
            writer << std::fixed << "\t" << setprecision(3) << vct.ratings[i];
        }

        // predicted ratings
        vct.getAspectRating(beta, (1+v));
        writer << "\t";
        for (size_t i = 0; i < vct.pred.size(); i++) {
            writer << std::fixed << "\t" << setprecision(3) << vct.pred[i];
        }

        // inferred weights (not meaningful for baseline logistic regression)
        writer << "\t";
        for (size_t i = 0; i < vct.alpha.size(); i++) {
            writer << std::fixed << "\t" << setprecision(3) << vct.alpha[i];
        }
        writer << endl;
    }
    writer.close();
}

void RatingRegression::saveModel(string filename) {
    ofstream writer;
    writer.open(filename);

    // \mu for \hat\alpha
    for (size_t i = 0; i < k; i++) {
        writer << alpha[i] << "\t";
    }
    writer << endl;

    // \sigma for \hat\alpha (unknown for logistic regression)
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < k; j++) {
            if (i == j){
                writer << "1.0\t";
            }
            else {
                writer << "0.0\t";
            }
        }
        writer << endl;
    }

    // \beta
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < v; j++) {
            writer << beta[i * (v + 1) + j] << "\t";
        }
        writer << endl;
    }

    // \sigma (unknown for logistic regression)
    writer << "1.0";
    writer.close();
}

}
