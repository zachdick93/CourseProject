#include "../../include/lara/LRR.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <memory>
#include "../../include/utilities/Utilities.hpp"
#include "../../include/optimization/LBFGS.hpp"
#include "../../include/lara/RatingRegression.hpp"

using std::min;
using std::max;
using optimizer::LBFGS;
using utilities::Utilities;
using std::cout;
using std::endl;
using std::abs;

namespace lara {
bool LRR::SIGMA = false;
double LRR::PI = 0.5;

LRR::LRR(int aStep, double aTol, int bStep, double bTol, double lmda)
: RatingRegression(aStep, aTol, bStep, bTol, lmda) {
    model = nullptr;
    oldAlpha.resize(0);
    PI = 0.5;
}

LRR::LRR(int aStep, double aTol, int bStep, double bTol, double lmda, string modelfile)
: RatingRegression(aStep, aTol, bStep, bTol, lmda) {
    model = std::make_unique<LRR_Model>(modelfile);
    oldAlpha.resize(model->k);
    PI = 0.5;
}

double LRR::init(size_t v) {
    RatingRegression::init(v);
    double initV = 1; // likelihood for the first iteration won't matter
    // keep track of the model update trace
    try {
        trace.open("trace.dat");
        for (size_t i = 0; i < k; i++) {
            trace << "Aspect_" << i << "    ";
        }
        trace << "alpha    beta    data    aux_data    sigma\n";
    } catch (int e) {
        std::cerr << "LRR::init[Error] " << e << std::endl;
    }

    if (model == nullptr) {
        model = std::make_unique<LRR_Model>(k, v);
        oldAlpha.resize(model->k);

        PI = 2.0; // try to seeka  better initialization of beta
        initV = MStep(false); // this is just estimated alpha, no need to update sigma yet
        PI = 0.5;
    }

    return initV;
}

double LRR::prediction(Vector4Review &vct) {
    // Step 1: infer the aspect ratings/weights
    EStep(vct);

    // Step 2: calculate the overall rating
    double orating = 0;
    for (size_t i = 0; i < model->k; i++) {
        orating += vct.alpha[i] * vct.pred[i];
    }
    return orating;
}

double LRR::EStep(Vector4Review &vct) {
    // Step 1: estimate aspect rating
    vct.getAspectRating(model->beta);

    // Step 2: infer aspect weight
    try {
        std::copy(std::begin(vct.alpha), std::end(vct.alpha), std::begin(oldAlpha));
        return inferAlpha(vct);
    } catch (int e) {
        std::copy(std::begin(oldAlpha), std::end(oldAlpha), std::begin(vct.alpha));
        // failed with exceptions
        return -2;
    }
}

// we are estimating \hat{alpha}
double LRR::getAlphaObjGradient(Vector4Review &vct) {
    double expSum = Utilities::expSum(vct.alphaHat);
    double orating = -vct.ratings[0];
    double s;
    double sum = 0;

    // initialize the gradient
    std::fill(std::begin(gAlpha), std::end(gAlpha), 0.0);

    for (size_t i = 0; i < model->k; i++) {
        // map to aspect weight
        vct.alpha[i] = exp(vct.alphaHat[i]) / expSum; 

        // estimate the overall rating
        orating += vct.alpha[i] * vct.pred[i]; 

        //difference with prior
        alphaCache[i] = vct.alphaHat[i] - model->mu[i];

        s = PI * (vct.pred[i] - vct.ratings[0]) * (vct.pred[i] - vct.ratings[0]);

        if (abs(s) > 1e-10) {
            for (size_t j = 0; j < model->k; j++) {
                if (j == i) {
                    gAlpha[j] += 0.5 * s * vct.alpha[i] * (1 - vct.alpha[i]);
                }
                else {
                    gAlpha[j] -= 0.5 * s * vct.alpha[i] * vct.alpha[j];
                }
            }
            sum += vct.alpha[i] * s;
        }
    }

    double diff = orating / model->delta;
    for (size_t i = 0; i < model->k; i++) {
        s = 0;
        for (size_t j = 0; j < model->k; j++) {
            // part I of objective function: data likelihood
            if (i == j) {
                gAlpha[j] += diff * vct.pred[i] * vct.alpha[i] * (1 - vct.alpha[i]);
            }
            else {
                gAlpha[j] -= diff * vct.pred[i] * vct.alpha[i] * vct.alpha[j];
            }
            // part II of objective function: prior
            s += alphaCache[j] * model->sigmaInv[i][j];
        }

        gAlpha[i] += s;
        sum += alphaCache[i] * s;
    }

    return 0.5 * (orating * orating / model->delta + sum);
}

double LRR::inferAlpha(Vector4Review &vct) {
    double f = 0;
    vector<int> iprint = {-1, 0};
    vector<int> iflag = {0};
    int icall = 0;
    int n = model->k;
    int m = 5;

    // initialize the diagonal matrix
    std::fill(std::begin(diagAlpha), std::begin(diagAlpha), 0.0);
    do {
        f = getAlphaObjGradient(vct); // to be minimized
        LBFGS::lbfgs(n, m, vct.alphaHat, f, gAlpha, false, diagAlpha, iprint, alphaTol, 1e-20, iflag);
    } while (iflag[0] != 0 && ++icall <= alphaStep);

    if (iflag[0] != 0) {
        return -1;
    }
    else {
        double expsum = Utilities::expSum(vct.alphaHat);
        for (n = 0; n < model->k; n++) {
            vct.alpha[n] = exp(vct.alphaHat[n]) / expsum;
        }
        return f;
    }
}

void LRR::testAlphaVariance(bool updateSigma) {
    try {
        double v;

        // test the variance of \hat\alpha estimation
        std::fill(std::begin(diagAlpha), std::end(diagAlpha), 0.0);
        for (Vector4Review vct : collection) {
            if (vct.forTrain == false) continue;
            for (size_t i = 0; i < k; i++) {
                v = vct.alphaHat[i] - model->mu[i];
                diagAlpha[i] += v * v; // just for variance
            }
        }

        for (size_t i = 0; i < k; i++) {
            diagAlpha[i] /= trainSize;
            if (i == 0 && updateSigma) {
                trace << "*";
            }
            // mean and variance of \hat\alpha
            trace << std::fixed << std::setprecision(3) << model->mu[i] << ":" << diagAlpha[i] << "\t";
        }
    } catch (int i) {
        std::cerr << "LRR::testAlphaVariance[Error] " << i << std::endl;
    }
}

double LRR::MStep(bool updateSigma) {
    updateSigma = false; // shall we update sigma?
    int i, j, k = model->k;
    double v;

    // step 0: initialize the statistics
    std::fill(std::begin(gAlpha), std::end(gAlpha), 0.0);

    // step 1: ML for \mu
    for (Vector4Review vct : collection) {
        if (vct.forTrain == false) continue;

        for (size_t i = 0; i < k; i++) {
            gAlpha[i] += vct.alphaHat[i];
        }
    }
    for (size_t i = 0; i < k; i++) {
        model->mu[i] = gAlpha[i] / trainSize;
    }
    testAlphaVariance(updateSigma);

    // step 2: ML for \sigma
    if (updateSigma) {
        for (size_t i = 0; i < k; i++) {
            std::fill(std::begin(model->sigmaInv[i]), std::end(model->sigmaInv[i]), 0.0);
        }

        for (Vector4Review vct : collection) {
            if (vct.forTrain == false) {
                continue; 
            }

            for (size_t i = 0; i < k; i++) {
                diagAlpha[i] = vct.alphaHat[i] - model->mu[i];
            }

            if (SIGMA) {
                for (size_t i = 0; i < k; i++){
                    for (size_t j = 0; j < k; j++) {
                        model->sigmaInv[i][j] += diagAlpha[i] * diagAlpha[j];
                    }
                }
            }
            else {
                for (size_t i = 0; i < k; i++) {
                    model->sigmaInv[i][i] += diagAlpha[i] * diagAlpha[i];
                }
            }
        }

        for (size_t i = 0; i < k; i++) {
            if (SIGMA){
                model->sigmaInv[i][i] = (1.0 + model->sigmaInv[i][i]) / (1 + trainSize); // prior
                for(size_t j = 0; j < k; j++)
                    model->sigma[i][j] = model->sigmaInv[i][j];
            } else {
                v = (1.0 + model->sigmaInv[i][i]) / (1 + trainSize);
                model->sigma[i][i] = v;
                model->sigmaInv[i][i] = 1.0 / v;
            }
        }
        model->calcSigmaInv(1);
    }

    // calculate the likelihood for the alpha part 
    double alphaLikelihood = 0, betaLikelihood = 0;
    for (Vector4Review vct : collection) {
        if (vct.forTrain == false) continue;

        for (size_t i = 0; i < k; i++) {
            diagAlpha[i] = vct.alphaHat[i] - model->mu[i];
        }
        alphaLikelihood += model->calcCovariance(diagAlpha);
    }
    alphaLikelihood += log(model->calcDet());

    // step 3: ML for \beta
    try {
        mlBeta();
    } catch (int e) {
        std::cerr << "LRR::MStep[Error] " << -3 << std::endl;
    }

    betaLikelihood = getBetaPriorObj();

    // step 4: ML for \delta
    double dataLikelihood = getDataLikelihood();
    double auxData = getAuxDataLikelihood();
    double oldDelta = model->delta;
    model->delta = dataLikelihood / trainSize;
    dataLikelihood /= oldDelta;

    try {
        trace << std::fixed << std::setprecision(3) << alphaLikelihood << "    "
              << betaLikelihood << "\t" << dataLikelihood << "    "
              << auxData << "    " << log(model->delta) << endl;
    } catch (int e) {
        std::cerr << "LRR::MStep[Error] " << -5;
    }
    return alphaLikelihood + betaLikelihood + dataLikelihood + auxData + log(model->delta);
}

double LRR::getBetaPriorObj() {
    double likelihood = 0;
    for (size_t i = 0; i < model->beta.size(); i++) {
        for (size_t j = 0; j < model->beta[i].size(); j++) {
            likelihood += model->beta[i][j] * model->beta[i][j];
        }
    }
    return lambda * likelihood;
}

double LRR::getDataLikelihood() {
    double likelihood = 0, orating;

    // part I of objective function: data likelihood
    for (Vector4Review vct : collection) {
        if (vct.forTrain == false) continue;

        orating = -vct.ratings[0];

        // apply the current model
        vct.getAspectRating(model->beta);
        for (size_t i = 0; i < vct.alpha.size(); i++) {
            orating += vct.alpha[i] * vct.pred[i];
        }
        likelihood += orating * orating;
    }
    return likelihood;
}

double LRR::getAuxDataLikelihood() {
    double likelihood = 0, orating;

    for (Vector4Review vct : collection) {
        if (vct.forTrain == false) continue;

        orating = vct.ratings[0];
        for (size_t i = 0; i < vct.alpha.size(); i++) {
            likelihood += vct.alpha[i] * (vct.pred[i] - orating) * (vct.pred[i] - orating);
        }
    }
    return PI * likelihood;
}

double LRR::getBetaObjGradient() {
    double likelihood = 0, auxLikelihood = 0, orating, diff, oRate;
    size_t vSize = model->v + 1, offset;

    std::fill(std::begin(gBeta), std::end(gBeta), 0.0);

    for (Vector4Review vct : collection) {
        if (vct.forTrain == false) continue;

        oRate = vct.ratings[0];
        orating = -oRate;

        vct.getAspectRating(beta, vSize);
        for (size_t i = 0; i < model->k; i++) {
            orating += vct.alpha[i] * vct.pred[i];
        }

        likelihood += orating * orating;
        orating /= model->delta;

        offset = 0;
        for (size_t i = 0; i < model->k; i++) {
            auxLikelihood += vct.alpha[i] * (vct.pred[i] - oRate) * (vct.pred[i] - oRate);
            if (RatingRegression::SCORE_SQUARE)
                diff = vct.alpha[i] * (orating + PI * (vct.pred[i] - oRate)) * vct.predCache[i];
            else
                diff = vct.alpha[i] * (orating + PI * (vct.pred[i] - oRate)) * vct.pred[i];
            
            SpaVector sVct = vct.aspectV[i];
            gBeta[offset] += diff;
            for (size_t j = 0; j < sVct.index.size(); j++) {
                gBeta[offset + sVct.index[j]] += diff * sVct.value[j];
            }
            offset += vSize;
        }
    }

    double reg = 0;
    for (size_t i = 0; i < beta.size(); i++) {
        gBeta[i] += lambda * beta[i];
        reg += beta[i] * beta[i];
    }

    return 0.5 * (likelihood / model->delta + PI * auxLikelihood + lambda * reg);
}

double LRR::mlBeta() {
    double f = 0;
    vector<int> iprint = {1, 0};
    vector<int> iflag = {0};
    int icall = 0;
    int n = (1 + model->v) * model->k;
    int m = 5;
    
    for (size_t i = 0; i < model->k; i++) {
        std::copy(std::begin(model->beta[i]), (std::begin(model->beta[i]) + (model->v + 1)), (std::begin(beta) + (i * (model->v + 1))));
    }
    std::fill(std::begin(diagBeta), std::end(diagBeta), 0.0);
    do {
        if ((icall % 1000) == 0)
            std::cout << ".";
        f = getBetaObjGradient();
        LBFGS::lbfgs(n, m, beta, f, gBeta, false, diagBeta, iprint, betaTol, 1e-20, iflag);
    } while (iflag[0] != 0 && ++icall <= betaStep);
    std::cout << icall + "    ";
    for (size_t i = 0; i < model->k; i++) {
        std::copy(std::begin(beta) + (i * (model->v + 1)), std::begin(beta) + ((i + 1) * (model->v + 1) ), std::begin(model->beta[i]));
    }
    return f;
}

void LRR::EMEst(string filename, int maxIter, double converge) {
    int iter = 0, alphaExp = 0, alphaCov = 0;
    double tag, diff = 10.0, likelihood = 0, oldLikelihood = init(loadVectors(filename));

    std::cout << "[Info]Step    oMSE    aMSE    aCorr    iCorr    cov(a)    exp(a)    obj    converge\n";
    while (iter < min(8, maxIter) || (iter < maxIter && diff > converge)) {
        alphaExp = 0;
        alphaCov = 0;

        // E-step
        for (Vector4Review vct : collection) {
            if (vct.forTrain) {
                tag = EStep(vct);
                if (tag == -1) { // failed to converge
                    alphaCov++;
                }
                else if (tag == -2) { // failed with exception
                    alphaExp++;
                }
            }
        }
        std::cout << iter << "    ";

        // M-step
        likelihood = MStep((iter % 4) == 3);

        evaluateAspect();
        diff = (oldLikelihood - likelihood) / oldLikelihood;
        oldLikelihood = likelihood;
        std::cout << std::fixed << "    " << alphaCov << "    " << alphaExp << "    " 
                  << std::setprecision(3) << likelihood << "    " << diff << endl;
        iter++;
    }

    try {
        trace.close();
    } catch (int e) {
        std::cerr << "LRR::EMEst[Error] " << e << std::endl;
    }
}

void LRR::saveModel(string filename) {\
    model->saveToFile(filename);
}

} 