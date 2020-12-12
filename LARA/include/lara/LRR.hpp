#ifndef LRR_H
#define LRR_H

#include "LRR_Model.hpp"
#include "RatingRegression.hpp"
#include <fstream>
#include <vector>
#include <memory>

using std::vector;
using std::ofstream;

namespace lara {
class LRR : public RatingRegression {
public: 
    static bool SIGMA;
    static double PI;
    static const int K = 7;

    // aspect will be determined by the input file for LRR
    LRR(int alphaStep, double alphaTol, int betaStep, double betaTol, double lambda);

    // if we want to load previous models
    LRR(int alphaStep, double alphaTol, int betaStep, double betaTol, double lambda, string modelfile);

    double MStep(bool updateSigma);
    void EMEst(string filename, int maxIter, double converge);
    void saveModel(string filename) override;

protected:
    std::unique_ptr<LRR_Model> model;
    vector<double> oldAlpha; // in case optimization for alpha failed
    
    // BufferWriter trace;
    std::ofstream trace;

    double init(size_t v) override;
    double prediction(Vector4Review &vct) override;
    double EStep(Vector4Review &vct);
    double getAlphaObjGradient(Vector4Review &vct);
    double inferAlpha(Vector4Review &vct);
    double getBetaPriorObj();
    double getDataLikelihood();
    double getAuxDataLikelihood();
    double getBetaObjGradient();
    double mlBeta();

private:
    void testAlphaVariance(bool updateSigma);
};
}
#endif