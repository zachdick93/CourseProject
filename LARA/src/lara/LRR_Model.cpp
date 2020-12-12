#include "../../include/lara/LRR_Model.hpp"
#include "../../include/utilities/Utilities.hpp"
#include "../../include/algebra/Algebra.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

using std::cout;
using std::endl;
using std::getline;
using std::ofstream;
using std::ifstream;
using algebra::Algebra;
using utilities::Utilities;

namespace lara {
LRR_Model::LRR_Model(size_t k_, size_t v_) {
    k = k_;
    v = v_;
    init();
}

LRR_Model::LRR_Model(string filename) { 
    loadFromFile(filename);
}

void LRR_Model::create(){
    mu.resize(k);
    for (size_t i = 0; i < k; i++) {
        vector<double> temp(v + 1, 0);
        beta.push_back(temp);
        temp.resize(k);
        sigmaInv.push_back(temp);
        sigma.push_back(temp);
    }
}

void LRR_Model::init() {
    create();
    srand(0);
    for (size_t i = 0; i < k; i++) {
        mu[i] = (2.0 * ((double) rand() / (double) RAND_MAX) - 1);
        sigmaInv[i][i] = 1.0;
        sigma[i][i] = 1.0;
        Utilities::randomize(beta[i]);
    }
    delta = 1.0;
}
//
double LRR_Model::calcCovariance(const vector<double> &vct) {
    double sum = 0, s;
    for (size_t i = 0; i < k; i++) {
        s = 0;
        for (size_t j = 0; j < k; j++) {
            s += vct[j] * sigmaInv[j][i];
        }
        sum += s * vct[i];
    }
    
    return sum;
}

double LRR_Model::calcDet() {
    return Algebra::determinant(sigma, sigma.size());
}

void LRR_Model::calcSigmaInv(double scale) {
    auto inv = Algebra::inverse(sigma, sigma.size());
    cout << "inverse size: " << inv.size() << endl;
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < k; j++) {
            sigmaInv[i][j] = scale * inv[i][j];
        }
    }
}

void LRR_Model::saveToFile(string filename) {
    ofstream writer;
    writer.open(filename);
    writer << k << "\t" << v << endl;

    // save \mu
    for (size_t i = 0; i < k; i++) {
        writer << mu[i] << "\t";
    }
    writer << endl;

    // save \sigma
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < k; j++) {
            writer << sigma[i][j] << "\t";
        }
        writer << endl;
    }

    // save /beta
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < v; j++) {
            writer << beta[i][j] << "\t";
        }
        writer << endl;
    }

    // save delta
    writer << delta;
    writer.close();
}

void LRR_Model::loadFromFile(string filename) {
    string tmpTxt;
    ifstream reader (filename, std::ifstream::in);
    vector<string> container;
    if (!reader.is_open()) {
        cout << "Failed to open file " << filename << endl;
        return;
    }
    // part 1: aspect size, vocabulary size
    getline(reader, tmpTxt);
    size_t pos = tmpTxt.find_first_of("\t", 0);
    k = stoi(tmpTxt.substr(0, pos));
    size_t lastPos = pos;
    pos = tmpTxt.find_first_of("\t", lastPos+1);
    v = stoi(tmpTxt.substr(lastPos, pos-lastPos));

    create();

    // part 2: \mu
    getline(reader, tmpTxt);
    pos = 0;
    size_t i = 0;
    while (pos != string::npos && i < k) {
        lastPos = pos;
        pos = tmpTxt.find_first_of("\t", pos+1);
        mu[i++] = stod(tmpTxt.substr(lastPos, pos-lastPos));
    }

    // part 3: \sigma
    for (i = 0; i < k; i++) {
        getline(reader, tmpTxt);
        size_t j = 0;
        pos = 0;
        while (pos != string::npos && j < k) {
            lastPos = pos;
            pos = tmpTxt.find_first_of("\t", pos+1);
            sigma[i][j++] = stod(tmpTxt.substr(lastPos, pos-lastPos));
        }
    }
    calcSigmaInv(1.0);

    // part 4: \beta
    for (i = 0; i < k; i++) {
        getline(reader, tmpTxt);
        size_t j = 0;
        pos = 0;
        while (pos != string::npos && j <= v) {
            lastPos = pos;
            pos = tmpTxt.find_first_of("\t", pos+1);
            beta[i][j++] = stod(tmpTxt.substr(lastPos, pos-lastPos));
        }
    }

    // part 5: \delta
    getline(reader, tmpTxt);
    delta = stod(tmpTxt);
    reader.close();
}
}