#include <iostream>
#include <iomanip>
#include "../include/lara/LRR.hpp"
#include "../include/lara/RatingRegression.hpp"

using lara::LRR;
using lara::RatingRegression;
using std::cout;
using std::endl;

void lrr_test();
void ratingregression_test();

int main() {
    //ratingregression_test();
    lrr_test();
}

void lrr_test() {
    cout << "Starting lrr_test()\n";
    cout << "(G2G)Building LRR Model\n";
    LRR model(500, 1e-2, 5000, 1e-2, 2.0); //, "Data/Model/model_hotel.dat");
    cout << "(Issue)Calling EMEst()\n";
    model.EMEst("Data/Vectors/Vector_CHI_4000.dat", 10, 1e-4);
    cout << "Saving Predictions\n";
    model.savePrediction("Data/Results/prediction.dat");
    cout << "Saving Model\n";
    model.saveModel("Data/Model/model_hotel.dat");
}

void ratingregression_test() {
    cout << "Starting ratingregression_test()\n";
    RatingRegression model(500, 5e-2, 5000, 1e-4, 1.0);
    model.estimateAspectModel("Data/Vectors/Vector_CHI_4000.dat");
	//model.savePrediction("Data/Results/prediction_baseline.dat");
	//model.saveModel("Data/Model/model_base_hotel.dat");

}