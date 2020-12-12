#ifndef MCSRCH_H
#define MCSRCH_H

#include <vector>

using std::vector;

namespace optimizer {
class Mcsrch {
public:
    static double sqr(double x) { return x * x; }
    static double max3 (double x, double y, double z) { return x < y ? (y < z ? z : y) : (x < z ? z : x); }
    static void mcsrch ( int n , vector<double> &x , double f , vector<double> &g , vector<double> &s , int is0 , vector<double> &stp , double ftol , double xtol , int maxfev , vector<int> &info , vector<int> &nfev , vector<double> &wa );
    static void mcstep ( vector<double> &stx , vector<double> &fx , vector<double> &dx , vector<double> &sty , vector<double> &fy , vector<double> &dy , vector<double> &stp , double fp , double dp , vector<bool> &brackt , double stpmin , double stpmax , vector<int> &info );

private:
    static vector<int> infoc;
    static double dg;
    static double dgm;
    static double dginit;
    static double dgtest;
    static vector<double> dgx;
    static vector<double> dgxm;
    static vector<double> dgy;
    static vector<double> dgym;
    static double finit;
    static double ftest1;
    static double fm;
    static vector<double> fx; 
    static vector<double> fxm;
    static vector<double> fy;
    static vector<double> fym;
    static double p5;
    static double p66;
    static vector<double> stx;
    static vector<double> sty;
    static double stmin; 
    static double stmax; 
    static double width;
    static double width1;
    static double xtrapf;
    static vector<bool> brackt;
    static bool stage1;
};
}

#endif