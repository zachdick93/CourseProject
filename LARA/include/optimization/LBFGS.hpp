#ifndef LBFGS_H
#define LBFGS_H
#include <vector>

using std::vector;
namespace optimizer {
class LBFGS {
public:
    static double gtol;
    static double stpmin;
    static double stpmax;
    static vector<double> solution_cache;

    static int nfevaluations() { return nfun; }

    static void lbfgs ( int n , int m , vector<double> &x , double f , vector<double> &g , bool diagco , vector<double> &diag , vector<int> &iprint , double eps , double xtol , vector<int> &iflag );

    static void lb1 ( vector<int> &iprint , int iter , int nfun , double gnorm , int n , int m , vector<double> &x , double f , vector<double> &g , vector<double> &stp , bool finish );
    static void daxpy ( int n , double da , vector<double> &dx , int ix0, int incx , vector<double> &dy , int iy0, int incy );
    static double ddot ( int n, vector<double> &dx, int ix0, int incx, vector<double> &dy, int iy0, int incy );

private:
    static double gnorm;
    static double stp1;
    static double ftol;
    static vector<double> stp;
    static double ys;
    static double yy;
    static double sq;
    static double yr;
    static double beta;
    static double xnorm;
	static int iter;
    static int nfun;
    static int point;
    static int ispt;
    static int iypt;
    static int maxfev;
    static vector<int> info;
    static int bound;
    static int npt;
    static int cp;
    static vector<int> nfev;
    static int inmc;
    static int iycn;
    static int iscn;
	static bool finish;
    static vector<double> w;
};
}

#endif