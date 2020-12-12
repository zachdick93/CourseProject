#include "../../include/optimization/LBFGS.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cmath>
#include "../../include/optimization/Mcsrch.hpp"

using std::cout;
using std::cerr;
using std::setprecision;
using std::endl;
using std::copy;
using std::begin;
using std::end;
using std::abs;
using std::min;
using std::max;

namespace optimizer {
double LBFGS::gtol = 2e-3;
double LBFGS::stpmin = 1e-20;
double LBFGS::stpmax = 1e20;

double LBFGS::gnorm = 0.0;
double LBFGS::stp1 = 0.0;
double LBFGS::ftol = 0.0;
vector<double> LBFGS::stp = vector<double>(1);
double LBFGS::ys = 0.0;
double LBFGS::yy = 0.0;
double LBFGS::sq = 0.0;
double LBFGS::yr = 0.0;
double LBFGS::beta = 0.0;
double LBFGS::xnorm = 0.0;
int LBFGS::iter = 0;
int LBFGS::nfun = 0;
int LBFGS::point = 0;
int LBFGS::ispt = 0;
int LBFGS::iypt = 0;
int LBFGS::maxfev = 0;
vector<int> LBFGS::info = vector<int>(1);
int LBFGS::bound = 0;
int LBFGS::npt = 0;
int LBFGS::cp = 0;
vector<int> LBFGS::nfev = vector<int>(1);
int LBFGS::inmc = 0;
int LBFGS::iycn = 0;
int LBFGS::iscn = 0;
bool LBFGS::finish = false;
vector<double> LBFGS::w = vector<double>();
vector<double> LBFGS::solution_cache = vector<double>();

void LBFGS::lbfgs ( int n , int m , vector<double> &x, double f , vector<double> &g , bool diagco , vector<double> &diag , vector<int> &iprint, double eps , double xtol , vector<int> &iflag ) {
    bool execute_entire_while_loop = false;

    if (w.size() != n*(2*m+1)+2*m )
    {
        w.resize(n*(2*m+1)+2*m);
    }

    if ( iflag[0] == 0 )
    {
        // Initialize.
        solution_cache.resize(n);
        copy(x.begin(),x.end(), begin(solution_cache));

        iter = 0;
        if ( n <= 0 || m <= 0 )
        {
            cerr << "Improper input parameters  (n or m are not positive.)"  << endl;
            iflag[0]= -3;
            throw iflag[0];
        }

        if ( gtol <= 0.0001 )
        {
            cerr << "LBFGS.lbfgs: gtol is less than or equal to 0.0001. It has been reset to 0.9.";
            gtol= 0.9;
        }

        nfun= 1;
        point= 0;
        finish= false;

        if ( diagco )
        {
            for (int i = 1 ; i <= n ; i += 1 )
            {
                if ( diag[i-1] <= 0.0 )
                {
                    cerr << "The " << i << "-th diagonal element of the inverse hessian approximation is not positive.\n";
                    iflag[0]=-2;
                    throw iflag[0];
                }
            }
        }
        else
        {
            for (int i = 1 ; i <= n ; i += 1 )
            {
                diag[i - 1] = 1;
            }
        }
        ispt= n+2*m;
        iypt= ispt+n*m;

        for (int i = 1 ; i <= n ; i += 1 )
        {
            w[ispt + i -1] = -(g[i - 1]) * (diag[i - 1]);
        }
        gnorm = sqrt( ddot( n , g , 0, 1 , g , 0, 1 ) );
        stp1= 1.0/gnorm;
        ftol= 0.0001; 
        maxfev= 100;

        if ( iprint[1 - 1] >= 0 ) lb1 ( iprint , iter , nfun , gnorm , n , m , x , f , g , stp , finish );

        execute_entire_while_loop = true;
    }

    while ( true )
    {
        if ( execute_entire_while_loop )
        {
            iter= iter+1;
            info[0]=0;
            bound=iter-1;
            if ( iter != 1 )
            {
                if ( iter > m ) bound = m;
                ys = ddot ( n , w , iypt + npt , 1 , w , ispt + npt , 1 );
                if ( ! diagco )
                {
                    yy = ddot ( n , w , iypt + npt , 1 , w , iypt + npt , 1 );

                    for (int i = 1 ; i <= n ; i += 1 )
                    {
                        diag[i - 1] = ys / yy;
                    }
                }
                else
                {
                    iflag[0]=2;
                    return;
                }
            }
        }

        if ( execute_entire_while_loop || iflag[0] == 2 )
        {
            if ( iter != 1 )
            {
                if ( diagco )
                {
                    for (int i = 1 ; i <= n ; i += 1 )
                    {
                        if ( diag[i - 1] <= 0 )
                        {
                            cerr << "The " << i << "-th diagonal element of the inverse hessian approximation is not positive.\n";
                            iflag[0]=-2;
                            throw iflag[0];
                        }
                    }
                }
                cp= point;
                if ( point == 0 ) cp = m;
                w[n + cp - 1] = 1 / ys;

                for (int i = 1 ; i <= n ; i += 1 )
                {
                    w[i - 1] = -g[i - 1];
                }

                cp= point;

                for (int i = 1 ; i <= bound ; i += 1 )
                {
                    cp=cp-1;
                    if ( cp == -1 ) cp = m - 1;
                    sq = ddot ( n , w , ispt + cp * n , 1 , w , 0 , 1 );
                    inmc=n+m+cp+1;
                    iycn=iypt+cp*n;
                    w[inmc - 1] = w[n + cp + 1 - 1] * sq;
                    daxpy ( n , -w[inmc - 1] , w , iycn , 1 , w , 0 , 1 );
                }

                for (int i = 1 ; i <= n ; i += 1 )
                {
                    w[i - 1] = diag[i - 1] * w[i - 1];
                }

                for (int i = 1 ; i <= bound ; i += 1 )
                {
                    yr = ddot ( n , w , iypt + cp * n , 1 , w , 0 , 1 );
                    beta = w[n + cp + 1 - 1] * yr;
                    inmc=n+m+cp+1;
                    beta = w[inmc - 1] - beta;
                    iscn=ispt+cp*n;
                    daxpy ( n , beta , w , iscn , 1 , w , 0 , 1 );
                    cp=cp+1;
                    if ( cp == m ) cp = 0;
                }

                for (int i = 1 ; i <= n ; i += 1 )
                {
                    w[ispt + point * n + i - 1] = w[i - 1];
                }
            }

            nfev[0]=0;
            stp[0]=1;
            if ( iter == 1 ) stp[0] = stp1;

            for (int i = 1 ; i <= n ; i += 1 )
            {
                w[i - 1] = g[i - 1];
            }
        }

        Mcsrch::mcsrch ( n , x , f , g , w , ispt + point * n , stp , ftol , xtol , maxfev , info ,nfev , diag );

        if ( info[0] == -1 )
        {
            iflag[0]=1;
            return;
        }

        if ( info[0] != 1 )
        {
            //cerr << "Line search failed. See documentation of routine mcsrch. Error return of line search: info = " << info[0]
                 //<< " Possible causes: function or gradient are incorrect, or incorrect tolerances."<< endl;
            iflag[0]=-1;
            throw iflag[0];
        }

        nfun= nfun + nfev[0];
        npt=point*n;

        for (int i = 1 ; i <= n ; i += 1 )
        {
            w[ispt + npt + i - 1] = stp[0] * w[ispt + npt + i - 1];
            w[iypt + npt + i - 1] = g[i - 1] - w[i - 1];
        }

        point=point+1;
        if ( point == m ) point = 0;

        gnorm = sqrt( ddot( n , g , 0 , 1 , g , 0 , 1 ) );
        xnorm = sqrt( ddot( n , x , 0 , 1 , x , 0 , 1 ) );
        xnorm = max( 1.0 , xnorm );

        if ( gnorm / xnorm <= eps ) finish = true;

        if ( iprint[1 - 1] >= 0 ) lb1( iprint , iter , nfun , gnorm , n , m , x , f , g ,stp , finish );

        // Cache the current solution vector. Due to the spaghetti-like
        // nature of this code, it's not possible to quit here and return;
        // we need to go back to the top of the loop, and eventually call
        // mcsrch one more time -- but that will modify the solution vector.
        // So we need to keep a copy of the solution vector as it was at
        // the completion (info[0]==1) of the most recent line search.

        copy(x.begin(), x.end(), begin(solution_cache));

        if ( finish )
        {
            iflag[0]=0;
                return;
        }

        execute_entire_while_loop = true;		// from now on, execute whole loop
    }
}

void LBFGS::lb1 ( vector<int> &iprint , int iter , int nfun , double gnorm , int n , int m , vector<double> &x , double f , vector<double> &g , vector<double> &stp , bool finish ) {
    if ( iter == 0 )
    {
        cout << "*************************************************\n";
        cout <<"  n = " << n << "   number of corrections = " << m << "\n       initial values\n";
        cout << " f =  " << f << "   gnorm =  " << gnorm << endl;
        if ( iprint[2 - 1] >= 1 )
        {
            cout << " vector x =";
            for (int i = 1; i <= n; i++ )
                cout << "  " << x[i-1];
            cout << endl;

            cout << " gradient vector g =";
            for (int i = 1; i <= n; i++ )
                cout << "  " << g[i-1];
            cout <<  endl;
        }
        cout << "*************************************************\n";
        cout << "    i    nfn    func    gnorm    steplength\n";
    }
    else
    {
        if ( ( iprint[1 - 1] == 0 ) && ( iter != 1 && ! finish ) ) return;
        if ( iprint[1 - 1] != 0 )
        {
            if ( (iter - 1) % iprint[1 - 1] == 0 || finish )
            {
                if ( iprint[2 - 1] > 1 && iter > 1 )
                    cout << "    i    nfn    func    gnorm    steplength\n";
                cout << "    " << iter << "    " << nfun << "    " << f << "    " << gnorm << "    " << stp[0] << endl;
            }
            else
            {
                return;
            }
        }
        else
        {
            if ( iprint[2 - 1] > 1 && finish )
                    cout << "    i    nfn    func    gnorm    steplength\n";
            cout << "    " << iter << "    " << nfun << "    " << f <<"    " << gnorm << "    " << stp[0] << endl;
        }
        if ( iprint[2 - 1] == 2 || iprint[2 - 1] == 3 )
        {
            if ( finish )
            {
                cout << " final point x =";
            }
            else
            {
                cout << " vector x =  " ;
            }
            for (int i = 1; i <= n; i++ )
                cout << "  " << x[i-1];
            cout << endl;
            if ( iprint[2 - 1] == 3 )
            {
                cout <<  " gradient vector g =";
                for (int i = 1; i <= n; i++ )
                    cout << "  " << g[i-1];
                cout << endl;
            }
        }
        if ( finish )
            cout << " The minimization terminated without detecting errors. iflag = 0\n";
    }
    return;
}

void LBFGS::daxpy( int n , double da , vector<double> &dx, int ix0, int incx , vector<double> &dy, int iy0, int incy ) {
    int i, ix, iy, m, mp1;
    //cout << "enter LBFGS::daxpy\n";

    if ( n <= 0 ) return;

    if ( da == 0 ) return;

    if  ( ! ( incx == 1 && incy == 1 ) )
    {
        ix = 1;
        iy = 1;

        if ( incx < 0 ) ix = ( -n + 1 ) * incx + 1;
        if ( incy < 0 ) iy = ( -n + 1 ) * incy + 1;

        for (int i = 1 ; i <= n ; i += 1 )
        {
            dy[iy0 + iy - 1] = dy[iy0 + iy - 1] + da * dx[ix0 + ix - 1];
            ix = ix + incx;
            iy = iy + incy;
        }

        return;
    }

    m = n % 4;
    if ( m != 0 )
    {
        for (int i = 1 ; i <= m ; i += 1 )
        {
            dy[iy0 + i - 1] = dy[iy0 + i - 1] + da * dx[ix0 + i - 1];
        }

        if ( n < 4 ) return;
    }

    mp1 = m + 1;
    for (int i = mp1 ; i <= n ; i += 4 )
    {
        dy[iy0 + i - 1] = dy[iy0 + i - 1] + da * dx[ix0 + i - 1];
        dy[iy0 + i + 1 - 1] = dy[iy0 + i + 1 - 1] + da * dx[ix0 + i + 1 - 1];
        dy[iy0 + i + 2 - 1] = dy[iy0 + i + 2 - 1] + da * dx[ix0 + i + 2 - 1];
        dy[iy0 + i + 3 - 1] = dy[iy0 + i + 3 - 1] + da * dx[ix0 + i + 3 - 1];
    }
    return;
}

double LBFGS::ddot ( int n, vector<double> &dx, int ix0, int incx, vector<double> &dy, int iy0, int incy ) {
    double dtemp = 0;
    // cout << "enter LBFGS::ddot\n";
    int i, ix, iy, m, mp1;


    if ( n <= 0 ) return 0;

    if ( !( incx == 1 && incy == 1 ) )
    {
        ix = 1;
        iy = 1;
        if ( incx < 0 ) ix = ( -n + 1 ) * incx + 1;
        if ( incy < 0 ) iy = ( -n + 1 ) * incy + 1;
        for (int i = 1 ; i <= n ; i += 1 )
        {
            dtemp = dtemp + dx[ix0 + ix - 1] * dy[iy0 + iy - 1];
            ix = ix + incx;
            iy = iy + incy;
        }
        return dtemp;
    }

    m = n % 5;
    if ( m != 0 )
    {
        for (int i = 1 ; i <= m ; i += 1 )
        {
            dtemp = dtemp + dx[ix0 + i - 1] * dy[iy0 + i - 1];
        }
        if ( n < 5 ) return dtemp;
    }

    mp1 = m + 1;
    for (int i = mp1 ; i <= n ; i += 5 )
    {
        dtemp = dtemp + dx[ix0 + i - 1] * dy[iy0 + i - 1] + dx[ix0 + i + 1 - 1] * dy[iy0 + i + 1 - 1] + dx[ix0 + i + 2 - 1] * dy[iy0 + i + 2 - 1] + dx[ix0 + i + 3 - 1] * dy[iy0 + i + 3 - 1] + dx[ix0 + i + 4 - 1] * dy[iy0 + i + 4 - 1];
    }

    return dtemp;
}
}