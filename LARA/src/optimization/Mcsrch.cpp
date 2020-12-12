#include "../../include/optimization/Mcsrch.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "../../include/optimization/LBFGS.hpp"

using std::max;
using std::min;
using std::cout;
using std::cerr;
using std::endl;
using std::abs;

namespace optimizer {
vector<int> Mcsrch::infoc = vector<int>(1);
double Mcsrch::dg = 0.0;
double Mcsrch::dgm = 0.0;
double Mcsrch::dginit = 0.0;
double Mcsrch::dgtest = 0.0;
vector<double> Mcsrch::dgx = vector<double>(1);
vector<double> Mcsrch::dgxm = vector<double>(1);
vector<double> Mcsrch::dgy = vector<double>(1);
vector<double> Mcsrch::dgym = vector<double>(1);
double Mcsrch::finit = 0.0;
double Mcsrch::ftest1 = 0.0;
double Mcsrch::fm = 0.0;
vector<double> Mcsrch::fx = vector<double>(1); 
vector<double> Mcsrch::fxm = vector<double>(1);
vector<double> Mcsrch::fy = vector<double>(1);
vector<double> Mcsrch::fym = vector<double>(1);
double Mcsrch::p5 = 0.0;
double Mcsrch::p66 = 0.0;
vector<double> Mcsrch::stx = vector<double>(1);
vector<double> Mcsrch::sty = vector<double>(1);
double Mcsrch::stmin = 0.0; 
double Mcsrch::stmax = 0.0; 
double Mcsrch::width = 0.0;
double Mcsrch::width1 = 0.0;
double Mcsrch::xtrapf = 0.0;
vector<bool> Mcsrch::brackt = vector<bool>(1);
bool Mcsrch::stage1 = false;

void Mcsrch::mcsrch ( int n , vector<double> &x , double f , vector<double> &g , vector<double> &s , int is0 , vector<double> &stp , double ftol , double xtol , int maxfev , vector<int> &info , vector<int> &nfev , vector<double> &wa )
{
    p5 = 0.5;
    p66 = 0.66;
    xtrapf = 4;

    if ( info[0] != -1 )
    {
        infoc[0] = 1;
        if ( n <= 0 || stp[0] <= 0.0 || ftol < 0.0 || LBFGS::gtol < 0.0 || xtol < 0.0 || LBFGS::stpmin < 0.0 || LBFGS::stpmax < LBFGS::stpmin || maxfev <= 0 ) 
            return;

        // Compute the initial gradient in the search direction
        // and check that s is a descent direction.

        dginit = 0;

        for (int j = 1 ; j <= n ; j += 1 )
        {
            dginit += g[j - 1] * s[is0 + j - 1];
        }

        if ( dginit >= 0.0 )
        {
            cout << "The search direction is not a descent direction.\n";
            return;
        }

        brackt[0] = false;
        stage1 = true;
        nfev[0] = 0;
        finit = f;
        dgtest = ftol * dginit;
        width = LBFGS::stpmax - LBFGS::stpmin;
        width1 = width/p5;

        for (int j = 1 ; j <= n ; j += 1 )
        {
            wa[ j - 1] = x[ j - 1];
        }

        // The variables stx, fx, dgx contain the values of the step,
        // function, and directional derivative at the best step.
        // The variables sty, fy, dgy contain the value of the step,
        // function, and derivative at the other endpoint of
        // the interval of uncertainty.
        // The variables stp, f, dg contain the values of the step,
        // function, and derivative at the current step.

        stx[0] = 0;
        fx[0] = finit;
        dgx[0] = dginit;
        sty[0] = 0;
        fy[0] = finit;
        dgy[0] = dginit;
    }

    while ( true )
    {
        if ( info[0] != -1 )
        {
            // Set the minimum and maximum steps to correspond
            // to the present interval of uncertainty.

            if ( brackt[0] )
            {
                stmin = min( stx[0] , sty[0] );
                stmax = max( stx[0] , sty[0] );
            }
            else
            {
                stmin = stx[0];
                stmax = stp[0] + xtrapf * ( stp[0] - stx[0] );
            }

            // Force the step to be within the bounds stpmax and stpmin.

            stp[0] = max ( stp[0] , LBFGS::stpmin );
            stp[0] = min ( stp[0] , LBFGS::stpmax );

            // If an unusual termination is to occur then let
            // stp be the lowest point obtained so far.

            if ( ( brackt[0] && ( stp[0] <= stmin || stp[0] >= stmax ) ) || nfev[0] >= maxfev - 1 || infoc[0] == 0 || ( brackt[0] && stmax - stmin <= xtol * stmax ) ) stp[0] = stx[0];

            // Evaluate the function and gradient at stp
            // and compute the directional derivative.
            // We return to main program to obtain F and G.

            for (int j = 1 ; j <= n ; j += 1 )
            {
                x[j - 1] = wa[ j - 1] + stp[0] * s[is0 + j - 1];
            }

            info[0] = -1;
            return;
        }

        info[0]=0;
        nfev[0] = nfev[0] + 1;
        dg = 0;

        for (int j = 1 ; j <= n ; j += 1 )
        {
            dg = dg + g[j - 1] * s[ is0 + j - 1];
        }

        ftest1 = finit + stp[0] * dgtest;

        // Test for convergence.

        if ( ( brackt[0] && ( stp[0] <= stmin || stp[0] >= stmax ) ) || infoc[0] == 0 ) info[0] = 6;

        if ( stp[0] == LBFGS::stpmax && f <= ftest1 && dg <= dgtest ) info[0] = 5;

        if ( stp[0] == LBFGS::stpmin && ( f > ftest1 || dg >= dgtest ) ) info[0] = 4;

        if ( nfev[0] >= maxfev ) info[0] = 3;

        if ( brackt[0] && stmax - stmin <= xtol * stmax ) info[0] = 2;

        if ( f <= ftest1 && abs( dg ) <= LBFGS::gtol * ( -dginit ) ) info[0] = 1;

        // Check for termination.

        if ( info[0] != 0 ) return;

        // In the first stage we seek a step for which the modified
        // function has a nonpositive value and nonnegative derivative.

        if ( stage1 && f <= ftest1 && dg >= min(ftol, LBFGS::gtol) * dginit) stage1 = false;

        // A modified function is used to predict the step only if
        // we have not obtained a step for which the modified
        // function has a nonpositive function value and nonnegative
        // derivative, and if a lower function value has been
        // obtained but the decrease is not sufficient.

        if ( stage1 && f <= fx[0] && f > ftest1 )
        {
            // Define the modified function and derivative values.

            fm = f - stp[0] * dgtest;
            fxm[0] = fx[0] - stx[0] * dgtest;
            fym[0] = fy[0] - sty[0] * dgtest;
            dgm = dg - dgtest;
            dgxm[0] = dgx[0] - dgtest;
            dgym[0] = dgy[0] - dgtest;

            // Call cstep to update the interval of uncertainty
            // and to compute the new step.

            mcstep( stx , fxm , dgxm , sty , fym , dgym , stp , fm , dgm , brackt , stmin , stmax , infoc );

            // Reset the function and gradient values for f.

            fx[0] = fxm[0] + stx[0] * dgtest;
            fy[0] = fym[0] + sty[0] * dgtest;
            dgx[0] = dgxm[0] + dgtest;
            dgy[0] = dgym[0] + dgtest;
        }
        else
        {
            // Call mcstep to update the interval of uncertainty
            // and to compute the new step.

            mcstep( stx , fx , dgx , sty , fy , dgy , stp , f , dg , brackt , stmin , stmax , infoc );
        }

        // Force a sufficient decrease in the size of the
        // interval of uncertainty.

        if ( brackt[0] )
        {
            if ( abs(sty[0] - stx[0]) >= p66 * width1) stp[0] = stx[0] + p5 * (sty[0] - stx[0]);
            width1 = width;
            width = abs(sty[0] - stx[0]);
        }
    }
}

void Mcsrch::mcstep(vector<double> &stx , vector<double> &fx , vector<double> &dx , vector<double> &sty , vector<double> &fy , vector<double> &dy , vector<double> &stp , double fp , double dp , vector<bool> &brackt , double stpmin , double stpmax , vector<int> &info) {
    bool bound;
    double gamma, p, q, r, s, sgnd, stpc, stpf, stpq, theta;

    info[0] = 0;
    
    if ((brackt[0] && (stp[0] <= min(stx[0] , sty[0]) || stp[0] >= max(stx[0], sty[0]))) || dx[0] * (stp[0] - stx[0]) >= 0.0 || stpmax < stpmin ) return;
    // Determine if the derivatives have opposite sign.

    sgnd = dp * (dx[0] / abs(dx[0]));

    if ( fp > fx[0] )
    {
        // First case. A higher function value.
        // The minimum is bracketed. If the cubic step is closer
        // to stx than the quadratic step, the cubic step is taken,
        // else the average of the cubic and quadratic steps is taken.
        info[0] = 1;
        bound = true;
        theta = 3 * ( fx[0] - fp ) / ( stp[0] - stx[0] ) + dx[0] + dp;
        s = max3( abs(theta) , abs(dx[0]) , abs(dp) );
        gamma = s * sqrt ( sqr( theta / s ) - ( dx[0] / s ) * ( dp / s ) );
        if ( stp[0] < stx[0] ) gamma = - gamma;
        p = ( gamma - dx[0] ) + theta;
        q = ( ( gamma - dx[0] ) + gamma ) + dp;
        r = p/q;
        stpc = stx[0] + r * ( stp[0] - stx[0] );
        stpq = stx[0] + ( ( dx[0] / ( ( fx[0] - fp ) / ( stp[0] - stx[0] ) + dx[0] ) ) / 2 ) * ( stp[0] - stx[0] );
        if ( abs(stpc - stx[0]) < abs(stpq - stx[0]))
        {
            stpf = stpc;
        }
        else
        {
            stpf = stpc + ( stpq - stpc ) / 2;
        }
        brackt[0] = true;
    }
    else if ( sgnd < 0.0 )
    {
        // Second case. A lower function value and derivatives of
        // opposite sign. The minimum is bracketed. If the cubic
        // step is closer to stx than the quadratic (secant) step,
        // the cubic step is taken, else the quadratic step is taken.
        info[0] = 2;
        bound = false;
        theta = 3 * ( fx[0] - fp ) / ( stp[0] - stx[0] ) + dx[0] + dp;
        s = max3 ( abs( theta ) , abs( dx[0] ) , abs( dp ) );
        gamma = s * sqrt ( sqr( theta / s ) - ( dx[0] / s ) * ( dp / s ) );
        if ( stp[0] > stx[0] ) gamma = - gamma;
        p = ( gamma - dp ) + theta;
        q = ( ( gamma - dp ) + gamma ) + dx[0];
        r = p/q;
        stpc = stp[0] + r * ( stx[0] - stp[0] );
        stpq = stp[0] + ( dp / ( dp - dx[0] ) ) * ( stx[0] - stp[0] );
        if ( abs( stpc - stp[0] ) > abs( stpq - stp[0] ) )
        {
            stpf = stpc;
        }
        else
        {
            stpf = stpq;
        }
        brackt[0] = true;
    }
    else if ( abs( dp ) < abs( dx[0] ) )
    {
        // Third case. A lower function value, derivatives of the
        // same sign, and the magnitude of the derivative decreases.
        // The cubic step is only used if the cubic tends to infinity
        // in the direction of the step or if the minimum of the cubic
        // is beyond stp. Otherwise the cubic step is defined to be
        // either stpmin or stpmax. The quadratic (secant) step is also
        // computed and if the minimum is bracketed then the the step
        // closest to stx is taken, else the step farthest away is taken.
        info[0] = 3;
        bound = true;
        theta = 3 * ( fx[0] - fp ) / ( stp[0] - stx[0] ) + dx[0] + dp;
        s = max3(abs( theta ) , abs( dx[0] ) , abs( dp ) );
        gamma = s * sqrt (max(0.0, sqr( theta / s ) - ( dx[0] / s ) * ( dp / s ) ) );
        if ( stp[0] > stx[0] ) gamma = - gamma;
        p = ( gamma - dp ) + theta;
        q = ( gamma + ( dx[0] - dp ) ) + gamma;
        r = p/q;
        if ( r < 0.0 && gamma != 0.0 )
        {
            stpc = stp[0] + r * ( stx[0] - stp[0] );
        }
        else if ( stp[0] > stx[0] )
        {
            stpc = stpmax;
        }
        else
        {
            stpc = stpmin;
        }
        stpq = stp[0] + ( dp / ( dp - dx[0] ) ) * ( stx[0] - stp[0] );
        if ( brackt[0] )
        {
            if ( abs( stp[0] - stpc ) < abs( stp[0] - stpq ) )
            {
                stpf = stpc;
            }
            else
            {
                stpf = stpq;
            }
        }
        else
        {
            if ( abs( stp[0] - stpc ) > abs( stp[0] - stpq ) )
            {
                stpf = stpc;
            }
            else
            {
                stpf = stpq;
            }
        }
    }
    else
    {
        // Fourth case. A lower function value, derivatives of the
        // same sign, and the magnitude of the derivative does
        // not decrease. If the minimum is not bracketed, the step
        // is either stpmin or stpmax, else the cubic step is taken.
        info[0] = 4;
        bound = false;
        if ( brackt[0] )
        {
            theta = 3 * ( fp - fy[0] ) / ( sty[0] - stp[0] ) + dy[0] + dp;
            s = max3(abs( theta ) , abs( dy[0] ) , abs( dp ) );
            gamma = s * sqrt ( sqr( theta / s ) - ( dy[0] / s ) * ( dp / s ) );
            if ( stp[0] > sty[0] ) gamma = - gamma;
            p = ( gamma - dp ) + theta;
            q = ( ( gamma - dp ) + gamma ) + dy[0];
            r = p/q;
            stpc = stp[0] + r * ( sty[0] - stp[0] );
            stpf = stpc;
        }
        else if ( stp[0] > stx[0] )
        {
            stpf = stpmax;
        }
        else
        {
            stpf = stpmin;
        }
    }

    // Update the interval of uncertainty. This update does not
    // depend on the new step or the case analysis above.

    if ( fp > fx[0] )
    {
        sty[0] = stp[0];
        fy[0] = fp;
        dy[0] = dp;
    }
    else
    {
        if ( sgnd < 0.0 )
        {
            sty[0] = stx[0];
            fy[0] = fx[0];
            dy[0] = dx[0];
        }
        stx[0] = stp[0];
        fx[0] = fp;
        dx[0] = dp;
    }

    // Compute the new step and safeguard it.

    stpf = min( stpmax , stpf );
    stpf = max( stpmin , stpf );
    stp[0] = stpf;

    if ( brackt[0] && bound )
    {
        if ( sty[0] > stx[0] )
        {
            stp[0] = min ( stx[0] + 0.66 * ( sty[0] - stx[0] ) , stp[0] );
        }
        else
        {
            stp[0] = max ( stx[0] + 0.66 * ( sty[0] - stx[0] ) , stp[0] );
        }
    }
    return;
}

}