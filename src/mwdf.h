#ifndef MWDF_H
#define MWDF_H

#include <complex>
#include <iostream>
#include <vector>

#include "fftw3.h"

using namespace std;

typedef vector<complex<double>> vcd1d;
typedef vector<vcd1d> vcd2d;

namespace ModifiedWDF {

vcd2d mwdf2(const vcd1d& x, const vector<double>& t, const vector<double>& f, double dt, double df,
            const vcd1d& window);

}

#endif
