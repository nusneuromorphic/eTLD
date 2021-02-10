#ifndef ECPARAM_H
#define ECPARAM_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <numeric>
#include "Matrix.h"

#define M_PI 3.14159265358979323846

class ECparam
{
    public:
        ECparam();
        ECparam(int Nr, int Nw, int Rmin, int Rmax, int MinNumEvents, int MaxNumEvents);
        ECparam(const ECparam &ECP);
        std::vector<double> get_r_bin_edges();
        void wedgeRing_lookupTable(Matrix &t_ring, Matrix &t_wedge);
    public:
        int nr;
        int nw;
        int rmin;
        int rmax;
        int minNumEvents;
        int maxNumEvents;
        std::vector<double> r_bin_edges;
        int kLookUpCenter {20};
};
#endif
