#include "ECparam.h"

ECparam::ECparam() : nr(7), nw(12), rmin(2), rmax(10), minNumEvents(100), maxNumEvents(2000)
{
    double arr[7] = { 2, 2.6153, 3.4200, 4.4721, 5.8480, 7.6472, 10 };
    std::vector<double> r_bin_edges(arr, arr + 7);
}

ECparam::ECparam(int Nr, int Nw, int Rmin, int Rmax, int MinNumEvents, int MaxNumEvents)
{
    nr = Nr;
    nw = Nw;
    rmin = Rmin;
    rmax = Rmax;
    minNumEvents = MinNumEvents;
    maxNumEvents = MaxNumEvents;
    r_bin_edges = get_r_bin_edges();
}

ECparam::ECparam(const ECparam &ECP)
{
    nr = ECP.nr;
    nw = ECP.nw;
    rmin = ECP.rmin;
    rmax = ECP.rmax;
    minNumEvents = ECP.minNumEvents;
    maxNumEvents = ECP.maxNumEvents;
    r_bin_edges = ECP.r_bin_edges;
}

std::vector<double> ECparam::get_r_bin_edges()
{
    std::vector<double> rbin;
    double a = log10(rmin);
    double b = log10(rmax);
    double step = (b - a) / (nr - 1);

    for (double i = 0; i<nr; i++)
    {
        double elem = pow(10, a + i*step);
        rbin.push_back(elem);
        //cout << elem << ' ';
    }
    //cout << endl;

    return rbin;
}

void ECparam::wedgeRing_lookupTable(Matrix &t_ring, Matrix &t_wedge)
{
    for (int y = 0; y <= kLookUpCenter * 2; y++)
    {
        for (int x = 0; x <= kLookUpCenter * 2; x++)
        {
            double rad = sqrt(pow(x - kLookUpCenter, 2) + pow(y - kLookUpCenter, 2));
            if (rad < r_bin_edges[nr - 1])
            {
                for (int m = 0; m < nr; m++)
                {
                    if (rad <= r_bin_edges[m])
                    {
                        int ringNum = ((m + 1) > nr) ? nr : (m + 1);
                        t_ring._matrix[y][x] = ringNum;
                        //std::cout << "ringNum: " << t_ring._matrix[y][x] << std::endl;
                        break;
                    }
                }
            }
            if (t_ring._matrix[y][x])
            {
                double theta = atan2(y - kLookUpCenter, x - kLookUpCenter);
                double wedgeNum = ceil(nw*theta / (2 * M_PI)) + nw / 2;
                t_wedge._matrix[y][x] = wedgeNum;
            }
            //std::cout << "wedgeNum: " << t_wedge._matrix[y][x] << std::endl;
        }
    }
}

