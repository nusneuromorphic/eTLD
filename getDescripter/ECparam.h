#ifndef ECPARAM_H
#define ECPARAM_H

#include<iostream>
#include<vector>
#include<algorithm>
#include <math.h>
#include<numeric>
#include "Matrix.h"

using namespace std;

#define M_PI 3.14159265358979323846
#define LookUpCenter 10
class ECparam
{
public:
	ECparam();
	ECparam(int Nr, int Nw, int Rmin, int Rmax, int MinNumEvents, int MaxNumEvents);
	vector<double> get_r_bin_edges();
	void wedgeRing_lookupTable(Matrix &t_ring, Matrix &t_wedge);
public:
	int nr;
	int nw;
	int rmin;
	int rmax;
	int minNumEvents;
	int maxNumEvents;
	vector<double> r_bin_edges;
	// 2.0000    2.6153    3.4200    4.4721    5.8480    7.6472   10.0000
};

ECparam::ECparam() : nr(7), nw(12), rmin(2), rmax(10), minNumEvents(100), maxNumEvents(2000)
{
	double arr[7] = { 2, 2.6153, 3.4200, 4.4721, 5.8480, 7.6472, 10 };
	vector<double> r_bin_edges(arr, arr + 7);
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

vector<double> ECparam::get_r_bin_edges()
{
	vector<double> rbin;
	double a = log10(rmin);
	double	b = log10(rmax);
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
	//double** t_wedge = matrix_build(20, 20);
	//double** t_ring = matrix_build(20, 20);

	for (int y = 0; y <= LookUpCenter * 2; y++)
	{
		for (int x = 0; x <= LookUpCenter * 2; x++)
		{
			double rad = sqrt(pow(x - LookUpCenter, 2) + pow(y - LookUpCenter, 2));
			if (rad < r_bin_edges[nr - 1])
			{
				for (int m = 0; m < nr; m++)
				{
					if (rad <= r_bin_edges[m])
					{
						int ringNum = ((m + 1) > nr) ? nr : (m + 1);
						t_ring._matrix[y][x] = ringNum;
						//cout << "ringNum: " << t_ring[y][x] << endl;
						break;
					}
				}
			}
			if (t_ring._matrix[y][x])
			{
				double theta = atan2(y - LookUpCenter, x - LookUpCenter);
				double wedgeNum = ceil(nw*theta / (2 * M_PI)) + nw / 2;
				t_wedge._matrix[y][x] = wedgeNum;
			}
			//cout << "wedgeNum: " << t_wedge[y][x] << endl;
		}
	}

}


#endif