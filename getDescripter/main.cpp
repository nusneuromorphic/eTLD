/********************************************
Project: Event-Based Motion Tracking
Author: JONATHAN LEE
Date: 27/09/2017
*********************************************/

#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <string>
#include <assert.h>
#include "Matrix.h"
#include "ECparam.h"
#include "currSpikeArr.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <numeric>
extern "C" {
	#include "vl/kdtree.h"
	#include "vl/homkermap.h"
	#include "vl/kmeans.h"
	#include "vl/generic.h"
	
}

using namespace std;

#define VOCABSIZE 200
#define cROW 240 //ATIS
#define cCOL 304 //ATIS
//#define cROW 180 // DAVIS
//#define cCOL 240 // DAVIS
#define tEND 10
#define CalNUM 2
#define LookUpCenter 10
#define EC_NR 10
#define EC_NW 12
#define ROItopLeftX 114
#define ROItopLeftY 38
#define ROIboxSizeX 45
#define ROIboxSizeY 31

/*void createCountMat(VlKDForest* forest, ECparam &ec, const string &str, Matrix &SVMwt, Matrix &SVMb);
void getDesctriptors_CountMat(vector<double> &desc, double countMat[cROW][cCOL], ECparam &ec,
	const int cur_loc_y, const int cur_loc_x, Matrix &t_ring, Matrix &t_wedge);
inline void readNumToMat(Matrix &mat, string str);*/


int main()
{
	string initial_TD = "../initialTD.txt";
	vector <double> allEvents;

	ifstream infile(initial_TD);
	if (!infile)	{
		cerr << "Oops, unable to open .txt..." << endl;
	}
	else {
		double ts, read_x, read_y, read_p;
		while ((infile >> ts) && (ts < tEND)) {
			allEvents.push_back(ts);
			infile >> read_x;
			allEvents.push_back(read_x);
			infile >> read_y;
			allEvents.push_back(read_y);
			infile >> read_p;
			//cout << "gcount:  " << gcount << '\t' << "ts: " << ts << endl;
		}
	}

	// separating ROI from non-ROI
	vector <double> ROI;
	vector <double> nonROI;
	for (int i = 0; i < allEvents.size(); i += 3) {
		if ((allEvents.at(i + 1) >= ROItopLeftX) && (allEvents.at(i+1) <= ROItopLeftX + ROIboxSizeX) && (allEvents.at(i+2) >= ROItopLeftY) && (allEvents.at(i+2) <= ROItopLeftY + ROIboxSizeY)) {
			for (int j = i; j < i + 3; j++) {
				ROI.push_back(allEvents.at(j));
			}
		}
		else {
			for (int j = i; j < i + 3; j++) {
				nonROI.push_back(allEvents.at(j));
			}
		}
	}

	/*cout << "ROI contains: \n";
	for (int i = 0; i < ROI.size(); i += 3) {
		for (int j = i; j < i + 3; j++) {
			cout << '\t' << ROI.at(j);
		}
		cout << '\n';
	}

	cout << "non-ROI contains: \n";
	for (int i = 0; i < nonROI.size(); i += 3) {
		for (int j = i; j < i + 3; j++) {
			cout << '\t' << nonROI.at(j);
		}
		cout << '\n';
	}*/

	double energy;
	double * centers;

	KMeans * kmeans = vl_kmeans_new(VLDistanceL2, VL_TYPE_FLOAT);

	system("pause");
	return 0;
	
}

