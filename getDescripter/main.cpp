/********************************************
Project: Real-time Recognition
Auther: YANG HONG
Date: 20/07/2017
Verified Date: 06/09/2017
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
#define ROItopleftrow 114
#define ROItopleftcolumn 38
#define ROIboxsizerow 45
#define ROIboxsizecolumn 31

/*void createCountMat(VlKDForest* forest, ECparam &ec, const string &str, Matrix &SVMwt, Matrix &SVMb);
void getDesctriptors_CountMat(vector<double> &desc, double countMat[cROW][cCOL], ECparam &ec,
	const int cur_loc_y, const int cur_loc_x, Matrix &t_ring, Matrix &t_wedge);
inline void readNumToMat(Matrix &mat, string str);*/


int main()
{
	string initial_TD = "../initialTD.txt";
	double* vocab = new double[EC_NR * EC_NW * VOCABSIZE];

	ifstream infile(initial_TD);
	if (!infile)	{
		cerr << "Oops, unable to open .txt..." << endl;
	}
	else	{
		string str;
		while(getline(infile, str))	{
			spikes.push_back(infile);
			std::cout << spikes[n] << '\t';
		}
		infile.close();
	}

	vector <double> ROI;
	vector <double> nonROI;

	system("pause");
	return 0;
	
}

