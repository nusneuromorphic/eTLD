/********************************************
Project: Event-Based Motion Tracking
Author: JONATHAN LEE
Date: 27/09/2017
*********************************************/

#include <iostream>
#include <stdlib.h>
#include <cstdlib>
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
#define EC_NR 7
#define EC_NW 12
#define ROItopLeftX 114
#define ROItopLeftY 38
#define ROIboxSizeX 45
#define ROIboxSizeY 31

void getDesctriptors_CountMat(vector<double> &desc, double countMat[cROW][cCOL], ECparam &ec,
	const int cur_loc_y, const int cur_loc_x, Matrix &t_ring, Matrix &t_wedge);


int main()
{
	const int EVENTS_PER_CLASSIFICATION = 3809;
	const int REFRACTORY_COUNT = 3;
	const double prob_threshold = 0.80;
	const int refresh_hist = 20;
	const int reset_num = 50;

	double countMat[cROW][cCOL];
	static int gcount = 0; // global event count
	static int countEvents = 0;
	static int descriptor_count = 0;
	int ROIhist[VOCABSIZE];
	int nonROIhist[VOCABSIZE];

	for (int i = 0; i < cROW; i++) {
		for (int j = 0; j < cCOL; j++) {
			countMat[i][j] = 0;
		}
	}
	for (int i = 0; i < VOCABSIZE; i++)	{
		ROIhist[i] = 0;
	}
	for (int i = 0; i < VOCABSIZE; i++)	{
		nonROIhist[i] = 0;
	}


	ECparam ec(7, 12, 2, 10, 100, 1000);
	string initial_TD = "../initialTD.txt";
	vector <double> allDescs;
	vector <vector<double>> ROIDescs;
	vector <vector<double>> nonROIDescs;

	vector<string> objList = { "0","1" };
	Matrix t_wedge(LookUpCenter * 2 + 1, LookUpCenter * 2 + 1);
	Matrix t_ring(LookUpCenter * 2 + 1, LookUpCenter * 2 + 1);
	ec.wedgeRing_lookupTable(t_ring, t_wedge);

	ifstream infile(initial_TD);
	if (!infile)	{
		cerr << "Oops, unable to open .txt..." << endl;
	}
	else {
		int x, y;
		double ts, read_x, read_y, read_p;
		while ((infile >> ts) && (ts < tEND)) {
			infile >> read_x;
			x = (int)read_x;
			infile >> read_y;
			y = (int)read_y;
			infile >> read_p;
			countEvents++;
			gcount++;
			countMat[y][x] += 1;

			if ((countEvents > ec.minNumEvents) && (countEvents <= ec.maxNumEvents) && (countMat[y][x] <= REFRACTORY_COUNT))  {
				vector<double> desc;
				getDesctriptors_CountMat(desc, countMat, ec, y, x, t_ring, t_wedge);
				allDescs.insert(allDescs.end(), desc.begin(), desc.end());
				if ((x >= ROItopLeftX) && (x <= ROItopLeftX + ROIboxSizeX) && (y >= ROItopLeftY) && (y <= ROItopLeftY + ROIboxSizeY)) {
					ROIDescs.push_back(desc);
				}
				else {
					nonROIDescs.push_back(desc);
				}
				
				descriptor_count++;

			}
			if (countEvents>ec.maxNumEvents)// Should we reset count_matrix?
			{
				// reset count_matrix, reset countEvents
				countEvents = 1;
				for (int iy = 0; iy < cROW; iy++)
				{
					for (int ix = 0; ix < cCOL; ix++)
					{
						countMat[iy][ix] = 0;
					}
				}

			} // end if
		} // end while
	} // end else

	//void const * vocab = new double[EC_NR * EC_NW * VOCABSIZE];
	int all_desc_size = allDescs.size();
	double* allD = new double[all_desc_size];
	allD = &allDescs[0];

	VlKMeans * kmeans = vl_kmeans_new(VL_TYPE_DOUBLE, VlDistanceL2);
	kmeans->verbosity = 1;
	vl_kmeans_set_algorithm(kmeans, VlKMeansANN);
	vl_kmeans_init_centers_with_rand_data(kmeans, allD, 84, descriptor_count, VOCABSIZE);
	vl_kmeans_set_max_num_iterations(kmeans, 100);
	vl_kmeans_refine_centers(kmeans, allD, descriptor_count);
	
	void const * vocab = vl_kmeans_get_centers(kmeans);
	double * vocab2 = new double[VOCABSIZE * EC_NR * EC_NW];
	vocab2 = (double*)vocab;

	/*for (int i = 0; i <= VOCABSIZE * EC_NR * EC_NW; i++) {
		cout << vocab2[i] << '\t';
	}*/

	// build a tree.
	VlKDForest* forest = vl_kdforest_new(VL_TYPE_DOUBLE, EC_NR * EC_NW, 1, VlDistanceL2);
	vl_kdforest_build(forest, VOCABSIZE, vocab2);
	vl_kdforest_set_max_num_comparisons(forest, 15);

	// build ROI histogram and nonROI histogram
	VlKDForestNeighbor neighbors[1]; // a structure
	VlKDForestSearcher* searcherObj = vl_kdforest_new_searcher(forest);

	for (int i = 0; i < ROIDescs.size(); i++) {
		vl_kdforestsearcher_query(searcherObj, neighbors, 1, &ROIDescs[i].front());
		//vl_kdforest_query(forest, neighbors, 1, pass_desc);
		int binsa_new = neighbors->index;
		ROIhist[binsa_new]++;
		//cout << "ROI: " << binsa_new << "+1.\n";
	}

	for (int i = 0; i < nonROIDescs.size(); i++) {
		vl_kdforestsearcher_query(searcherObj, neighbors, 1, &nonROIDescs[i].front());
		//vl_kdforest_query(forest, neighbors, 1, pass_desc);
		int binsa_new = neighbors->index;
		nonROIhist[binsa_new]++;
		//cout << "nonROI: " << binsa_new << "+1.\n";
	}
	
	

	// bootstrapping

	vector<int> nextROISample;
	vector<vector<int>> ROISamples;
	vector<int> nextNonROISample;
	vector<vector<int>> nonROISamples;

	for (int i = 0; i < 1000; i++) {
		for (int j = 0; j < VOCABSIZE; j++) {
			nextROISample.push_back(rand() % (ROIhist[j]+1));
			nextNonROISample.push_back(rand() % (nonROIhist[j] + 1));
		}
		ROISamples.push_back(nextROISample);
		nonROISamples.push_back(nextNonROISample);
		nextROISample.clear();
		nextNonROISample.clear();
	}


	system("pause");
	return 0;
	
}

void getDesctriptors_CountMat(vector<double> &desc, double countMat[cROW][cCOL], ECparam &ec,
	const int cur_loc_y, const int cur_loc_x, Matrix &t_ring, Matrix &t_wedge)
{
	//vector<double>* desc = new vector<double>;
	//vector<double> desc;
	// get current spike vectors from "Count_Mat".
	currSpikeArr currSpikeArr;
	/*for (int j = 0; j < countMat._row; j++)
	{
	for (int i = 0; i < countMat._col; i++)
	{
	if (countMat._matrix[j][i]>0)
	{
	currSpikeArr.y.push_back(j + 1); // size=887 is right.
	currSpikeArr.x.push_back(i + 1); // acctual coordinate is one bigger than matrix subscription
	}
	}
	}*/

	//get ring and wedge num from lookup table*********************
	int dy = cur_loc_y - LookUpCenter;
	int dx = cur_loc_x - LookUpCenter;

	int jmin = (cur_loc_y - ec.rmax >= 0) ? (cur_loc_y - ec.rmax) : 0;
	int jmax = (cur_loc_y + ec.rmax < cROW) ? (cur_loc_y + ec.rmax) : cROW;
	int imin = (cur_loc_x - ec.rmax >= 0) ? (cur_loc_x - ec.rmax) : 0;
	int imax = (cur_loc_x + ec.rmax < cCOL) ? (cur_loc_x + ec.rmax) : cCOL;
	for (int j = jmin; j < jmax; j++)
	{
		for (int i = imin; i < imax; i++)
		{
			if (countMat[j][i]>0)
			{
				currSpikeArr.y.push_back(j + 1);
				currSpikeArr.x.push_back(i + 1);
				int ringNum = t_ring._matrix[j - dy][i - dx];
				currSpikeArr.ringNum.push_back(ringNum);
				int wedgeNum = t_wedge._matrix[j - dy][i - dx];
				currSpikeArr.wedgeNum.push_back(wedgeNum);
			}
		}
	}

	//calculate logHistEmpty, count total number of spikes within our log-polar coordinate**********
	int scount = 0;
	Matrix logHistEmpty(ec.nw, ec.nr);
	for (int n = 0; n < currSpikeArr.x.size(); n++)
	{
		int log_y = currSpikeArr.wedgeNum[n];
		int log_x = currSpikeArr.ringNum[n];
		if ((log_y > 0) && (log_x > 0))//exclude those spikes whos ringNum and wedgeNum are 0.
		{
			scount++;
			assert(log_y < cROW);
			assert(log_x < cCOL);
			int loc_count = countMat[currSpikeArr.y[n] - 1][currSpikeArr.x[n] - 1];
			logHistEmpty._matrix[log_y - 1][log_x - 1] += loc_count;//wedgeNum and ringNum begin from 1. matrix subscriptions begin from 0;
		}
	}
	/*cout << "total spikes within our log-polar coordinate: " << '\n'
	<< "sum_logHist= " << sum_logHist << endl;*/
	// calculate desc
	double sum_logHist = scount;
	if (sum_logHist)
	{
		for (int ix = 0; ix < ec.nr; ix++)
		{
			for (int iy = 0; iy < ec.nw; iy++)
			{
				logHistEmpty._matrix[iy][ix] /= sum_logHist;  // normalization
				desc.push_back(logHistEmpty._matrix[iy][ix]);
			}
		}
	}
	else
	{
		cout << "We didn't get any spikes." << endl;
	}
	//return desc;
}

