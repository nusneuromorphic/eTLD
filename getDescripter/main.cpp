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
#include <vl/svm.h>
#include "Matrix.h"
#include "ECparam.h"
#include "currSpikeArr.h"
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#define _USE_MATH_DEFINES
#include <math.h>
#include <numeric>
#include <deque>
extern "C" {
	#include "vl/kdtree.h"
	#include "vl/homkermap.h"
	#include "vl/kmeans.h"
	#include "vl/generic.h"
	
}

using namespace std;
using namespace cv;

#define VOCABSIZE 200
#define cROW 240 //ATIS
#define cCOL 304 //ATIS
//#define cROW 180 // DAVIS
//#define cCOL 240 // DAVIS
#define tEND 15
#define LookUpCenter 10
#define EC_NR 10
#define EC_NW 12
#define ROItopLeftX 114
#define ROItopLeftY 38
#define ROIboxSizeX 45
#define ROIboxSizeY 31
#define BOOTSTRAP 1000
#define PADDING 2
#define QueueSize 500

void getDesctriptors_CountMat(vector<double> &desc, double countMat[cROW][cCOL], ECparam &ec,
	const int cur_loc_y, const int cur_loc_x, Matrix &t_ring, Matrix &t_wedge);
void rescale(Mat &countMat, int a, int b);

int main()
{
	double countMat[cROW][cCOL];
	//Mat countMat = Mat::zeros(cROW, cCOL, CV_8UC1); //cROWxcCOL zero matrix 
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


	ECparam ec(EC_NR, EC_NW, 2, 10, 100, 1000);
	string initial_TD = "../initialTD.txt";
	vector <double> allDescs;
	vector <vector<double>> ROIDescs;
	vector <vector<double>> nonROIDescs;
	deque<int> eventQueue;

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
		while ((infile >> ts) && (!infile.eof())) {
			infile >> read_x;
			x = (int)read_x;
			infile >> read_y;
			y = (int)read_y;
			infile >> read_p;
			countEvents++;
			eventQueue.push_back(x);
			eventQueue.push_back(y);
			countMat[y][x] += 1;

			if ((countEvents > ec.minNumEvents) && (countEvents <= ec.maxNumEvents))  {
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
			if (eventQueue.size() > (QueueSize * 2)) // Pop out the oldest event in the queue.
			{
				x = eventQueue.front();
				eventQueue.pop_front();
				y = eventQueue.front();
				eventQueue.pop_front();
				countMat[y][x] -= 1;
				countEvents--;

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
	vl_kmeans_init_centers_with_rand_data(kmeans, allD, EC_NR * EC_NW, descriptor_count, VOCABSIZE);
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

	vector<double> nextROISample;
	vector<double> ROISamples;
	vector<double> nextNonROISample;
	vector<double> nonROISamples;
	vector<double> allNormalizedSamples;
	int nextROIrandom, nextNonROIrandom;
	int ROItotal = 0;
	int nonROItotal = 0;

	for (int i = 0; i < BOOTSTRAP; i++) {
		for (int j = 0; j < VOCABSIZE; j++) {
			nextROIrandom = rand() % (ROIhist[j] + 1);
			ROItotal = ROItotal + nextROIrandom;
			nextROISample.push_back(nextROIrandom);

			nextNonROIrandom = rand() % (nonROIhist[j] + 1);
			nonROItotal = nonROItotal + nextNonROIrandom;
			nextNonROISample.push_back(nextNonROIrandom);
		}

		//  histogram normalization
		for (int j = 0; j < VOCABSIZE; j++) {
			nextROISample[j] /= ROItotal;
			nextNonROISample[j] /= nonROItotal;
		}

		ROISamples.insert(ROISamples.end(), nextROISample.begin(), nextROISample.end());
		nonROISamples.insert(nonROISamples.end(), nextNonROISample.begin(), nextNonROISample.end());
		nextROISample.clear();
		nextNonROISample.clear();
		ROItotal = 0;
		nonROItotal = 0;
	}

	allNormalizedSamples.insert(allNormalizedSamples.end(), ROISamples.begin(), ROISamples.end());
	allNormalizedSamples.insert(allNormalizedSamples.end(), nonROISamples.begin(), nonROISamples.end());
	int totalSize = allNormalizedSamples.size();
	double * SVMdata = new double[totalSize];
	SVMdata = &allNormalizedSamples[0];


	// homkermap
	VlHomogeneousKernelMap* hom;
	//double * psi = new double[3];
	double psi[3];
	double * all_psi = new double[totalSize * 3];
	hom = vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, 0.5, 1, -5, VlHomogeneousKernelMapWindowRectangular);
	//vl_homogeneouskernelmap_new(kernelType, gamma, order, period, windowType);
	for (int j = 0; j < allNormalizedSamples.size(); j++)
	{
		vl_homogeneouskernelmap_evaluate_d(hom, psi, 1, SVMdata[j]);
		all_psi[3 * j] = psi[0];
		all_psi[3 * j + 1] = psi[1];
		all_psi[3 * j + 2] = psi[2];
	}

	double labels[BOOTSTRAP * 2];
	for (int i = 0; i < BOOTSTRAP; i++) {
		labels[i] = 1;
	}
	for (int i = BOOTSTRAP; i < BOOTSTRAP * 2; i++) {
		labels[i] = -1;
	}

	double lambda = 0.00005;	// lambda = 1 / (svmOpts.C * length(train_label)) ; -> From the matlab code

	VlSvm * svm = vl_svm_new(VlSvmSolverSgd, all_psi, VOCABSIZE * 3, 2000, labels, lambda);
	// 9.3266e-06, C = 10, length(train_label) = 2000.
	vl_svm_train(svm);

	const double * model = vl_svm_get_model(svm);
	double bias = vl_svm_get_bias(svm);

	cout << "Model w = [ " << model[0] << " " << model[1] << " ], bias b = " << bias << "\n";


	// padding
	int origBB_boxSizeX = ROIboxSizeX; // original bounding box
	int origBB_boxSizeY = ROIboxSizeY;
	int padBB_boxSizeX = origBB_boxSizeX + (PADDING * 2); // padded bounding box
	int padBB_boxSizeY = origBB_boxSizeY + (PADDING * 2);

	/*if (padBB_topLeftX < 0) padBB_topLeftX = 0; // checking for out of bounds
	if (padBB_topLeftY < 0) padBB_topLeftY = 0;*/


	// Sliding Window Lookup Table: BBLookupTable[HEIGHT][WIDTH][25x1]
	vector<vector<vector<bool>>> BBLookupTable; // 3D Lookup Table of area = padded bounding box. Each pixel contains a 25x1 vector of which BBs they are in.
	BBLookupTable.resize(padBB_boxSizeY);
	for (int i = 0; i < padBB_boxSizeY; i++) {
		BBLookupTable[i].resize(padBB_boxSizeX);
		for (int j = 0; j < padBB_boxSizeX; j++) {
			BBLookupTable[i][j].resize(25);
			/*for (int k = 0; k < 25; k++) {
				BBLookupTable[i][j][k] = true;
			}*/
		}
	}


	for (int i = 0; i < padBB_boxSizeY; i++) {
		for (int j = 0; j < padBB_boxSizeX; j++) {
			for (int k = 0; k < 5; k++) { // moving down
				for (int l = 0; l < 5; l++) { // moving right
					if ((i <= k + origBB_boxSizeY - 1) && (j <= l + origBB_boxSizeX - 1) && (i >= k) && (j >= l))
						BBLookupTable[i][j][(k * 5) + l] = true;
					else
						BBLookupTable[i][j][(k * 5) + l] = false;
				}
			}
		}
	}

	int origBB_topLeftX = ROItopLeftX;
	int origBB_topLeftY = ROItopLeftY;
	int padBB_topLeftX = origBB_topLeftX - PADDING;
	int padBB_topLeftY = origBB_topLeftY - PADDING;


	// Generating 25 histograms for 25 sliding window candidates
	vector<vector<double>> SWcandidate_hist;
	SWcandidate_hist.resize(25);
	for (int i = 0; i < 25; i++) {
		SWcandidate_hist[i].resize(VOCABSIZE);
		for (int j = 0; j < VOCABSIZE; j++) {
			SWcandidate_hist[i][j] = 0;
		}
	}



	cout << "Performing tracking.\n";
	string tracking_TD = "../trackingTD.txt";
	const int EVENTS_PER_CLASSIFICATION = ROIboxSizeX * ROIboxSizeY * 2 / 3;
	eventQueue.clear();
	int x, y;
	double ts, read_x, read_y, read_p;
	countEvents = 0;
	int ROIEvents = 0;
	double globalBestScore = 0;
	int bestCandidate;
	Mat disp_countMat = Mat::zeros(cROW, cCOL, CV_8UC1); //cROWxcCOL zero matrix  
	Rect boundingBox;
	namedWindow("SW", CV_WINDOW_NORMAL);

	for (int i = 0; i < cROW; i++) {
		for (int j = 0; j < cCOL; j++) {
			countMat[i][j] = 0;
		}
	}

	ifstream trackfile(tracking_TD);
	if (!trackfile) {
		cerr << "Oops, unable to open .txt..." << endl;
	}
	else {
		while ((trackfile >> ts) && (ts < tEND)) {
			trackfile >> read_x;
			x = (int)read_x;
			trackfile >> read_y;
			y = (int)read_y;
			trackfile >> read_p;
			countEvents++;
			eventQueue.push_back(x);
			eventQueue.push_back(y);
			countMat[y][x] += 1;

			if ((x >= padBB_topLeftX) && (x < padBB_topLeftX + padBB_boxSizeX) && (y >= padBB_topLeftY) && (y < padBB_topLeftY + padBB_boxSizeY)) {
				if ((countEvents > ec.minNumEvents) && (countEvents <= ec.maxNumEvents)) {
					vector<double> desc;
					getDesctriptors_CountMat(desc, countMat, ec, y, x, t_ring, t_wedge);

					vl_kdforestsearcher_query(searcherObj, neighbors, 1, &desc.front());
					//vl_kdforest_query(forest, neighbors, 1, pass_desc);
					int binsa_new = neighbors->index;
					for (int i = 0; i < 25; i++) {
						if (BBLookupTable[y - padBB_topLeftY][x - padBB_topLeftX][i] == true) {
							SWcandidate_hist[i][binsa_new]++; // increment the histogram of the corresponding SW candidate
						}
					}
					ROIEvents++;

				}
			}

			if (eventQueue.size() > (QueueSize * 2)) // Pop out the oldest event in the queue.
			{
				x = eventQueue.front();
				eventQueue.pop_front();
				y = eventQueue.front();
				eventQueue.pop_front();
				countMat[y][x] -= 1;
				countEvents--;

			} // end if

			  // classify.. cout the result. reset bins
			if (ROIEvents >= EVENTS_PER_CLASSIFICATION)
			{
				rectangle(disp_countMat, boundingBox, Scalar(0, 0, 0), 1, 8, 0);
				for (int i = 0; i < 25; i++) { // perform classification for all 25 candidates
					//histogram normalization
					double histTotal = 0;
					for (int j = 0; j < VOCABSIZE; j++) {
						histTotal += SWcandidate_hist[i][j];
					}
					for (int j = 0; j < VOCABSIZE; j++) {
						SWcandidate_hist[i][j] = SWcandidate_hist[i][j] / histTotal;
					}

					// homkermap
					VlHomogeneousKernelMap* hom;
					double psi[3], all_psi[VOCABSIZE * 3];
					hom = vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, 0.5, 1, -5, VlHomogeneousKernelMapWindowRectangular);
					//vl_homogeneouskernelmap_new(kernelType, gamma, order, period, windowType);
					for (int j = 0; j < VOCABSIZE; j++)
					{
						vl_homogeneouskernelmap_evaluate_d(hom, psi, 1, SWcandidate_hist[i][j]);
						all_psi[3 * j] = psi[0];
						all_psi[3 * j + 1] = psi[1];
						all_psi[3 * j + 2] = psi[2];
					}

					// do the classification
					double score = 0;
					//double temp[VOCABSIZE*3];
					double temp_sum = 0;
					int class_events;

					for (int j = 0; j < VOCABSIZE * 3; j++)
					{
						//temp[i] = SVMwt._matrix[j][i] * all_psi[i];
						temp_sum = temp_sum + model[j] * all_psi[j];
					}
					//score[j] = accumulate(temp, temp + VOCABSIZE * 3, 0) + SVMb._matrix[0][j];
					score = temp_sum + bias;
					temp_sum = 0;
					
					/*objcnt[class_events]++;
					probcnt[class_events] = objcnt[class_events] / rcgCnt;
					std::cout << objList[class_events] << "(" << objcnt[class_events] << "/" << rcgCnt << ")\t";
					for (int jj = 0; jj < CalNUM; jj++)
						std::cout << probcnt[jj] << "\t";
					std::cout << "\r";
					if (rcgCnt > reset_num)
					{
						rcgCnt = 0;
						for (int i = 0; i < CalNUM; i++)
						{
							objcnt[i] = 0;
							probcnt[i] = 0;
						}
						std::cout << std::endl << std::endl << std::endl
							<< "/////////Resetting the classification histogram/////////" << std::endl
							<< "/////////Resetting the classification histogram/////////" << std::endl
							<< std::endl << std::endl << std::endl;
						std::cout << "Class \t 0   1   2   3   4   5   6   7   8   9\n";

					}
					else
					{
						if ((probcnt[class_events] > prob_threshold) && (rcgCnt > refresh_hist))
						{
							std::cout << std::endl << std::endl
								<< "**********Move to the next @.@************ " << std::endl
								<< "**********Move to the next @.@************ " << std::endl
								<< "**********Move to the next @.@************ " << std::endl
								<< "**********Move to the next @.@************ " << std::endl
								<< "**********Move to the next @.@************ " << std::endl
								<< "**********Move to the next @.@************ " << std::endl
								<< std::endl << std::endl;
							std::cout << "Class \t 0   1   2   3   4   5   6   7   8   9\n";
							rcgCnt = 0;
							for (int i = 0; i < CalNUM; i++)
							{
								objcnt[i] = 0;
								probcnt[i] = 0;
							}

						}
					}*/

					if (score > globalBestScore) {
						globalBestScore = score;
						bestCandidate = i;
					}

					for (int j = 0; j < VOCABSIZE; j++)
						SWcandidate_hist[i][j] = 0;
					ROIEvents = 0;// this can make sure doing classification one time within EVENTS_PER_CLASSIFICATION.
				}

				origBB_topLeftX = origBB_topLeftX + (bestCandidate % 5) - 2;
				origBB_topLeftY = origBB_topLeftY + (bestCandidate / 5) - 2;
				if (origBB_topLeftX < 0)
					origBB_topLeftX = 0;
				if (origBB_topLeftY < 0)
					origBB_topLeftY = 0;
				if (origBB_topLeftY > cCOL)
					origBB_topLeftY = cCOL;
				if (origBB_topLeftX > cROW)
					origBB_topLeftX = cROW;
				padBB_topLeftX = origBB_topLeftX - 2;
				padBB_topLeftY = origBB_topLeftY - 2;
				globalBestScore = 0;

				cout << "Best candidate: " << bestCandidate << "\n";


				// Display Sliding Window
				//namedWindow("SW", CV_WINDOW_AUTOSIZE);
				disp_countMat = Mat(cROW, cCOL, CV_32FC1, &countMat);
				resize(disp_countMat, disp_countMat, Size(disp_countMat.cols / 2, disp_countMat.rows / 2));
				//rescale(disp_countMat, 0, 255);
				boundingBox = Rect(origBB_topLeftX, origBB_topLeftY, origBB_boxSizeX, origBB_boxSizeY);
				rectangle(disp_countMat, boundingBox, Scalar(255, 0, 0), 1, 8, 0);
				imshow("SW", disp_countMat);
				waitKey(25);
				


			} // end classification


		} // end while
	} // end else


	waitKey(0);
	destroyWindow("SW");
	vl_svm_delete(svm);
	delete[] all_psi;
	all_psi = NULL;
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

void rescale(Mat &countMat, int a, int b)
{
	double min, max;
	cv::minMaxIdx(countMat, &min, &max);
	//int m = (int)min;
	//int M = (int)max;
	for (int i = 0; i < countMat.rows; i++)
	{
		for (int j = 0; j < countMat.cols; j++)
		{
			countMat.at<float>(i, j) = ((b - a)*(countMat.at<float>(i, j) - min) / (max - min) + a);
			//countMat.at<uchar>(i, j) = ((b - a)*(countMat.at<uchar>(i, j) - min) / (max - min) + a);
		}
	}

}

