/********************************************
Project: Real-time Recognition
Auther: YANG HONG
Date: 20/07/2017
Verified Date: 06/09/2017; 28/09/2017
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
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
extern "C" {
#include "vl/kdtree.h"
#include "vl/homkermap.h"
}

using namespace std;
using namespace cv;

#define VOCABSIZE 3000
//#define cROW 240 //ATIS
//#define cCOL 304 //ATIS
#define cROW 180 // DAVIS
#define cCOL 240 // DAVIS
#define tEND 10 
#define CalNUM 4
#define LookUpCenter 10
#define EC_NR 7
#define EC_NW 12
#define ASSOC_SIZE 14524800
#define DET_THRESHOLD 0.8
#define OBJ_DET_NUM 2 //thumper
#define Thump_Clusters 66

void createCountMat(VlKDForest* forest, ECparam &ec, const string &str, Matrix &SVMwt, Matrix &SVMb,
	vector<int> &det_init, Mat &det_countMat);
void getDesctriptors_CountMat(vector<double> &desc, double countMat[cROW][cCOL], ECparam &ec,
	const int cur_loc_y, const int cur_loc_x, Matrix &t_ring, Matrix &t_wedge);
inline void readNumToMat(Matrix &mat, string str);
void detector_initialization(vector<int> &det_init);
inline void rescale(Mat &det_countMat, int a, int b);


int main()
{
	ECparam ec(7, 12, 2, 10, 200, 5000);
	double* vocab = new double[EC_NR * EC_NW * VOCABSIZE];
	string str_v = "../modelTD4cl_RBgen_DEMO_forDET_3000500010.txt";
	ifstream infile(str_v);
	if (!infile)
	{
		cerr << "Oops, unable to open .txt..." << endl;
	}
	else
	{
		for (int iy = 0; iy < VOCABSIZE; iy++)
		{
			for (int ix = 0; ix < EC_NR * EC_NW; ix++)
			{
				infile >> vocab[(iy*EC_NR * EC_NW) + ix];
				//std::cout << vocab[(iy * EC_NR * EC_NW) + ix] << '\t';
			}
		}
		infile.close();
	}
	/*cout << endl
	<<"vocab[1][0](should be 0.0039): " << vocab[EC_NR * EC_NW *0 + 1] << '\n'
	<< "vocab[56][2223](should be 0.0213): " << vocab[EC_NR * EC_NW * 2223 + 56] << '\n'
	<< "vocab[80][2822](should be 0.0012): " << vocab[EC_NR * EC_NW * 2822 + 80] << endl;*/

	// build a tree.
	VlKDForest* forest = vl_kdforest_new(VL_TYPE_DOUBLE, EC_NR * EC_NW, 1, VlDistanceL2);
	vl_kdforest_build(forest, VOCABSIZE, vocab);
	vl_kdforest_set_max_num_comparisons(forest, 15);

	
	vector<int> det_init;
	int det_arr[Thump_Clusters];
	string str_detThump = "../det_init08_new.txt";
	try 
	{
		ifstream infile(str_detThump);
		if (!infile)
		{
			throw "Cannot-read-detector-initialization-file";
		}
		else
		{
			for (int i = 0; i < Thump_Clusters; i++)
			{
				infile >> det_arr[i];
				std::cout << det_arr[i] << '\t';
			}
			infile.close();
			det_init = { det_arr, det_arr + Thump_Clusters };
		}
	}
	catch(char *str)
	{
		std::cout << str << std::endl;
		detector_initialization(det_init);
	}
	std::cout << std::endl;
	// create the detection count matrix using opencv
	Mat det_countMat = Mat::zeros(cROW, cCOL, CV_8UC1); //cROWxcCOL zero matrix  
	//std::cout << "det_countMat=" << std::endl << " " << det_countMat << std::endl << std::endl;

	// load svm
	Matrix SVMwt(CalNUM, VOCABSIZE * 3), SVMb(1, CalNUM);
	string str_w = "../svvmodel_DEMO_forDET_wt.txt";
	readNumToMat(SVMwt, str_w);
	/*cout << "SVMwt[2][6](should be 0.0134): " << SVMwt._matrix[2][6] << '\n'
	<< "SVMwt[0][10](should be -0.042): " << SVMwt._matrix[0][10] << '\n'
	<< "SVMwt[0][6819](should be 0.0232): " << SVMwt._matrix[0][6819] << endl;*/
	string str_b = "../svvmodel_DEMO_forDET_b.txt";
	readNumToMat(SVMb, str_b);

	string str_e = "../thumper_h1.txt";
	createCountMat(forest, ec, str_e, SVMwt, SVMb, det_init, det_countMat);
	vl_kdforest_delete(forest);

	

	delete[] vocab;
	vocab = NULL;
	system("pause");
	return 0;
}


void createCountMat(VlKDForest* forest, ECparam &ec, const string &str_e, Matrix &SVMwt, Matrix &SVMb,
	vector<int> &det_init, Mat &det_countMat)
{
	const int EVENTS_PER_CLASSIFICATION = 10000;
	const int REFRACTORY_COUNT = 3;
	const double prob_threshold = 0.80;
	const int refresh_hist = 20;
	const int reset_num = 50;
	const int min_heat_map_count = 3000;

	vector<string> objList = { "Background","Obstacle", "Thumper", "UAV" };
	Matrix t_wedge(LookUpCenter * 2 + 1, LookUpCenter * 2 + 1);
	Matrix t_ring(LookUpCenter * 2 + 1, LookUpCenter * 2 + 1);
	ec.wedgeRing_lookupTable(t_ring, t_wedge);

	int rcgCnt = 0;
	double objcnt[CalNUM];
	double probcnt[CalNUM];
	double countMat[cROW][cCOL];
	static int gcount = 0; // global event count
	static int countEvents = 0;
	static int descriptor_count = 0;
	static int heat_map_count = 0;
	double hist[VOCABSIZE];

	for (int i = 0; i<CalNUM; i++)
	{
		objcnt[i] = 0;
		probcnt[i] = 0;
	}
	for (int i = 0; i < cROW; i++)
	{
		for (int j = 0; j < cCOL; j++)
		{
			countMat[i][j] = 0;
		}
	}
	for (int i = 0; i < VOCABSIZE; i++)
	{
		hist[i] = 0;
	}

	// read events.txt file.
	ifstream infile(str_e);
	if (!infile)
	{
		cerr << "Oops, unable to open .txt..." << endl;
	}
	int x, y;
	double ts;
	double read_x, read_y, read_p;
	VlKDForestNeighbor neighbors[1]; // a structure
	VlKDForestSearcher* searcherObj = vl_kdforest_new_searcher(forest);
	// Accord 2000 events into countMat matrix
	//std::cout << "Class \t\t BG \t OBS \t THP \t UAV \n";
	while ((infile >> ts) && (ts<tEND))
	{
		infile >> read_x;
		infile >> read_y;
		infile >> read_p;
		x = (int)read_x;
		y = (int)read_y;
		countEvents++;
		gcount++;
		//cout << "gcount:  " << gcount << '\t' << "ts: " << ts << endl;
		countMat[y][x] += 1;
		if ((countEvents > ec.minNumEvents) && (countEvents <= ec.maxNumEvents) && (countMat[y][x] <= REFRACTORY_COUNT))
		{
			vector<double> desc;
			getDesctriptors_CountMat(desc, countMat, ec, y, x, t_ring, t_wedge);

			vl_kdforestsearcher_query(searcherObj, neighbors, 1, &desc.front());
			//vl_kdforest_query(forest, neighbors, 1, pass_desc);
			int binsa_new = neighbors->index;
			hist[binsa_new]++;
			descriptor_count++;
			// accumulate detector heat map
			vector<int>::iterator iter;
			iter = find(det_init.begin(), det_init.end(), binsa_new);
			if (iter != det_init.end())
			{

				int dot = det_countMat.at<uchar>(y, x);  // at<uchar>
				det_countMat.at<uchar>(y, x) = ++dot;
				//det_countMat.at<uchar>(y, x)++;
				heat_map_count++;
				if (heat_map_count >= min_heat_map_count)
				{
					rescale(det_countMat, 0, 255);
					/*CvSize size = cvSize(det_countMat.col, det_countMat.row);
					IplImage* image2 = cvCreateImage(size, 8, 3);
					IplImage ipltemp = det_countMat;
					cvCopy(&ipltemp, image2);
					//image2->imageData = (char*)ipltemp;
					cvNamedWindow("thumper", 1);
					cvShowImage("thumper", image2);
					waitKey(6000);
					cvDestroyWindow("thumper");
					cvReleaseImage(&image2);*/
					namedWindow("Thumper", CV_WINDOW_AUTOSIZE);
					imshow("Thumper", det_countMat);
					waitKey(0);
					destroyWindow("Thumper");
				}
			}
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

		}

		// classify.. cout the result. reset bins
		if (gcount >= EVENTS_PER_CLASSIFICATION)
		{
			rcgCnt++;
			//histogram normalization
			for (int j = 0; j < VOCABSIZE; j++)
			{
				hist[j] /= descriptor_count;
			}
			// homkermap
			VlHomogeneousKernelMap* hom;
			double psi[3], all_psi[VOCABSIZE * 3];
			hom = vl_homogeneouskernelmap_new(VlHomogeneousKernelChi2, 0.5, 1, -5, VlHomogeneousKernelMapWindowRectangular);
			//vl_homogeneouskernelmap_new(kernelType, gamma, order, period, windowType);
			for (int j = 0; j < VOCABSIZE; j++)
			{
				vl_homogeneouskernelmap_evaluate_d(hom, psi, 1, hist[j]);
				all_psi[3 * j] = psi[0];
				all_psi[3 * j + 1] = psi[1];
				all_psi[3 * j + 2] = psi[2];
			}

			// do the classification
			double score[CalNUM];
			double max_score = 0;
			//double temp[VOCABSIZE*3];
			double temp_sum = 0;
			int class_events;
			for (int j = 0; j < CalNUM; j++)
			{
				for (int i = 0; i < VOCABSIZE * 3; i++)
				{
					//temp[i] = SVMwt._matrix[j][i] * all_psi[i];
					temp_sum = temp_sum + SVMwt._matrix[j][i] * all_psi[i];
				}
				//score[j] = accumulate(temp, temp + VOCABSIZE * 3, 0) + SVMb._matrix[0][j];
				score[j] = temp_sum + SVMb._matrix[0][j];
				temp_sum = 0;
				if (j == 0)
				{
					max_score = score[j];
					class_events = j;
				}
				else
					if (score[j] > max_score)
					{
						max_score = score[j];
						class_events = j;
					}
			}
			objcnt[class_events]++;
			for (int jj = 0; jj < CalNUM; jj++)
				probcnt[jj] = objcnt[jj] / rcgCnt;
			//probcnt[class_events] = objcnt[class_events] / rcgCnt;
			std::cout << objList[class_events] << "(" << objcnt[class_events] << "/" << rcgCnt << ")\t";
			for (int jj = 0; jj < CalNUM; jj++)
				std::cout << probcnt[jj] << "\t";
			std::cout << "\r";
			if (rcgCnt > reset_num)
			{
				rcgCnt = 0;
				for (int i = 0; i<CalNUM; i++)
				{
					objcnt[i] = 0;
					probcnt[i] = 0;
				}
				std::cout << std::endl << std::endl << std::endl
					<< "/////////Resetting the classification histogram/////////" << std::endl
					<< "/////////Resetting the classification histogram/////////" << std::endl
					<< std::endl << std::endl << std::endl;
				std::cout << "Class \t\t BG \t OBS \t THP \t UAV \n";

			}
			else
			{
				if ((probcnt[class_events]>prob_threshold) && (rcgCnt>refresh_hist))
				{
					std::cout << std::endl << std::endl
						<< "**********Move to the next @.@************ " << std::endl
						<< "**********Move to the next @.@************ " << std::endl
						<< "**********Move to the next @.@************ " << std::endl
						<< "**********Move to the next @.@************ " << std::endl
						<< "**********Move to the next @.@************ " << std::endl
						<< "**********Move to the next @.@************ " << std::endl
						<< std::endl << std::endl;
					std::cout << "Class \t\t BG \t OBS \t THP \t UAV \n";
					rcgCnt = 0;
					for (int i = 0; i<CalNUM; i++)
					{
						objcnt[i] = 0;
						probcnt[i] = 0;
					}

				}
			}
			for (int i = 0; i < VOCABSIZE; i++)
				hist[i] = 0;
			descriptor_count = 0;
			gcount = 0;// this can make sure doing classification one time within EVENTS_PER_CLASSIFICATION.

		}//end of classification

	}
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


void readNumToMat(Matrix &mat, string str)
{
	ifstream infile(str);
	if (!infile)
	{
		cerr << "Oops, unable to open .txt (readNumToMat)..." << endl;
	}
	else
	{
		for (int iy = 0; iy < mat._row; iy++)
		{
			for (int ix = 0; ix < mat._col; ix++)
			{
				infile >> mat._matrix[iy][ix];
				//cout << mat[iy][ix] << '\t';
			}
		}
		infile.close();
	}
}


void detector_initialization(vector<int> &det_init)
{
	double* cluster_tabulation = new double[VOCABSIZE];
	for (int i = 0; i < VOCABSIZE; i++)
	{
		cluster_tabulation[i] = 0;
	}
	double per_class[CalNUM];
	for (int j = 0; j < CalNUM; j++)
	{
		per_class[j] = 0;
	}
	Matrix assoc(1, ASSOC_SIZE), loctrain_label(1, ASSOC_SIZE);
	string str_assoc = "../assoc.txt";
	readNumToMat(assoc, str_assoc);
	//std::cout << assoc._matrix[0][5] << std::endl;
	string str_loctrain_label = "../loctrain_label.txt"; //I subtracted every elem by 1.
	readNumToMat(loctrain_label, str_loctrain_label);
	//std::cout << loctrain_label._matrix[0][3026000] << std::endl;
	for (int i = 0; i < ASSOC_SIZE; i++)
	{
		cluster_tabulation[int(assoc._matrix[0][i] - 1)]++;
	}

	for (int j = 0; j < VOCABSIZE; j++)
	{
		for (int i = 0; i < ASSOC_SIZE; i++)
		{
			if (assoc._matrix[0][i] == (j + 1))
			{
				per_class[int(loctrain_label._matrix[0][i])]++;
			}
		}
		for (int k = 0; k < CalNUM; k++)
		{
			per_class[k] = per_class[k] / cluster_tabulation[j];
		}
		if (per_class[OBJ_DET_NUM] > DET_THRESHOLD) // class 3 is thumper
		{
			//det_init.push_back(j + 1); // j+1 is the cluster number in MATLAB
			det_init.push_back(j);
		}
	}
	


}

void rescale(Mat &det_countMat, int a, int b)
{
	double min, max;
	cv::minMaxIdx(det_countMat, &min, &max);
	//int m = (int)min;
	//int M = (int)max;
	for (int i = 0; i < det_countMat.rows; i++)
	{
		for (int j = 0; j < det_countMat.cols; j++)
		{
			det_countMat.at<uchar>(i, j) = ((b - a)*(det_countMat.at<uchar>(i, j) - min) / (max - min) + a);
		}
	}
	
}
