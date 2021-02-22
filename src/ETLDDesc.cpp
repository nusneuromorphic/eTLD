#include "ETLDDesc.h"
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <assert.h>
#include "currSpikeArr.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 

void rescale(cv::Mat &countMat, int a, int b);

ETLDDesc::ETLDDesc(std::pair<int,int> dims, int vocab_size)
    : dims_{dims}, kVocabSize_{vocab_size}
{
    allocateMatrices();
}

ETLDDesc::ETLDDesc(std::pair<int,int> dims, int vocab_size, int kEC_NR, int kEC_NW, int kEC_RMIN, int kEC_RMAX, std::pair<int,int> numEvents)
    : dims_{dims}, kVocabSize_{vocab_size}, kEC_NR_{kEC_NR}, kEC_NW_{kEC_NW}, kEC_RMIN_{kEC_RMIN}, kEC_RMAX_{kEC_RMAX}, numEvents_{numEvents}
{
    ECparam ec_temp(kEC_NR_, kEC_NW_, kEC_RMIN_, kEC_RMAX_, numEvents_.first, numEvents_.second);
    ec_ = ec_temp;

    allocateMatrices();
}

ETLDDesc::~ETLDDesc() {
    if (countMat_) {
        for (int i = 0; i<dims_.first; i++)
        {
            delete[] countMat_[i];
        }
        delete[] countMat_;
    }
    if (detMat_) {
        for (int i = 0; i<dims_.first; i++)
        {
            delete[] detMat_[i];
        }
        delete[] detMat_;
    }
    delete[] ROIhist_;
    delete[] nonROIhist_;
    delete[] detection_list_;
}

void ETLDDesc::allocateMatrices() {
    countMat_ = matrixAllocate(countMat_);
    detMat_ = matrixAllocate(detMat_);

    ROIhist_ = new int[kVocabSize_];
    nonROIhist_ = new int[kVocabSize_];
    detection_list_ = new int[kVocabSize_];
}

template<typename T> T** ETLDDesc::matrixAllocate(T **M)
{
    M = new T*[dims_.first];
    for (int i = 0; i < dims_.first; i++){
        M[i] = new T[dims_.second];
    }

    for (int i = 0; i < dims_.first; i++) {
        for (int j = 0; j < dims_.second; j++) {
            M[i][j] = 0;
        }
    }
    return M;
}

void ETLDDesc::train(std::string initial_TD, int ROItopLeftX, int ROItopLeftY, int ROIboxSizeX, int ROIboxSizeY, bool verbose)
{
    ROItopLeftX_ = ROItopLeftX;
    ROItopLeftY_ = ROItopLeftY;
    ROIboxSizeX_ = ROIboxSizeX;
    ROIboxSizeY_ = ROIboxSizeY;

    Matrix t_wedge(kEC_RMAX_ * 2 + 1, kEC_RMAX_ * 2 + 1);
    Matrix t_ring(kEC_RMAX_ * 2 + 1, kEC_RMAX_ * 2 + 1);
    ec_.wedgeRing_lookupTable(t_ring, t_wedge);

    std::ifstream infile(initial_TD);
    int countEvents = 0;
    int descriptor_count = 0;
    if (!infile) {
        std::cerr << "Oops, unable to open .txt..." << std::endl;
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
            eventQueue_.push_back(x);
            eventQueue_.push_back(y);
            countMat_[y][x] += 1.0;

            if (countEvents > ec_.minNumEvents)  {
                std::vector<double> desc;
                getDesctriptors_CountMat(desc, y, x, t_ring, t_wedge);
                allDescs_.insert(allDescs_.end(), desc.begin(), desc.end());
                if ((x >= ROItopLeftX) && (x <= ROItopLeftX + ROIboxSizeX) && (y >= ROItopLeftY) && (y <= ROItopLeftY + ROIboxSizeY)) {
                    ROIDescs_.push_back(desc);
                }
                else {
                    nonROIDescs_.push_back(desc);
                }

                descriptor_count++;
            }
            if (eventQueue_.size() > (kQueueSize_ * 2)) // Pop out the oldest event in the queue.
            {
                x = eventQueue_.front();
                eventQueue_.pop_front();
                y = eventQueue_.front();
                eventQueue_.pop_front();
                countMat_[y][x] -= 1.0;
                countEvents--;
            }
        }
    }
    infile.close();

    int all_desc_size = allDescs_.size();
    double* allD = new double[all_desc_size];
    allD = &allDescs_[0];

    VlKMeans * kmeans = vl_kmeans_new(VL_TYPE_DOUBLE, VlDistanceL2);
    kmeans->verbosity = verbose;
    vl_kmeans_set_algorithm(kmeans, VlKMeansANN);
    vl_kmeans_init_centers_with_rand_data(kmeans, allD, kEC_NR_ * kEC_NW_, descriptor_count, kVocabSize_);
    vl_kmeans_set_max_num_iterations(kmeans, 100);
    vl_kmeans_refine_centers(kmeans, allD, descriptor_count);

    void const * vocab = vl_kmeans_get_centers(kmeans);
    double * vocab2 = new double[kVocabSize_ * kEC_NR_ * kEC_NW_];
    vocab2 = (double*)vocab;

    // build a tree.
    VlKDForest* forest = vl_kdforest_new(VL_TYPE_DOUBLE, kEC_NR_ * kEC_NW_, 1, VlDistanceL2);
    vl_kdforest_build(forest, kVocabSize_, vocab2);
    vl_kdforest_set_max_num_comparisons(forest, 15);

    // initialise detection list
    for (int i = 0; i < kVocabSize_; i++) {
        detection_list_[i] = 0;
    }

    // build ROI histogram and nonROI histogram
    searcherObj_ = vl_kdforest_new_searcher(forest);

    for (int i = 0; i < ROIDescs_.size(); i++) {
        vl_kdforestsearcher_query(searcherObj_, neighbors_, 1, &ROIDescs_[i].front());
        //vl_kdforest_query(forest, neighbors, 1, pass_desc);
        int binsa_new = neighbors_->index;
        ROIhist_[binsa_new]++;
        detection_list_[binsa_new]++; // positive value means ROI cluster
        //cout << "ROI: " << binsa_new << "+1.\n";
    }

    for (int i = 0; i < nonROIDescs_.size(); i++) {
        vl_kdforestsearcher_query(searcherObj_, neighbors_, 1, &nonROIDescs_[i].front());
        //vl_kdforest_query(forest, neighbors, 1, pass_desc);
        int binsa_new = neighbors_->index;
        nonROIhist_[binsa_new]++;
        detection_list_[binsa_new]--; // negative value means non-ROI cluster
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

    for (int i = 0; i < kBootstrap_; i++) {
        for (int j = 0; j < kVocabSize_; j++) {
            nextROIrandom = rand() % (ROIhist_[j] + 1);
            ROItotal = ROItotal + nextROIrandom;
            nextROISample.push_back(nextROIrandom);

            nextNonROIrandom = rand() % (nonROIhist_[j] + 1);
            nonROItotal = nonROItotal + nextNonROIrandom;
            nextNonROISample.push_back(nextNonROIrandom);
        }

        //  histogram normalization
        for (int j = 0; j < kVocabSize_; j++) {
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

    double labels[kBootstrap_ * 2];
    for (int i = 0; i < kBootstrap_; i++) {
        labels[i] = 1;
    }
    for (int i = kBootstrap_; i < kBootstrap_ * 2; i++) {
        labels[i] = 2;
    }

    double lambda = 1.0 / (10 * kBootstrap_ * 2);	// lambda = 1 / (svmOpts.C * length(train_label)) ; -> From the matlab code
    // 0.0000125

    svm = vl_svm_new(VlSvmSolverSgd, SVMdata, kVocabSize_, kBootstrap_ * 2, labels, lambda);
    // 9.3266e-06, C = 10, length(train_label) = 4000.
    vl_svm_train(svm);

    const double * model = vl_svm_get_model(svm);
    double bias = vl_svm_get_bias(svm);

    if (verbose)
        cout << "Model w = [ " << model[0] << " " << model[1] << " ], bias b = " << bias << "\n";

    // padding
    origBB_boxSizeX_ = ROIboxSizeX; // original bounding box
    origBB_boxSizeY_ = ROIboxSizeY;
    padBB_boxSizeX_ = origBB_boxSizeX_ + (kPadding * 2); // padded bounding box
    padBB_boxSizeY_ = origBB_boxSizeY_ + (kPadding * 2);

    // Sliding Window Lookup Table: BBLookupTable[HEIGHT][WIDTH][25x1]
    //vector<vector<vector<bool>>> BBLookupTable; // 3D Lookup Table of area = padded bounding box. Each pixel contains a 25x1 vector of which BBs they are in.
    BBLookupTable_.resize(padBB_boxSizeY_);
    for (int i = 0; i < padBB_boxSizeY_; i++) {
        BBLookupTable_[i].resize(padBB_boxSizeX_);
        for (int j = 0; j < padBB_boxSizeX_; j++) {
            BBLookupTable_[i][j].resize(25);
        }
    }

    for (int i = 0; i < padBB_boxSizeY_; i++) {
        for (int j = 0; j < padBB_boxSizeX_; j++) {
            for (int k = 0; k < 5; k++) { // moving down
                for (int l = 0; l < 5; l++) { // moving right
                    if ((i <= k + origBB_boxSizeY_ - 1) && (j <= l + origBB_boxSizeX_ - 1) && (i >= k) && (j >= l))
                        BBLookupTable_[i][j][(k * 5) + l] = true;
                    else
                        BBLookupTable_[i][j][(k * 5) + l] = false;
                }
            }
        }
    }


    // Generating 25 histograms for 25 sliding window candidates
    SWcandidate_hist_.resize(25);
    for (int i = 0; i < 25; i++) {
        SWcandidate_hist_[i].resize(kVocabSize_);
        for (int j = 0; j < kVocabSize_; j++) {
            SWcandidate_hist_[i][j] = 0;
        }
    }
}

void ETLDDesc::getDesctriptors_CountMat(std::vector<double> &desc, const int cur_loc_y, const int cur_loc_x, Matrix &t_ring, Matrix &t_wedge)
{
    // get current spike vectors from "Count_Mat".
    currSpikeArr currSpikeArr;

    //get ring and wedge num from lookup table*********************
    int dy = cur_loc_y - kEC_RMAX_;
    int dx = cur_loc_x - kEC_RMAX_;

    int jmin = (cur_loc_y - ec_.rmax >= 0) ? (cur_loc_y - ec_.rmax) : 0;
    int jmax = (cur_loc_y + ec_.rmax < dims_.first) ? (cur_loc_y + ec_.rmax) : dims_.first;
    int imin = (cur_loc_x - ec_.rmax >= 0) ? (cur_loc_x - ec_.rmax) : 0;
    int imax = (cur_loc_x + ec_.rmax < dims_.second) ? (cur_loc_x + ec_.rmax) : dims_.second;
    for (int j = jmin; j < jmax; j++)
    {
        for (int i = imin; i < imax; i++)
        {
            if (countMat_[j][i]>0)
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
    Matrix logHistEmpty(ec_.nw, ec_.nr);
    for (int n = 0; n < currSpikeArr.x.size(); n++)
    {
        int log_y = currSpikeArr.wedgeNum[n];
        int log_x = currSpikeArr.ringNum[n];
        if ((log_y > 0) && (log_x > 0))//exclude those spikes whos ringNum and wedgeNum are 0.
        {
            scount++;
            assert(log_y < dims_.first);
            assert(log_x < dims_.second);
            int loc_count = countMat_[currSpikeArr.y[n] - 1][currSpikeArr.x[n] - 1];
            logHistEmpty._matrix[log_y - 1][log_x - 1] += loc_count;//wedgeNum and ringNum begin from 1. matrix subscriptions begin from 0;
        }
    }
    // calculate desc
    double sum_logHist = scount;
    if (sum_logHist)
    {
        for (int ix = 0; ix < ec_.nr; ix++)
        {
            for (int iy = 0; iy < ec_.nw; iy++)
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
}

void ETLDDesc::track(std::string later_TD, bool verbose, bool show_window)
{
    if (verbose)
        cout << "Performing tracking.\n";

    const int EVENTS_PER_CLASSIFICATION = ROIboxSizeX_ * ROIboxSizeY_ * 0.40;
    eventQueue_.clear();
    deque<int> detQueue;
    int x, y;
    double ts, read_x, read_y, read_p;
    int countEvents = 0;
    int ROIEvents = 0;
    double globalBestScore = 0, globalAverageScore = 0, localAverageScore = 0;
    double allScores[25];
    int bestCandidate = 0, classificationCount = 0;
    cv::Mat disp_countMat = cv::Mat::zeros(dims_.first, dims_.second, CV_8UC1); //cROWxcCOL zero matrix
    cv::Mat disp_detMat = cv::Mat::zeros(dims_.first, dims_.second, CV_8UC1); //cROWxcCOL zero matrix 
    cv::Rect boundingBox;

    for (int i = 0; i < dims_.first; i++) {
        for (int j = 0; j < dims_.second; j++) {
            countMat_[i][j] = 0;
        }
    }

    Matrix t_wedge(kEC_RMAX_ * 2 + 1, kEC_RMAX_ * 2 + 1);
    Matrix t_ring(kEC_RMAX_ * 2 + 1, kEC_RMAX_ * 2 + 1);
    ec_.wedgeRing_lookupTable(t_ring, t_wedge);

    int origBB_topLeftX = ROItopLeftX_;
    int origBB_topLeftY = ROItopLeftY_;
    int padBB_topLeftX = origBB_topLeftX - kPadding;
    int padBB_topLeftY = origBB_topLeftY - kPadding;
    tracks_out.clear();

    const double * model = vl_svm_get_model(svm);
    double bias = vl_svm_get_bias(svm);

    const int tEND {15};
    const int kDetQueueSize {1000};
    ifstream trackfile(later_TD);
    if (!trackfile) {
        cerr << "Oops, unable to open .txt..." << endl;
    }
    else {
        long AVGnumevent = 0;
        double AVGsumevent = 0;
        while ((trackfile >> ts) && (ts < tEND)) {
            trackfile >> read_x;
            x = (int)read_x;
            trackfile >> read_y;
            y = (int)read_y;
            trackfile >> read_p;
            countEvents++;
            eventQueue_.push_back(x);
            eventQueue_.push_back(y);
            countMat_[y][x] += 1.0;

            if (countEvents > ec_.minNumEvents) {
                AVGnumevent++;
                vector<double> desc;
                getDesctriptors_CountMat(desc, y, x, t_ring, t_wedge);

                vl_kdforestsearcher_query(searcherObj_, neighbors_, 1, &desc.front());
                int binsa_new = neighbors_->index;
                if (detection_list_[binsa_new] > 0) { // if descriptor belongs to ROI cluster
                    detQueue.push_back(x);
                    detQueue.push_back(y);
                    detMat_[y][x] += 1.0; // update detection matrix
                }

                if ((x >= padBB_topLeftX) && (x < padBB_topLeftX + padBB_boxSizeX_) && (y >= padBB_topLeftY) && (y < padBB_topLeftY + padBB_boxSizeY_)) {
                    for (int i = 0; i < 25; i++) {
                        if (BBLookupTable_[y - padBB_topLeftY][x - padBB_topLeftX][i] == true) {
                            SWcandidate_hist_[i][binsa_new]++; // increment the histogram of the corresponding SW candidate
                        }
                    }
                    ROIEvents++;
                }
            }

            if (eventQueue_.size() > (kQueueSize_ * 2)) // Pop out the oldest event in the queue.
            {
                x = eventQueue_.front();
                eventQueue_.pop_front();
                y = eventQueue_.front();
                eventQueue_.pop_front();
                countMat_[y][x] -= 1.0;
                countEvents--;

            } // end if

            if (detQueue.size() > (kDetQueueSize * 2)) // Pop out the oldest event in the detection queue.
            {
                x = detQueue.front();
                detQueue.pop_front();
                y = detQueue.front();
                detQueue.pop_front();
                detMat_[y][x] -= 1.0;

            } // end if

            // classify.. cout the result. reset bins
            if (ROIEvents >= EVENTS_PER_CLASSIFICATION)
            {
                cv::rectangle(disp_countMat, boundingBox, cv::Scalar(0, 0, 0), 1, 8, 0);
                double temp[kVocabSize_];
                for (int i = 0; i < 25; i++) { // perform classification for all 25 candidates
                    //histogram normalization
                    double histTotal = 0;
                    for (int j = 0; j < kVocabSize_; j++) {
                        histTotal += SWcandidate_hist_[i][j];
                    }
                    for (int j = 0; j < kVocabSize_; j++) {
                        SWcandidate_hist_[i][j] = SWcandidate_hist_[i][j] / histTotal;
                        temp[j] = SWcandidate_hist_[i][j];
                    }

                    // do the classification
                    double score = 0;
                    double temp_sum = 0;
                    int class_events;

                    for (int j = 0; j < kVocabSize_; j++)
                    {
                        temp_sum = temp_sum + model[j] * temp[j];
                    }
                    score = temp_sum + bias;
                    temp_sum = 0;
                    allScores[i] = score;

                    if (score == globalBestScore) {
                        if ((abs((bestCandidate % 5) - 2) + abs((bestCandidate / 5) - 2)) > (abs((i % 5) - 2) + abs((i / 5) - 2))) // if current candidate is closer to center
                            globalBestScore = score;
                        bestCandidate = i;
                    }

                    if (score > globalBestScore) {
                        globalBestScore = score;
                        bestCandidate = i;
                    }

                    for (int j = 0; j < kVocabSize_; j++)
                        SWcandidate_hist_[i][j] = 0;
                    ROIEvents = 0;// this can make sure doing classification one time within EVENTS_PER_CLASSIFICATION.
                } // end for

                // Checking if scores are all similar
                if (globalBestScore < 0.97 * globalAverageScore) { // perform detection

                    double totalDetEvents = 0, highestDetEvents = 0;
                    int detX = -1, detY = -1;

                    for (int i = 0; i < dims_.second - ROIboxSizeX_; i++) {
                        for (int j = 0; j < dims_.first - ROIboxSizeY_; j++) {
                            for (int k = i; k < i + ROIboxSizeX_ - 1; k++) {
                                for (int l = j; l < j + ROIboxSizeY_ - 1; l++) {
                                    totalDetEvents += detMat_[l][k];
                                }
                            }
                            if (highestDetEvents < totalDetEvents) {
                                detX = i;
                                detY = j;
                                highestDetEvents = totalDetEvents;
                            }
                            totalDetEvents = 0;
                        }
                    }

                    if (detX == -1 || detY == -1)
                        cout << "Object not detected.\n";
                    else {
                        origBB_topLeftX = detX;
                        origBB_topLeftY = detY;
                        padBB_topLeftX = origBB_topLeftX - 2;
                        padBB_topLeftY = origBB_topLeftY - 2;
                        if (verbose)
                            cout << "Object detected at " << detX << ", " << detY << "\n";
                        globalBestScore = globalAverageScore;
                    }
                }

                else {

                    globalAverageScore = ((globalAverageScore * (classificationCount)) + globalBestScore) / (classificationCount + 1.0);

                    // Moving the bounding box
                    origBB_topLeftX = origBB_topLeftX + (bestCandidate % 5) - 2;
                    origBB_topLeftY = origBB_topLeftY + (bestCandidate / 5) - 2;
                    if (origBB_topLeftX < 0)
                        origBB_topLeftX = 0;
                    if (origBB_topLeftY < 0)
                        origBB_topLeftY = 0;
                    if (origBB_topLeftY > dims_.second)
                        origBB_topLeftY = dims_.second;
                    if (origBB_topLeftX > dims_.first)
                        origBB_topLeftX = dims_.first;
                    padBB_topLeftX = origBB_topLeftX - 2;
                    padBB_topLeftY = origBB_topLeftY - 2;
                    globalBestScore = 0;
                    localAverageScore = 0;

                    if (verbose)
                        cout << "Best candidate: " << bestCandidate << "\n";
                    classificationCount++;

                }
                std::tuple<int,int,int,int> boundingBoxDims (origBB_topLeftX, origBB_topLeftY, origBB_boxSizeX_, origBB_boxSizeY_);
                std::tuple<double,std::tuple<int,int,int,int >> current_track (ts, boundingBoxDims);
                tracks_out.push_back(current_track);

                if (show_window) {
                    // Display Sliding Window
                    for (int i = 0; i < dims_.first; i++) {
                        for (int j = 0; j < dims_.second; j++) {
                            disp_countMat.at<uchar>(i, j) = countMat_[i][j];
                            disp_detMat.at<uchar>(i, j) = detMat_[i][j];
                        }
                    }

                    rescale(disp_countMat, 0, 255);
                    boundingBox = cv::Rect(origBB_topLeftX, origBB_topLeftY, origBB_boxSizeX_, origBB_boxSizeY_);
                    rectangle(disp_countMat, boundingBox, cv::Scalar(255, 0, 0), 1, 8, 0);
                    cv::imshow("TD", disp_countMat);

                    rescale(disp_detMat, 0, 255);
                    rectangle(disp_detMat, boundingBox, cv::Scalar(255, 0, 0), 1, 8, 0);
                    cv::imshow("Detector", disp_detMat);

                    cv::waitKey(1);
                }

            } // end classification
        } // end while
    } // end else

    trackfile.close();
    cv::waitKey(0);
    if (show_window)
        cv::destroyWindow("SW");
    vl_svm_delete(svm);
}

void rescale(cv::Mat &countMat, int a, int b)
{
    double min, max;
    cv::minMaxIdx(countMat, &min, &max);
    for (int i = 0; i < countMat.rows; i++)
    {
        for (int j = 0; j < countMat.cols; j++)
        {
            countMat.at<uchar>(i, j) = ((b - a)*(countMat.at<uchar>(i, j) - min) / (max - min) + a);
        }
    }
}
