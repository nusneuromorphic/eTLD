#ifndef ETLDDESC_H
#define ETLDDESC_H

#include <utility>
#include <iostream>
#include <deque>
#include "ECparam.h"

#include <svm.h>
extern "C" {
#include "kdtree.h"
#include "homkermap.h"
#include "kmeans.h"
#include "generic.h"
}

class ETLDDesc{
    public:
        ETLDDesc(std::pair<int,int> dims, int vocab_size);
        ETLDDesc(std::pair<int,int> dims, int vocab_size, int kEC_NR, int kEC_NW, int kEC_RMIN, int kEC_RMAX, std::pair<int,int> numEvents);        ~ETLDDesc();
        void train(std::string initial_TD, int ROItopLeftX, int ROItopLeftY, int ROIboxSizeX, int ROIboxSizeY, bool verbose);
        void track(std::string initial_TD, bool verbose, bool show_window);
        void getDesctriptors_CountMat(std::vector<double> &desc, const int cur_loc_y, const int cur_loc_x, Matrix &t_ring, Matrix &t_wedge);
    private:
        const int kQueueSize_ {5000};
        const int kBootstrap_ {12000};
        const int kVocabSize_ {500};
        const int kPadding {2};

        const int kEC_NR_ {10};
        const int kEC_NW_ {12};
        const int kEC_RMIN_ {3};
        const int kEC_RMAX_ {30};
        ECparam ec_;

        const std::pair<int,int> dims_;
        const std::pair<int,int> numEvents_;

        double **countMat_;
        double **detMat_;
        int *ROIhist_;
        int *nonROIhist_;
        int *detection_list_;
        std::deque<int> eventQueue_;
        VlSvm * svm;
        VlKDForestSearcher* searcherObj_;
        VlKDForestNeighbor neighbors_[1];

        int ROItopLeftX_, ROItopLeftY_;
        int ROIboxSizeX_, ROIboxSizeY_;
        int origBB_boxSizeX_, origBB_boxSizeY_;
        int padBB_boxSizeX_, padBB_boxSizeY_;
        std::vector<double> allDescs_;
        std::vector<std::vector<double>> ROIDescs_;
        std::vector<std::vector<double>> nonROIDescs_;
        std::vector<std::vector<std::vector<bool>>> BBLookupTable_; // 3D Lookup Table of area = padded bounding box. Each pixel contains a 25x1 vector of which BBs they are in.
        std::vector<std::vector<double>> SWcandidate_hist_;

        void allocateMatrices();
        template<typename T> T** matrixAllocate(T **M);
};
#endif
