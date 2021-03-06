#include <iostream>
#include "ETLDDesc.h"
#include <tuple>

int main () {
    // for DAVIS240c
    int cROW {180}, cCOL {240};
    std::pair<int,int> dims (cROW, cCOL);

    int minNumEvents {100}, maxNumEvents {1000};
    std::pair<int,int> numEvents (minNumEvents, maxNumEvents);

    int vocab_size {500};
    int EC_RMIN {3}, EC_RMAX {30};
    int EC_NR {10}, EC_NW {12};
    ETLDDesc eTLDdesc(dims, vocab_size, EC_NR, EC_NW, EC_RMIN, EC_RMAX, numEvents);

    std::string initial_TD = "../sample_td/monitor_initial.txt";

    // Location of object for training (ROI)
    int ROItopLeftX {44}, ROItopLeftY {15};
    int ROIboxSizeX {55}, ROIboxSizeY {37};
    eTLDdesc.train(initial_TD, ROItopLeftX, ROItopLeftY, ROIboxSizeX, ROIboxSizeY, false);

    std::string test_TD = "../sample_td/monitor_later.txt";
    eTLDdesc.track(test_TD, false, true);

    for (auto & track_it : eTLDdesc.tracks_out) {
        std::tuple<int,int,int,int> boundingBox;
        double ts;
        std::tie (ts, boundingBox) = track_it;
        std::cout << "Found Bounding Box with top Left coordinates: ("
            << std::get<0>(boundingBox) << "," << std::get<1>(boundingBox)
            << ") and width= " << std::get<2>(boundingBox)
            << ", height= " << std::get<3>(boundingBox)
            << " at Ts= " << ts << '\n';
    }

    std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Done!\n";
    return 0;
}
