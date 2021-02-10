# e-TLD
Event-based Framework for Dynamic Object Tracking

<p align="center">
  <a href="https://tinyurl.com/ske6nk7">
    <img src="https://i.ibb.co/Y2ySryz/DETECT-e-TLD-play.png" alt="eTLD Video Demo" width="600"/>
  </a>
</p>

The final paper is available [here](https://ieeexplore.ieee.org/document/9292994).
The arXiv preprint is available [here](https://arxiv.org/abs/2009.00855).

# Dependencies
- OpenCV >= 3.0
- VLFeat 0.9.21

# How to run the example?
- Edit `CMakeLists.txt` to set the correct path to VLFeat root.
- Build.
```
  $ cd eTLD
  $ mkdir build && cd build
  $ cmake ..
  $ make
```
- Run.
```
  $ ./ETLDDesc
```
# How to use?
- Include source and include files into your project.
- Create an ETLDDesc object.
```
ETLDDesc eTLDdesc(sensor_dims, vocab_size);
```
- Train from event data and ROI.
```
eTLDdesc.train(initial_TD, ROItopLeftX, ROItopLeftY, ROIboxSizeX, ROIboxSizeY, false);
```
- Track and visualize.
```
eTLDdesc.track(test_TD, false, true);
```

## Citation ##
Bharath Ramesh, Shihao Zhang, Hong Yang, Andres Ussa, Matthew Ong, Garrick Orchard and Cheng Xiang "e-TLD: Event-based Framework for Dynamic Object Tracking," in IEEE Transactions on Circuits and Systems for Video Technology, doi: 10.1109/TCSVT.2020.3044287.

```bibtex
@ARTICLE{9292994,
  author={B. {Ramesh} and S. {Zhang} and H. {Yang} and A. {Ussa} and M. {Ong} and G. {Orchard} and C. {Xiang}},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={e-TLD: Event-based Framework for Dynamic Object Tracking}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2020.3044287}}
}
```
