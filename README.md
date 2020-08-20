# Simple Demo for Hand PointNet-based Real-Time 3D Hand Pose Estimation
A simple demo of the general pipeline for 3D hand pose estimation based on [Hand PointNet](https://sites.google.com/site/geliuhaontu/home/cvpr2018) using [Intel RealSense](https://github.com/IntelRealSense/librealsense.git) Depth Camera. The corresponding model is only trained on [MSRA](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sun_Cascaded_Hand_Pose_2015_CVPR_paper.pdf) dataset. 

- **The codes in this repository have not been fully tested. To be updated...**
- **The pre-trained models have not been submitted. To be updated...**

## Getting Started
### Requirements
The demo is developed and tested with Ubuntu 18.04, Python 2.7, and CUDA 9.0 using Intel RealSense depth camera D415. The following dependencies are also required:
```
pytorch==1.1: following official instruction
numba==0.45.0
pyrealsense2==2.24.0.965
opencv=4.1.0
tensorflow-gpu==1.9.0
numpy
matlibplot
scikit-learn
```
**Note**: After successfully installing tensorflow, you may not be able to import it into Python yet. You may need to add the following path into file `.bashrc`:
```
cd 
gedit .bashrc
```
Add the following lines at the end of the file `.bashrc`:
```
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
### Notes
- Change path where files `*.so` for sampling and grouping are in your pointnet++ build in `rs_demo.py` (line 31-32).

- For the given version, we assume hand is the cloest object to the depth camera for easy hand segmentation. More general solutions for hand segmentation can also be used, such as [PointNet++](https://github.com/charlesq34/pointnet2.git).

- Surface normals can be added using existing library, such as `PCL` library.

- Based on the original [Hand PointNet](https://sites.google.com/site/geliuhaontu/home/cvpr2018), we add two-layer MLP after the final PCA layer.

- Current model is trained only on [MSRA](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sun_Cascaded_Hand_Pose_2015_CVPR_paper.pdf) dataset.

### Pre-trained models
Download pre-trained models from [online drive]() and copy the file to folder `./model`.

### Launch Demo
Run programme with:
```
python rs_demo.py 
```
In this simple version, we use depth thresholding to segment hand from input 3D scene. Surface normals are not used.

## Current Resultss
<img src="./demo/demo.gif" width="500">

## Acknowledgements
This work is a study of [Hand PointNet](https://sites.google.com/site/geliuhaontu/home/cvpr2018) by Liuhao et al. and [PointNet++](https://github.com/charlesq34/pointnet2.git) by Charles et al. This code is mainly built upon and adapted from [Hand PointNet](https://sites.google.com/site/geliuhaontu/home/cvpr2018), [PointNet++](https://github.com/charlesq34/pointnet2.git), and [Intel RealSense](https://github.com/IntelRealSense/librealsense.git)