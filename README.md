# yolov3-MIVisionX

usage: 

python detect_one_miv.py --image=images/dog-cycle-car.png

### Acknowledgements

Some parts are taken from github project https://github.com/jasonlovescoding/YOLOv3-caffe

The weight files can be generated from the MIVisionX model compiler following the commands caffe2nnir and nnir2openvx.
The model can be downloaded from https://download.csdn.net/download/jason_ranger/10519464
Copy the weights.bin and libannpython.so to the same root project folder.

### Notes

A simplest YOLOv3 model in using MIVisionX model compiler and libraries for python.

This is merely a practice project. Note that the interp layer is interpreted as upsample (by 2) using nearest neighbour in MIVisionX..

This is because interp layer is only viable in deeplab caffe, not in the official one. 

