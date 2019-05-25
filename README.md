# Panoptic Feature Pyramid Networks

This is an unofficial implementation of [Panoptic-FPN](https://arxiv.org/abs/1901.02446) in a [gluon-cv](http://gluon-cv.mxnet.io) style, we implemented this framework in a fully [Gluon](https://mxnet.incubator.apache.org/versions/master/gluon/index.html) API, please stay tuned! 

## Main Results
* TODO

## Installation 
1. Install cuda `10.0` and mxnet `1.4.0`.
```Shell
sudo pip3 install mxnet-cu100==1.4.0.post0
```
2. Clone the code, and install gluoncv with ``setup.py``.
```Shell
cd fcos-gluon-cv
sudo python3 setup.py build
sudo python3 setup.py install
```

## Preparation
### Cityscapes
1. Download `Cityscapes` datasets follow the official [tutorials](https://gluon-cv.mxnet.io/build/examples_datasets/cityscapes.html#sphx-glr-build-examples-datasets-cityscapes-py) and create a soft link.
```Shell
ln -s $DOWNLOAD_PATH ~/.mxnet/datasets/citys
```
You can also download from [Cityscapes](https://www.cityscapes-dataset.com/) and execute the command above.

2. Create Panoptic images for training and Inference, the code can be found [here](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py). Then execute the command below:
```Shell
python3 createPanopticImgs.py --dataset-folder ~/.mxnet/datasets/citys/gtFine/ --output-folder ~/.mxnet/datasets/citys/gtFine/
```
the correct data structure is shown below:
```Shell
$ ls ~/.mxnet/datasets/citys
├── dir1
│   ├── file11.ext
│   └── file12.ext
├── dir2
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── dir3
├── file_in_root.ext
└── README.md
```
   
3. More preparations can also refer to [GluonCV](https://gluon-cv.mxnet.io/index.html).

4. All experiments are performed on `8 * 2080ti` GPU with `Python3.5`, `cuda10.0` and `cudnn7.5.0`.
### COCO
* TODO


## Structure
```Shell
* Model : $ROOT/gluoncv/model_zoo/panoptic
* Train & valid scripts : $ROOT/scripts/panoptic
```

## Training & Inference 
1. Clone the training scripts [here](https://github.com/Angzz/panoptic-fpn-gluon/blob/master/scripts/panoptic/train_panoptic_fpn.py), then train `panoptic_fpn_resnet50_v1b_citys` with:
  ```Shell
  python3 train_panoptic_fpn.py --network resnet50_v1b --gpus 0,1,2,3,4,5,6,7 --num-workers 32 --batch-size 8 --log-interval 10 
  ```

## Reference 
* **Panoptic FPN:** Alexander Kirillov, Ross Girshick, Kaiming He, Piotr Dollár.<br />"Panoptic Feature Pyramid Networks." CVPR (2019 **oral**). [[paper](https://arxiv.org/pdf/1901.02446.pdf)] 
