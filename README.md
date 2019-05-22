# Panoptic Feature Pyramid Networks

This is an unofficial implementation of [Panoptic-FPN](https://arxiv.org/abs/1901.02446) in a [gluon-cv](http://gluon-cv.mxnet.io) style, we implemented this panoptic segmentation in a fully [Gluon](https://mxnet.incubator.apache.org/versions/master/gluon/index.html) API, please stay tuned! 

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
1. Download `Cityscapes` dataset follow the official [tutorials](https://gluon-cv.mxnet.io/build/examples_datasets/cityscapes.html#sphx-glr-build-examples-datasets-cityscapes-py) and create a soft link.
  ```Shell
  ln -s $DOWNLOAD_PATH ~/.mxnet/datasets/citys
  ```
   You can also download from [Cityscapes](https://www.cityscapes-dataset.com/) and execute the command above.

2. Preparing the panoptic images follow the official code [here.](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py) You can clone this script and execute:
  ```Shell
  python3 createPanopticImgs.py --dataset-folder ~/.mxnet/datasets/citys/gtFine/ --output-folder ~/.mxnet/datasets/citys/gtFine/
  ```
   
3. More preparations can also refer to [GluonCV](https://gluon-cv.mxnet.io/index.html).

4. All experiments are performed on `8 * 2080ti` GPU with `Python3.5`, `cuda10.0` and `cudnn7.5.0`.

## Structure
```Shell
* Model : $ROOT/gluoncv/model_zoo/panoptic
* Train & valid scripts : $ROOT/scripts/panoptic
* Data Transform : $ROOT/gluoncv/data/transform/presets
```

## Training & Inference 
* TODO

## Reference 

* **Panoptic FPN:** Alexander Kirillov, Ross Girshick, Kaiming He, Piotr Dollár.<br />"Panoptic Feature Pyramid Networks." CVPR (2019 **oral**). [[paper](https://arxiv.org/pdf/1901.02446.pdf)]
