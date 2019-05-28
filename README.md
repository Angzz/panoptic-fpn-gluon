# Panoptic Feature Pyramid Networks

This is an unofficial implementation of [Panoptic-FPN](https://arxiv.org/abs/1901.02446) in a [gluon-cv](http://gluon-cv.mxnet.io) style, we implemented this framework in a fully [Gluon](https://mxnet.incubator.apache.org/versions/master/gluon/index.html) API, please stay tuned! 

## Main Results
### Cityscapes
* panoptic_fpn_resnet50_v1b_citys

| - | PQ | SQ | RQ | N | 
| :----------: | :----------: | :----------: | :----------: | :----------: | 
| All | 55.4 | 77.9 | 69.3 | 19 | 
| Things | 52.4 | 78.1 | 66.6 | 8 | 
| Stuff | 57.6 | 77.7 | 71.2 | 11 | 


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
Note that the correct data structure is shown below:
```Shell
$ ls ~/.mxnet/datasets/citys
├── gtFine
│   ├── train/
│   ├── val/
│   ├── test/
│   ├── cityscapes_panoptic_train/
│   ├── cityscapes_panoptic_val/
│   ├── cityscapes_panoptic_test/
│   ├── cityscapes_panoptic_train.json
│   └── cityscapes_panoptic_val.json
├── leftImg8bit
│   ├── train/
│   ├── val/
│   └── test/
```

3. More preparations can also refer to [GluonCV](https://gluon-cv.mxnet.io/index.html).

4. All experiments are performed on `8 * 2080ti` GPU with `Python3.5`, `cuda10.0` and `cudnn7.5.0`.

### COCO
* TODO

## Structure
```Shell
* Model : $ROOT/gluoncv/model_zoo/panoptic/
* Train & valid scripts : $ROOT/scripts/panoptic/
* Metric : $ROOT/gluoncv/utils/metric/
```

## Training & Inference 
### Cityscapes
1. Clone the training scripts [here](https://github.com/Angzz/panoptic-fpn-gluon/blob/master/scripts/panoptic/train_panoptic_fpn.py), then train `panoptic_fpn_resnet50_v1b_citys` with:
```Shell
python3 train_panoptic_fpn.py --network resnet50_v1b --gpus 0,1,2,3,4,5,6,7 --num-workers 32 --batch-size 8 --log-interval 10 --save-interval 20 --epochs 700 --lr_decay_epoch 430,590 --lr-warmup 4000
```
Note that we follow the training settings described in original [paper](https://arxiv.org/pdf/1901.02446.pdf).

2. Clone the validation scripts [here](https://github.com/Angzz/panoptic-fpn-gluon/blob/master/scripts/panoptic/eval_panoptic_fpn.py), then validate `panoptic_fpn_resnet50_v1b_citys` with: 
```Shell
python3 eval_panoptic_fpn.py --network resnet50_v1b --gpus 0,1,2,3,4,5,6,7 --pretrained ./XXX.params
```
### COCO
* TODO


## Reference 
* **Panoptic FPN:** Alexander Kirillov, Ross Girshick, Kaiming He, Piotr Dollár.<br />"Panoptic Feature Pyramid Networks." CVPR (2019 **oral**). [[paper](https://arxiv.org/pdf/1901.02446.pdf)] 
