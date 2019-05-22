"""Transforms for RCNN series."""
from __future__ import absolute_import

import copy
from random import randint

import mxnet as mx
import numpy as np

from .. import bbox as tbbox
from .. import image as timage
from .. import mask as tmask

__all__ = ['transform_test', 'load_test',
           'PanopticFPNDefaultTrainTransform', 'PanopticFPNDefaultValTransform']


def transform_test(imgs, short=600, max_size=1000, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : NDArray or iterable of NDArray
        Image(s) to be transformed.
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        img = timage.resize_short_within(img, short, max_size)
        orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


def load_test(filenames, short=600, max_size=1000, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or list of filenames.

    Parameters
    ----------
    filenames : str or list of str
        Image filename(s) to be loaded.
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(filenames, str):
        filenames = [filenames]
    imgs = [mx.image.imread(f) for f in filenames]
    return transform_test(imgs, short, max_size, mean, std)


class PanopticFPNDefaultTrainTransform(object):
    def __init__(self, net=None, dataset="citys_panoptic", mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), box_norm=(1., 1., 1., 1.), num_sample=256,
                 pos_iou_thresh=0.7,neg_iou_thresh=0.3, pos_ratio=0.5, ashape=128,
                 multi_stage=True, down_sample_rate=4, **kwargs):
        self._mean = mean
        self._std = std
        self._dataset = dataset
        self._anchors = None
        self._multi_stage = multi_stage
        self.down_sample_rate = down_sample_rate
        if net is None:
            return

        # use fake data to generate fixed anchors for target generation
        anchors = []  # [P2, P3, P4, P5]
        # in case network has reset_ctx to gpu
        anchor_generator = copy.deepcopy(net.rpn.anchor_generator)
        anchor_generator.collect_params().reset_ctx(None)
        if self._multi_stage:
            for ag in anchor_generator:
                anchor = ag(mx.nd.zeros((1, 3, ashape, ashape))).reshape((1, 1, ashape, ashape, -1))
                ashape = max(ashape // 2, 16)
                anchors.append(anchor)
        self._anchors = anchors
        # record feature extractor for infer_shape
        if not hasattr(net, 'features'):
            raise ValueError("Cannot find features in network, it is a Mask RCNN network?")
        self._feat_sym = net.features(mx.sym.var(name='data'))
        from ....model_zoo.rpn.rpn_target import RPNTargetGenerator
        self._target_generator = RPNTargetGenerator(
            num_sample=num_sample, pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh, pos_ratio=pos_ratio,
            stds=box_norm, **kwargs)

    def __call__(self, img, segms, inst, bbox):
        '''
        src : [H, W, 3]
        segms : [H, W]
        inst : [N, H, W]
        bbox: [N, 5]
        '''
        """Apply transform to training image/label."""
        h, w, _ = img.shape
        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        sh = img.shape[1] // self.down_sample_rate
        sw = img.shape[2] // self.down_sample_rate
        segms = mx.image.imresize(segms.expand_dims(-1), sw, sh, interp=0)
        segms = segms.squeeze()

        # generate RPN target so cpu workers can help reduce the workload
        # feat_h, feat_w = (img.shape[1] // self._stride, img.shape[2] // self._stride)
        gt_bboxes = mx.nd.array(bbox[:, :4])
        if self._multi_stage:
            oshapes = []
            anchor_targets = []
            for feat_sym in self._feat_sym:
                oshapes.append(feat_sym.infer_shape(data=(1, 3, img.shape[1], img.shape[2]))[1][0])
            for anchor, oshape in zip(self._anchors, oshapes):
                anchor = anchor[:, :, :oshape[2], :oshape[3], :].reshape((-1, 4))
                anchor_targets.append(anchor)
            anchor_targets = mx.nd.concat(*anchor_targets, dim=0)
            cls_target, box_target, box_mask = self._target_generator(
                gt_bboxes, anchor_targets, img.shape[2], img.shape[1])
        return img, bbox.astype(img.dtype), segms, inst, cls_target, box_target, box_mask


class PanoptocFPNDefaultValTransform(object):
    """Default Mask RCNN validation transform.

    Parameters
    ----------
    short : int, default is 600
        Resize image shorter side to ``short``.
    max_size : int, default is 1000
        Make sure image longer side is smaller than ``max_size``.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """

    def __init__(self, short=600, max_size=1000,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std
        self._short = short
        self._max_size = max_size

    def __call__(self, src, label, mask):
        """Apply transform to validation image/label."""
        # resize shorter side but keep in max_size
        h, _, _ = src.shape
        img = timage.resize_short_within(src, self._short, self._max_size, interp=1)
        # no scaling ground-truth, return image scaling ratio instead
        im_scale = float(img.shape[0]) / h

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, mx.nd.array([img.shape[-2], img.shape[-1], im_scale])
