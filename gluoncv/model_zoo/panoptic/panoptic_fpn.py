"""Panoptic Feature Pyramid Networks Model."""
from __future__ import absolute_import

import os
import warnings

import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn

from ..mask_rcnn.mask_rcnn import MaskRCNN
from ...nn.feature import FPNFeatureExpander

__all__ = ['PanopticFPN', 'get_panoptic_fpn',
           'panoptic_fpn_resnet50_v1b_citys',
           'panoptic_fpn_resnet101_v1d_citys']


class GroupNorm(nn.HybridBlock):
    """
    If the batch size is small, it's better to use GroupNorm instead of BatchNorm.
    GroupNorm achieves good results even at small batch sizes.
    Reference:
      https://arxiv.org/pdf/1803.08494.pdf
    """
    def __init__(self, num_channels, num_groups=32, eps=1e-5,
                 multi_precision=False, prefix=None, **kwargs):
        super(GroupNorm, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get('{}_weight'.format(prefix), grad_req='write',
                                          shape=(1, num_channels, 1, 1))
            self.bias = self.params.get('{}_bias'.format(prefix), grad_req='write',
                                        shape=(1, num_channels, 1, 1))
        self.C = num_channels
        self.G = num_groups
        self.eps = eps
        self.multi_precision = multi_precision
        assert self.C % self.G == 0

    def hybrid_forward(self, F, x, weight, bias):
        # (N,C,H,W) -> (N,G,H*W*C//G)
        x_new = F.reshape(x, (0, self.G, -1))
        if self.multi_precision:
            # (N,G,H*W*C//G) -> (N,G,1)
            mean = F.mean(F.cast(x_new, "float32"), axis=-1, keepdims=True)
            mean = F.cast(mean, "float16")
        else:
            mean = F.mean(x_new, axis=-1, keepdims=True)
        # (N,G,H*W*C//G)
        centered_x_new = F.broadcast_minus(x_new, mean)
        if self.multi_precision:
            # (N,G,H*W*C//G) -> (N,G,1)
            var = F.mean(F.cast(F.square(centered_x_new),"float32"), axis=-1, keepdims=True)
            var = F.cast(var, "float16")
        else:
            var = F.mean(F.square(centered_x_new), axis=-1, keepdims=True)
        # (N,G,H*W*C//G) -> (N,C,H,W)
        x_new = F.broadcast_div(centered_x_new, F.sqrt(var + self.eps)).reshape_like(x)
        x_new = F.broadcast_add(F.broadcast_mul(x_new, weight),bias)
        return x_new


class SegHead(nn.HybridBlock):
    def __init__(self, fpn_features, seg_classes, seg_channels, **kwargs):
        super(SegHead, self).__init__(**kwargs)
        init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2.)
        with self.name_scope():
            self._features = fpn_features
            self.seghead = nn.HybridSequential()
            for i in range(4):
                block = nn.HybridSequential()
                if i == 3:
                    block.add(nn.Conv2D(
                        seg_channels, 3, 1, 1, weight_initializer=init))
                    block.add(GroupNorm(num_channels=seg_channels, prefix="gnseg{}".format(i)))
                    block.add(nn.Activation('relu'))
                else:
                    for j in range(3 - i):
                        block.add(nn.Conv2D(
                            seg_channels, 3, 1, 1, weight_initializer=init, activation='relu'))
                        block.add(nn.Activation('relu'))
                        block.add(GroupNorm(num_channels=seg_channels, prefix="gnseg{}".format(i)))
                        block.add(nn.Conv2DTranspose(
                            seg_channels, 2, 2, 0, weight_initializer=init))
                self.seghead.add(block)
            self.seg_predictor = nn.Conv2D(seg_classes, 1, 1, 0, weight_initializer=init)

    def hybrid_forward(self, F, x):
        feats = self._features(x)
        feats = feats[::-1][1:]
        seg_feats = []
        for i, f in enumerate(feats):
            f = self.seghead[i](f)
            f = F.slice_like(f, F.zeros_like(feats[-1]), axes=(2, 3))
            seg_feats.append(f)
        seg_feats = F.add_n(*seg_feats)
        out = self.seg_predictor(seg_feats)
        out = F.softmax(out, axis=1)
        return out


class PanopticFPN(MaskRCNN):
    def __init__(self, features, mask_classes, seg_classes,
                 mask_channels=256, seg_channels=128, rcnn_max_dets=1000,
                 deep_fcn=False, top_features=None, **kwargs):
        # Panoptic FPN do not use top features
        super(PanopticFPN, self).__init__(features, top_features, mask_classes,
                                          mask_channels=mask_channels, deep_fcn=deep_fcn,
                                          rcnn_max_dets=rcnn_max_dets, **kwargs)
        self.num_seg_class = len(seg_classes)
        with self.name_scope():
            self.seg_head = SegHead(features, self.num_seg_class, seg_channels)

    def hybrid_forward(self, F, x, gtbox=None):
        # Stuff predict
        if autograd.is_training():
            cls_pred, box_pred, mask_pred, rpn_box, samples, matches, \
                raw_rpn_score, raw_rpn_box, anchors = \
                super(PanopticFPN, self).hybrid_forward(F, x, gtbox)
            seg_pred = self.seg_head(x)
            return cls_pred, box_pred, mask_pred, seg_pred, rpn_box, samples, \
                   matches, raw_rpn_score, raw_rpn_box, anchors
        else:
            cls_ids, cls_scores, boxes, masks = \
                super(PanopticFPN, self).hybrid_forward(F, x)
            segms = self.seg_head(x)
            return cls_ids, cls_scores, boxes, masks, segms


def get_panoptic_fpn(name, dataset, pretrained=False, ctx=mx.cpu(),
                     root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    net = PanopticFPN(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('panoptic_fpn', name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    return net


def panoptic_fpn_resnet50_v1b_citys(pretrained=False, pretrained_base=True, **kwargs):
    from ..resnetv1b import resnet50_v1b
    from ...data import CitysPanoptic
    mask_classes = CitysPanoptic.MASK_CLASS
    seg_classes = CitysPanoptic.SEG_CLASS
    seg_classes.extend(mask_classes)
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    features = FPNFeatureExpander(
        network=base_network, use_p6=True, no_bias=False, pretrained=pretrained_base,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256])

    top_features = None
    box_features = nn.HybridSequential()
    box_features.add(nn.AvgPool2D(pool_size=(3, 3), strides=2, padding=1))  # reduce to 7x7
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                         nn.Activation('relu'))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*mask', 'P',
                               '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])

    return get_panoptic_fpn(
        name='resnet50_v1b', dataset='citys', pretrained=pretrained,
        features=features, top_features=top_features, mask_classes=mask_classes,
        seg_classes=seg_classes, box_features=box_features, mask_channels=256,
        seg_channels=128, rcnn_max_dets=1000, min_stage=2, max_stage=6,
        train_patterns=train_patterns, nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=(4, 8, 16, 32, 64),
        clip=4.42, rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(512, 512), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000, rpn_test_pre_nms=6000,
        rpn_test_post_nms=1000, rpn_min_size=0, num_sample=512, pos_iou_thresh=0.5,
        pos_ratio=0.25, deep_fcn=True, **kwargs)

def panoptic_fpn_resnet101_v1d_citys(pretrained=False, pretrained_base=True, **kwargs):
    from ..resnetv1b import resnet101_v1d
    from ...data import CitysPanoptic
    mask_classes = CitysPanoptic.MASK_CLASS
    seg_classes = CitysPanoptic.SEG_CLASS
    seg_classes.extend(mask_classes)
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    features = FPNFeatureExpander(
        network=base_network, use_p6=True, no_bias=False, pretrained=pretrained_base,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu68_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256])

    top_features = None
    box_features = nn.HybridSequential()
    box_features.add(nn.AvgPool2D(pool_size=(3, 3), strides=2, padding=1))  # reduce to 7x7
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                         nn.Activation('relu'))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*mask', 'P',
                               '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])

    return get_panoptic_fpn(
        name='resne101_v1d', dataset='citys', pretrained=pretrained,
        features=features, top_features=top_features, mask_classes=mask_classes,
        seg_classes=seg_classes, box_features=box_features, mask_channels=256,
        seg_channels=128, rcnn_max_dets=1000, min_stage=2, max_stage=6,
        train_patterns=train_patterns, nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=(4, 8, 16, 32, 64),
        clip=4.42, rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(512, 512), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000, rpn_test_pre_nms=6000,
        rpn_test_post_nms=1000, rpn_min_size=0, num_sample=512, pos_iou_thresh=0.5,
        pos_ratio=0.25, deep_fcn=True, **kwargs)
