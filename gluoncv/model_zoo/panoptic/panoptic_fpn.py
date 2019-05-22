"""Panoptic Feature Pyramid Networks Model."""
from __future__ import absolute_import

import os
import warnings

import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn

from ..mask_rcnn.mask_rcnn import MaskRCNN
from ...nn.feature import PanopticFPNFeatureExpander

__all__ = ['PanopticFPN', 'get_panoptic_fpn',
           'panoptic_fpn_resnet50_v1b_citys',
           'panoptic_fpn_resnet101_v1d_citys']


class Seg(nn.HybridBlock):
    def __init__(self, classes, seg_channels, **kwargs):
        super(Seg, self).__init__(**kwargs)
        init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
        with self.name_scope():
            self.seg_head = nn.Conv2D(classes, 1, 1, 0, weight_initializer=init)
            self.seg_deconv = nn.Conv2DTranspose(seg_channels, 4, 4, 0)

    def hybrid_forward(self, F, x, im=None):
        # x: [B, C, H, W]
        x = self.seg_head(x)
        if not autograd.is_training():
            x = self.seg_deconv(x)
            x = F.slice_like(x, im, axes=(2, 3))
        x = F.softmax(x, axis=1)
        return x


class PanopticFPN(MaskRCNN):
    def __init__(self, features, seg_features, mask_classes, seg_classes,
                 mask_channels=256, seg_channels=128, rcnn_max_dets=1000,
                 deep_fcn=False, top_features=None, **kwargs):

        # Panoptic FPN do not use top features
        super(PanopticFPN, self).__init__(features, top_features, mask_classes,
                                          mask_channels=mask_channels, deep_fcn=deep_fcn,
                                          rcnn_max_dets=rcnn_max_dets, **kwargs)
        self.num_seg_class = len(seg_classes)
        with self.name_scope():
            self.seg_features = seg_features
            self.seg = Seg(self.num_seg_class, seg_channels)

    def hybrid_forward(self, F, x, gtbox=None):
        # Stuff predict
        seg_feats = self.seg_features(x)
        seg_feats = F.add_n(*seg_feats)

        if autograd.is_training():
            cls_pred, box_pred, mask_pred, rpn_box, samples, matches, \
                raw_rpn_score, raw_rpn_box, anchors = \
                super(PanopticFPN, self).hybrid_forward(F, x, gtbox)
            seg_pred = self.seg(seg_feats)
            return cls_pred, box_pred, mask_pred, seg_pred, rpn_box, samples, \
                   matches, raw_rpn_score, raw_rpn_box, anchors
        else:
            cls_ids, cls_scores, boxes, masks = \
                super(PanopticFPN, self).hybrid_forward(F, x)
            segms = self.seg(seg_feats, x)
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
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    panoptic_features = PanopticFPNFeatureExpander(network=base_network, pretrained=pretrained_base,
                                          outputs=['layers1_relu8_fwd',
                                                   'layers2_relu11_fwd',
                                                   'layers3_relu17_fwd',
                                                   'layers4_relu8_fwd'])
    # split features
    _input = mx.sym.var('data')
    out = panoptic_features(_input)
    mask_out, seg_out = out[:5], out[5:]
    mask_features = mx.gluon.SymbolBlock(
            mask_out, _input, params=panoptic_features.collect_params())
    seg_features = mx.gluon.SymbolBlock(
            seg_out, _input, params=panoptic_features.collect_params())

    top_features = None
    box_features = nn.HybridSequential()
    box_features.add(nn.AvgPool2D(pool_size=(3, 3), strides=2, padding=1))  # reduce to 7x7
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                         nn.Activation('relu'))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*mask', 'P',
                               '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])

    return get_panoptic_fpn(
        name='resnet50_v1b', dataset='citys', pretrained=pretrained, features=mask_features,
        seg_features=seg_features, top_features=top_features, mask_classes=mask_classes,
        seg_classes=seg_classes, box_features=box_features, mask_channels=256,
        seg_channels=128, rcnn_max_dets=1000, min_stage=2, max_stage=6,
        train_patterns=train_patterns, nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14),
        strides=(4, 8, 16, 32, 64), clip=4.42, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=0,
        num_sample=512, pos_iou_thresh=0.5, pos_ratio=0.25, deep_fcn=True,
        **kwargs)

def panoptic_fpn_resnet101_v1d_citys(pretrained=False, pretrained_base=True, **kwargs):
    pass
