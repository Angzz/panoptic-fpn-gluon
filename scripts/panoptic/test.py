# -*- coding: utf-8 -*-
import sys
sys.path.append('/data/tmp/gluon-cv/')
import mxnet as mx

from gluoncv.nn.feature import PanopticFPNFeatureExpander, FPNFeatureExpander
from gluoncv.model_zoo.resnetv1b import resnet50_v1b
from IPython import embed

base_network = resnet50_v1b(pretrained=True, dilated=False, use_global_stats=True)
feature = PanopticFPNFeatureExpander(network=base_network, pretrained=True,
                                     outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd',
                                              'layers3_relu17_fwd', 'layers4_relu8_fwd'])

out = feature(mx.sym.var('data'))
for o in out:
    print(o.infer_shape(data=(1, 3, 3213, 4532))[1][0])
