from __future__ import division
from __future__ import print_function

import argparse
import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import nd
from mxnet import gluon
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

def parse_args():
    parser = argparse.ArgumentParser(description='Eval SSD networks.')
    parser.add_argument('--network', type=str, default='vgg16_atrous',
                        help="Base network name")
    parser.add_argument('--quantized', action='store_true',
                        help='use int8 pretrained model')
    parser.add_argument('--data-shape', type=int, default=300,
                        help="Input data shape")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='eval mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='eval dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    args = parser.parse_args()
    return args

def get_dataset(dataset, data_shape):
    if dataset.lower() == 'voc':
        val_dataset = gdata.VOCDetection(splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(data_shape, data_shape))
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric

def get_dataloader(val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)), batchify_fn=batchify_fn,
        batch_size=batch_size, shuffle=False, last_batch='keep', num_workers=num_workers)
    return val_loader

def validate(net, val_data, ctx, classes, size, metric):
    """Test on validation dataset."""
    net.collect_params().reset_ctx(ctx)
    metric.reset()
    if not args.quantized:
        net.set_nms(nms_thresh=0.45, nms_topk=400)
        net.hybridize()
    else:
        net.hybridize(static_alloc=True, static_shape=True)
    with tqdm(total=size) as pbar:
        start = time.time()
        for ib, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
            pbar.update(batch[0].shape[0])
        end = time.time()
        speed = size / (end - start)
        print('Throughput is %f img/sec.'% speed)
    return metric.get()

if __name__ == '__main__':
    args = parse_args()

    # eval contexts
    num_gpus = args.num_gpus
    if num_gpus > 0:
        args.batch_size *= num_gpus
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

    # network
    net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.dataset))
    if args.quantized:
        net_name = '_'.join((net_name, 'int8'))
    args.save_prefix += net_name
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(net_name, pretrained=True)
    else:
        net = gcv.model_zoo.get_model(net_name, pretrained=False)
        net.load_parameters(args.pretrained.strip())

    # eval data
    val_dataset, val_metric = get_dataset(args.dataset, args.data_shape)
    val_data = get_dataloader(
        val_dataset, args.data_shape, args.batch_size, args.num_workers)
    classes = val_dataset.classes  # class names

    # eval
    names, values = validate(net, val_data, ctx, classes, len(val_dataset), val_metric)
    for k, v in zip(names, values):
        print(k, v)
