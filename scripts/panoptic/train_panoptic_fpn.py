# -*- coding: utf-8 -*-
"""Train Panoptic FPN end to end."""
import argparse
import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import logging
import time
import cv2
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify
from gluoncv.data import get_segmentation_dataset
from gluoncv.data.transforms.presets.panoptic import PanopticFPNDefaultTrainTransform
from gluoncv.utils.metrics.citys_panoptic import CitysPanopticMetric


def parse_args():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN network end to end.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--dataset', type=str, default='citys',
                        help='Training dataset. Now support coco.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=str, default='',
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./mask_rcnn_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=str, default='',
                        help='Learning rate, default is 0.00125 for coco single gpu training.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='',
                        help='epochs at which learning rate decays. default is 17,23 for coco.')
    parser.add_argument('--lr-warmup', type=str, default='',
                        help='warmup iterations to adjust learning rate, default is 8000 for coco.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=str, default='',
                        help='Weight decay, default is 1e-4 for coco')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Print helpful debugging info once set.')
    # FPN options
    parser.add_argument('--use-fpn', action='store_true',
                        help='Whether to use feature pyramid network.')

    # Performance options
    parser.add_argument('--disable-hybridization', action='store_true',
                        help='Whether to disable hybridize the entire model. '
                             'Memory usage and speed will decrese.')
    parser.add_argument('--static-alloc', action='store_true',
                        help='Whether to use static memory allocation. Memory usage will increase.')

    args = parser.parse_args()
    args.epochs = int(args.epochs) if args.epochs else 26
    args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '17,23'
    args.lr = float(args.lr) if args.lr else 0.00125
    args.lr_warmup = int(args.lr_warmup) if args.lr_warmup else 8000
    args.wd = float(args.wd) if args.wd else 1e-4
    num_gpus = len(args.gpus.split(','))
    if num_gpus == 1:
        args.lr_warmup = -1
    else:
        args.lr *= num_gpus
        args.lr_warmup /= num_gpus
    return args


class RPNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')

    def update(self, labels, preds):
        # label: [rpn_label, rpn_weight]
        # preds: [rpn_cls_logits]
        rpn_label, rpn_weight = labels
        rpn_cls_logits = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_weight)

        # cls_logits (b, na*h*w, 1)
        pred_label = mx.nd.sigmoid(rpn_cls_logits) >= 0.5
        # label (b, na*h*w, 1)
        num_acc = mx.nd.sum((pred_label == rpn_label) * rpn_weight)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()


class RPNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')

    def update(self, labels, preds):
        # label = [rpn_bbox_target, rpn_bbox_weight]
        # pred = [rpn_bbox_reg]
        rpn_bbox_target, rpn_bbox_weight = labels
        rpn_bbox_reg = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(
            rpn_bbox_weight * mx.nd.smooth_l1(rpn_bbox_reg - rpn_bbox_target, scalar=3))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')

    def update(self, labels, preds):
        # label = [rcnn_label]
        # pred = [rcnn_cls]
        rcnn_label = labels[0]
        rcnn_cls = preds[0]

        # calculate num_acc
        pred_label = mx.nd.argmax(rcnn_cls, axis=-1)
        num_acc = mx.nd.sum(pred_label == rcnn_label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += rcnn_label.size


class RCNNL1LossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')

    def update(self, labels, preds):
        # label = [rcnn_bbox_target, rcnn_bbox_weight]
        # pred = [rcnn_reg]
        rcnn_bbox_target, rcnn_bbox_weight = labels
        rcnn_bbox_reg = preds[0]

        # calculate num_inst
        num_inst = mx.nd.sum(rcnn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(
            rcnn_bbox_weight * mx.nd.smooth_l1(rcnn_bbox_reg - rcnn_bbox_target, scalar=1))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


class MaskAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MaskAccMetric, self).__init__('MaskAcc')

    def update(self, labels, preds):
        # label = [rcnn_mask_target, rcnn_mask_weight]
        # pred = [rcnn_mask]
        rcnn_mask_target, rcnn_mask_weight = labels
        rcnn_mask = preds[0]

        # calculate num_inst
        num_inst = mx.nd.sum(rcnn_mask_weight)

        # rcnn_mask (b, n, c, h, w)
        pred_label = mx.nd.sigmoid(rcnn_mask) >= 0.5
        label = rcnn_mask_target >= 0.5
        # label (b, n, c, h, w)
        num_acc = mx.nd.sum((pred_label == label) * rcnn_mask_weight)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()


class MaskFGAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(MaskFGAccMetric, self).__init__('MaskFGAcc')

    def update(self, labels, preds):
        # label = [rcnn_mask_target, rcnn_mask_weight]
        # pred = [rcnn_mask]
        rcnn_mask_target, rcnn_mask_weight = labels
        rcnn_mask = preds[0]

        # calculate num_inst
        num_inst = mx.nd.sum(rcnn_mask_target)

        # rcnn_mask (b, n, c, h, w)
        pred_label = mx.nd.sigmoid(rcnn_mask) >= 0.5
        label = rcnn_mask_target >= 0.5
        # label (b, n, c, h, w)
        num_acc = mx.nd.sum((pred_label == label) * label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()


class SegAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(SegAccMetric, self).__init__('SegAcc')

    def update(self, labels, preds):
        rcnn_segms = preds[0][0]
        labels = labels[0][0]

        pred_label = mx.nd.softmax(rcnn_segms, axis=0)
        pred_label = mx.nd.argmax(pred_label, axis=0)
        pred_label = pred_label.astype('int32')

        valid_mask = labels != -1
        num_inst = mx.nd.sum(valid_mask)
        num_acc = mx.nd.sum((pred_label == labels) * valid_mask)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()


class SoftmaxCrossEntropyLoss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, ignore_label=-1, from_softmax=True,
            eps=1e-5, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(None, batch_axis, **kwargs)
        self._ignore_label = ignore_label
        self._from_softmax = from_softmax
        self._eps = eps

    def hybrid_forward(self, F, pred, label):
        """Compute loss"""
        mask = label >= 0
        vlabel = label * mask
        loss = -F.pick(F.log(pred + self._eps), vlabel, axis=1, keepdims=False)
        loss = F.where(mask, loss, F.zeros_like(loss))
        return F.sum(loss) / F.maximum(F.sum(mask), 1).astype(loss.dtype)


def get_dataset(dataset):
    if dataset.lower() == 'citys':
        train_dataset = get_segmentation_dataset(dataset+'_panoptic', split='train', mode='train')
        val_dataset = get_segmentation_dataset(dataset+'_panoptic', split='val', mode='val')
        val_metric = CitysPanopticMetric(val_dataset, "cityscapes_panoptic_val")
    elif dataset.lower() == 'coco':
        pass
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, train_transform, val_transform, batch_size,
                   num_workers, multi_stage):
    """Get dataloader."""
    train_bfn = batchify.Tuple(*[batchify.Append() for _ in range(7)])
    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(train_transform(net, ashape=net.ashape, multi_stage=multi_stage)),
        batch_size, True, batchify_fn=train_bfn, last_batch='rollover', num_workers=num_workers)
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(2)])
    val_loader = mx.gluon.data.DataLoader(val_dataset, batch_size, False, batchify_fn=val_bfn,
        last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric, args):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    net.hybridize(static_alloc=True)
    for ib, batch in enumerate(val_data):
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        det_masks = []
        det_segms = []
        det_infos = []
        for x, im_info in zip(*batch):
            # get prediction results
            ids, scores, bboxes, masks, segms = net(x)
            det_bboxes.append(clipper(bboxes, x))
            det_ids.append(ids)
            det_scores.append(scores)
            det_masks.append(masks)
            det_segms.append(segms)
            det_infos.append(im_info)
        # update metric
        for det_bbox, det_id, det_score, det_mask, det_segm, det_info in zip(
                det_bboxes, det_ids, det_scores, det_masks, det_segms, det_infos):
            for i in range(det_info.shape[0]):
                # numpy everything
                det_bbox = det_bbox[i].asnumpy()
                det_id = det_id[i].asnumpy()
                det_score = det_score[i].asnumpy()
                det_mask = det_mask[i].asnumpy()
                det_segm = det_segm[i].asnumpy()
                det_info = det_info[i].asnumpy()
                # filter by conf threshold
                im_height, im_width, im_scale = det_info
                valid = np.where(((det_id >= 0) & (det_score >= 0.001)))[0]
                det_id = det_id[valid]
                det_score = det_score[valid]
                det_bbox = det_bbox[valid] / im_scale
                det_mask = det_mask[valid]
                # fill full mask
                im_height, im_width = int(round(im_height / im_scale)), int(
                    round(im_width / im_scale))
                full_masks = []
                for bbox, mask in zip(det_bbox, det_mask):
                    full_masks.append(gdata.transforms.mask.fill(
                        mask, bbox, (im_width, im_height)))
                full_masks = np.array(full_masks)
                # fill full segm
                full_segms = cv2.resize(det_segm.transpose(1, 2, 0), (im_width, im_height),
                        interpolation=cv2.INTER_LINEAR)
                full_segms = np.argmax(full_segms, axis=2)
                eval_metric.update(det_bbox, det_id, det_score, full_masks, full_segms)
    return eval_metric.get()


def get_lr_at_iter(alpha):
    return 1. / 10. * (1 - alpha) + alpha


def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().setattr('grad_req', 'null')
    net.collect_train_params().setattr('grad_req', 'write')
    trainer = gluon.Trainer(
        net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
        'sgd',
        {'learning_rate': args.lr,
         'wd': args.wd,
         'momentum': args.momentum,
         'clip_gradient': 5})

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    lr_warmup = float(args.lr_warmup)  # avoid int division

    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1 / 9.)  # == smoothl1
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1
    rcnn_mask_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rcnn_segm_loss = SoftmaxCrossEntropyLoss()
    metrics = [mx.metric.Loss('RPN_Conf'),
               mx.metric.Loss('RPN_SmoothL1'),
               mx.metric.Loss('RCNN_CrossEntropy'),
               mx.metric.Loss('RCNN_SmoothL1'),
               mx.metric.Loss('RCNN_Mask'),
               mx.metric.Loss('RCNN_Seg')]

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    rcnn_mask_metric = MaskAccMetric()
    rcnn_fgmask_metric = MaskFGAccMetric()
    rcnn_seg_metric = SegAccMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric,
                rcnn_acc_metric, rcnn_bbox_metric,
                rcnn_mask_metric, rcnn_fgmask_metric,
                rcnn_seg_metric]

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    if args.verbose:
        logger.info('Trainable parameters:')
        logger.info(net.collect_train_params().keys())
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        for metric in metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        if not args.disable_hybridization:
            net.hybridize(static_alloc=args.static_alloc)
        base_lr = trainer.learning_rate
        for i, batch in enumerate(train_data):
            if epoch == 0 and i <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup)
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info(
                            '[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
                    trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx)
            batch_size = len(batch[0])
            losses = []
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]
            with autograd.record():
                for data, label, gt_segms, gt_mask, rpn_cls_targets, rpn_box_targets, \
                    rpn_box_masks in zip(*batch):
                    gt_label = label[:, :, 4:5]
                    gt_box = label[:, :, :4]
                    cls_pred, box_pred, mask_pred, seg_pred, roi, samples, matches, rpn_score, \
                        rpn_box, anchors = net(data, gt_box)
                    # losses of rpn
                    rpn_score = rpn_score.squeeze(axis=-1)
                    num_rpn_pos = (rpn_cls_targets >= 0).sum()
                    rpn_loss1 = rpn_cls_loss(rpn_score, rpn_cls_targets,
                                             rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
                    rpn_loss2 = rpn_box_loss(rpn_box, rpn_box_targets,
                                             rpn_box_masks) * rpn_box.size / num_rpn_pos
                    # rpn overall loss, use sum rather than average
                    rpn_loss = rpn_loss1 + rpn_loss2
                    # generate targets for rcnn
                    cls_targets, box_targets, box_masks = net.target_generator(
                            roi, samples, matches, gt_label, gt_box)
                    # losses of rcnn
                    num_rcnn_pos = (cls_targets >= 0).sum()
                    rcnn_loss1 = rcnn_cls_loss(cls_pred, cls_targets,
                                               cls_targets >= 0) * cls_targets.size / \
                                 cls_targets.shape[0] / num_rcnn_pos
                    rcnn_loss2 = rcnn_box_loss(box_pred, box_targets, box_masks) * box_pred.size / \
                                 box_pred.shape[0] / num_rcnn_pos
                    rcnn_loss = rcnn_loss1 + rcnn_loss2
                    # generate targets for mask
                    mask_targets, mask_masks = net.mask_target(roi, gt_mask, matches, cls_targets)
                    # loss of mask
                    mask_loss = rcnn_mask_loss(mask_pred, mask_targets, mask_masks) * \
                                mask_targets.size / mask_targets.shape[0] / mask_masks.sum()
                    # loss of seg
                    seg_loss = rcnn_segm_loss(seg_pred, gt_segms)
                    # overall losses
                    panoptic_loss = 0.75 * mask_loss + 1. * seg_loss
                    losses.append(rpn_loss.sum() + rcnn_loss.sum() + panoptic_loss.sum())
                    metric_losses[0].append(rpn_loss1.sum())
                    metric_losses[1].append(rpn_loss2.sum())
                    metric_losses[2].append(rcnn_loss1.sum())
                    metric_losses[3].append(rcnn_loss2.sum())
                    metric_losses[4].append(mask_loss.sum())
                    metric_losses[5].append(seg_loss.sum())
                    add_losses[0].append([[rpn_cls_targets, rpn_cls_targets >= 0], [rpn_score]])
                    add_losses[1].append([[rpn_box_targets, rpn_box_masks], [rpn_box]])
                    add_losses[2].append([[cls_targets], [cls_pred]])
                    add_losses[3].append([[box_targets, box_masks], [box_pred]])
                    add_losses[4].append([[mask_targets, mask_masks], [mask_pred]])
                    add_losses[5].append([[mask_targets, mask_masks], [mask_pred]])
                    add_losses[6].append([[gt_segms], [seg_pred]])
                autograd.backward(losses)
                for metric, record in zip(metrics, metric_losses):
                    metric.update(0, record)
                for metric, records in zip(metrics2, add_losses):
                    for pred in records:
                        metric.update(pred[0], pred[1])
            trainer.step(batch_size)
            # update metrics
            if args.log_interval and not (i + 1) % args.log_interval:
                msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics + metrics2])
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                    epoch, i, args.log_interval * batch_size / (time.time() - btic), msg))
                btic = time.time()

        msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
        logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
            epoch, (time.time() - tic), msg))
        if not (epoch + 1) % args.val_interval:
            # consider reduce the frequency of validation to save time
            result = validate(net, val_data, ctx, eval_metric, args)
        else:
            current_map = 0.
        save_params(net, logger, best_map, current_map, epoch, args.save_interval, args.save_prefix)


if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx)  # 1 batch per device

    # network
    module_list = []
    if args.use_fpn:
        module_list.append('fpn')
    net_name = '_'.join(('panoptic', *module_list, args.network, args.dataset))
    args.save_prefix += net_name
    net = get_model(net_name, pretrained_base=True)
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    net.collect_params().reset_ctx(ctx)

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset)
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, PanopticFPNDefaultTrainTransform,
        None, args.batch_size, args.num_workers, args.use_fpn)

    # training
    train(net, train_data, val_data, eval_metric, ctx, args)
