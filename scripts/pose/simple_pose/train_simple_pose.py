from __future__ import division

import argparse, time, logging, os, math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler, LRSequential
from gluoncv.data.transforms.presets.simple_pose import SimplePoseDefaultTrainTransform
from gluoncv.utils.metrics import HeatmapAccuracy

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/coco',
                    help='training and validation pictures to use.')
parser.add_argument('--num-joints', type=int, required=True,
                    help='Number of joints to detect')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--num-epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--lr-mode', type=str, default='step',
                    help='learning rate scheduler mode. options are step, poly and cosine.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-period', type=int, default=0,
                    help='interval for periodic learning rate decays. default is 0 to disable.')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays. default is 40,60.')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate. default is 0.0.')
parser.add_argument('--warmup-epochs', type=int, default=0,
                    help='number of warmup epochs.')
parser.add_argument('--last-gamma', action='store_true',
                    help='whether to init gamma of the last BN layer in each bottleneck to 0.')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--input-size', type=str, default='256,192',
                    help='size of the input image size. default is 256,192')
parser.add_argument('--sigma', type=float, default=2,
                    help='value of the sigma parameter of the gaussian target generation. default is 2')
parser.add_argument('--mean', type=str, default='0.485,0.456,0.406',
                    help='mean vector for normalization')
parser.add_argument('--std', type=str, default='0.229,0.224,0.225',
                    help='std vector for normalization')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--use-pretrained-base', action='store_true',
                    help='enable using pretrained base model from gluon.')
parser.add_argument('--no-wd', action='store_true',
                    help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
parser.add_argument('--save-frequency', type=int, default=1,
                    help='frequency of model saving.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--log-interval', type=int, default=20,
                    help='Number of batches to wait before logging.')
parser.add_argument('--logging-file', type=str, default='keypoints.log',
                    help='name of training log file')
opt = parser.parse_args()

filehandler = logging.FileHandler(opt.logging_file)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(opt)

batch_size = opt.batch_size
num_joints = opt.num_joints

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

model_name = opt.model

kwargs = {'ctx': context, 'num_joints': num_joints,
          'pretrained': opt.use_pretrained,
          'pretrained_base': opt.use_pretrained_base,
          'pretrained_ctx': context}

net = get_model(model_name, **kwargs)
net.cast(opt.dtype)

def get_data_loader(data_dir, batch_size, num_workers, input_size):

    def train_batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        weight = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        imgid = gluon.utils.split_and_load(batch[3], ctx_list=ctx, batch_axis=0)
        return data, label, weight, imgid

    train_dataset = mscoco.keypoints.COCOKeyPoints(data_dir, aspect_ratio=4./3.,
                                                   splits=('person_keypoints_train2017'))
    heatmap_size = [int(i/4) for i in input_size]

    meanvec = [float(i) for i in opt.mean.split(',')]
    stdvec = [float(i) for i in opt.std.split(',')]
    transform_train = SimplePoseDefaultTrainTransform(num_joints=train_dataset.num_joints,
                                                      joint_pairs=train_dataset.joint_pairs,
                                                      image_size=input_size, heatmap_size=heatmap_size,
                                                      sigma=opt.sigma, scale_factor=0.30, rotation_factor=40,
                                                      mean=meanvec, std=stdvec, random_flip=True)

    train_data = gluon.data.DataLoader(
        train_dataset.transform(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

    return train_dataset, train_data, train_batch_fn

input_size = [int(i) for i in opt.input_size.split(',')]
train_dataset, train_data,  train_batch_fn = get_data_loader(opt.data_dir, batch_size,
                                                             num_workers, input_size)

num_training_samples = len(train_dataset)
lr_decay = opt.lr_decay
lr_decay_period = opt.lr_decay_period
if opt.lr_decay_period > 0:
    lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
else:
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]
num_batches = num_training_samples // batch_size
lr_scheduler = LRSequential([
    LRScheduler('linear', base_lr=0, target_lr=opt.lr,
                nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
    LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                nepochs=opt.num_epochs - opt.warmup_epochs,
                iters_per_epoch=num_batches,
                step_epoch=lr_decay_epoch,
                step_factor=lr_decay, power=2)
])

# optimizer = 'sgd'
# optimizer_params = {'wd': opt.wd, 'momentum': 0.9, 'lr_scheduler': lr_scheduler}
optimizer = 'adam'
optimizer_params = {'wd': opt.wd, 'lr_scheduler': lr_scheduler}
if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True

save_frequency = opt.save_frequency
if opt.save_dir and save_frequency:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_frequency = 0

def train(ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    if opt.use_pretrained_base:
        net.deconv_layers.initialize(ctx=ctx)
        net.final_layer.initialize(ctx=ctx)
    else:
        net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

    L = gluon.loss.L2Loss()
    metric = HeatmapAccuracy()

    best_val_score = 1

    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)

    for epoch in range(opt.num_epochs):
        loss_val = 0
        tic = time.time()
        btic = time.time()
        metric.reset()

        for i, batch in enumerate(train_data):
            data, label, weight, imgid = train_batch_fn(batch, ctx)

            with ag.record():
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                loss = [nd.cast(L(nd.cast(yhat, 'float32'), y, w), opt.dtype)
                        for yhat, y, w in zip(outputs, label, weight)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)

            metric.update(label, outputs)

            loss_val += sum([l.mean().asscalar() for l in loss]) / num_gpus
            if opt.log_interval and not (i+1)%opt.log_interval:
                metric_name, metric_score = metric.get()
                logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\tloss=%f\tlr=%f\t%s=%.3f'%(
                             epoch, i, batch_size*opt.log_interval/(time.time()-btic),
                             loss_val / (i+1), trainer.learning_rate, metric_name, metric_score))
                btic = time.time()

        time_elapsed = time.time() - tic
        logger.info('Epoch[%d]\t\tSpeed: %d samples/sec over %d secs\tloss=%f\n'%(
                     epoch, int(i*batch_size / time_elapsed), int(time_elapsed), loss_val / (i+1)))
        if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
            net.save_parameters('%s/%s-%d.params'%(save_dir, model_name, epoch))
            trainer.save_states('%s/%s-%d.states'%(save_dir, model_name, epoch))

    if save_frequency and save_dir:
        net.save_parameters('%s/%s-%d.params'%(save_dir, model_name, opt.num_epochs-1))
        trainer.save_states('%s/%s-%d.states'%(save_dir, model_name, opt.num_epochs-1))

    return net

def main():
    net = train(context)

if __name__ == '__main__':
    main()
