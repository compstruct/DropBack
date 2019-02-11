#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified from https://github.com/mitmul/chainer-cifar10
"""
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import argparse
from functools import partial
import os
import random
import re
import shutil
import time
import numpy as np
import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.datasets import TransformDataset
from chainer.datasets import cifar
import chainer.links as L
from chainer.training import extensions
from chainercv import transforms
from skimage import transform as skimage_transform
from chainer.training.triggers import EarlyStoppingTrigger
from chainer import serializers
import si_prefix
import json

from cifar10 import densenet
from cifar10 import wrn
from cifar10 import vgg
from cifar10 import vgg_vd
from pruning_chainer import pruning

import sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

import dropback

try:
    import cv2 as cv
    USE_OPENCV = True
except ImportError:
    USE_OPENCV = False


def cv_rotate(img, angle):

    if USE_OPENCV:
        img = img.transpose(1, 2, 0) / 255.
        center = (img.shape[0] // 2, img.shape[1] // 2)
        r = cv.getRotationMatrix2D(center, angle, 1.0)
        img = cv.warpAffine(img, r, img.shape[:2])
        img = img.transpose(2, 0, 1) * 255.
        img = img.astype(np.float32)
    else:
        # scikit-image's rotate function is almost 7x slower than OpenCV
        img = img.transpose(1, 2, 0) / 255.
        img = skimage_transform.rotate(img, angle, mode='edge')
        img = img.transpose(2, 0, 1) * 255.
        img = img.astype(np.float32)
    return img


def transform(
        inputs, mean, std, random_angle=15., pca_sigma=255., expand_ratio=1.0,
        crop_size=(32, 32), train=True):
    img, label = inputs
    img = img.copy()

    # Random rotate
    if random_angle != 0:
        angle = np.random.uniform(-random_angle, random_angle)
        img = cv_rotate(img, angle)

    # Color augmentation
    if train and pca_sigma != 0:
        img = transforms.pca_lighting(img, pca_sigma)

    # Standardization
    img -= mean[:, None, None]
    img /= std[:, None, None]

    if train:
        # Random flip
        img = transforms.random_flip(img, x_random=True)
        # Random expand
        if expand_ratio > 1:
            img = transforms.random_expand(img, max_ratio=expand_ratio)
        # Random crop
        if tuple(crop_size) != (32, 32):
            img = transforms.random_crop(img, tuple(crop_size))

    return img, label


def create_result_dir(prefix):
    result_dir = 'results/{}_{}_0'.format(
        prefix, time.strftime('%Y-%m-%d_%H-%M-%S'))
    while os.path.exists(result_dir):
        i = result_dir.split('_')[-1]
        result_dir = re.sub('_[0-9]+$', result_dir, '_{}'.format(i))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    shutil.copy(__file__, os.path.join(result_dir, os.path.basename(__file__)))
    return result_dir


def run_training(
        net, train, valid, result_dir, batchsize=64, devices=-1,
        training_epoch=300, initial_lr=0.05, lr_decay_rate=0.5,
        lr_decay_epoch=30, weight_decay=0.005, tracked=0, freeze=0,
        momentum=False, decay_init=False, delay_lr=False, use_pruning=False):
    # Iterator
    train_iter = iterators.SerialIterator(train, batchsize)
    test_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)

    #Init model
    x = net.xp.random.randn(1, 3, 32, 32).astype(np.float32)
    net(x)
    net = L.Classifier(net)
    # Optimizer
    if momentum:
        optimizer = optimizers.MomentumSGD(lr=initial_lr)
    else:
        optimizer = optimizers.SGD(lr=initial_lr)
    optimizer.setup(net)
    if weight_decay > 0:
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

    # Updater
    updater = dropback.DropBack(train_iter, optimizer, result_dir, device=devices, tracked_size=tracked,
                                freeze=freeze, decay_init=decay_init)

    # 6. Trainerz
    trainer = training.Trainer(
        updater, out=result_dir, stop_trigger=(training_epoch, 'epoch'))

    # 7. Trainer extensions
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    if tracked == 0 and use_pruning:
        masks = pruning.create_model_mask(net, args.pruning)
        trainer.extend(pruning.pruned(net, masks))
    trainer.extend(extensions.Evaluator(
        test_iter, net, device=devices), name='val')
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'val/main/loss',
         'val/main/accuracy', 'elapsed_time', 'lr']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'val/main/accuracy'], x_key='epoch',
        file_name='accuracy.png'))
    if delay_lr:
        no_improve = EarlyStoppingTrigger(patients=10, max_trigger=(500, 'epoch'), monitor='val/main/loss')
        trainer.extend(extensions.ExponentialShift(
            'lr', lr_decay_rate), trigger=no_improve)
    else:
        trainer.extend(extensions.ExponentialShift(
            'lr', lr_decay_rate), trigger=(lr_decay_epoch, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=5))
    try:
        trainer.run()
    except KeyboardInterrupt:
        pass
    finally:
        print('{} params.'.format(sum([p.size for p in list(optimizer.target.params())])))
        if tracked:
            print("Compressed {:.2f}".format(sum([p.size for p in list(optimizer.target.params())]) / tracked))
        if decay_init:
            print("Sparsity is {:.2f}".format(float(
                sum([(i.data == 0).sum() for i in list(optimizer.target.params())])/optimizer.target.count_params())))
        print("Eval: ")
        eval = trainer.get_extension('val')()
        print(eval)
    return net


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=78623674)
    parser.add_argument('--extra', default='')

    # Train settings
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--training_epoch', type=int, default=500)
    parser.add_argument('--initial_lr', type=float, default=0.05)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=float, default=25)
    parser.add_argument('--weight_decay', type=float, default=0.0000)
    parser.add_argument('--model', default='vgg', choices=['vgg', 'wrn', 'densenet', 'vgg_vd'])
    parser.add_argument('--use_pruning', default=False, action='store_true')
    parser.add_argument('--momentum', default=False, action='store_true')
    parser.add_argument('--decay_init', default=False, action='store_true')
    parser.add_argument('--delay_lr', default=False, action='store_true')
    parser.add_argument('--tracked', type=int, default=0)
    parser.add_argument('--freeze', default=0, type=int)

    # Data augmentation settings
    parser.add_argument('--random_angle', type=float, default=15.0)
    parser.add_argument('--pca_sigma', type=float, default=25.5)
    parser.add_argument('--expand_ratio', type=float, default=1.2)
    parser.add_argument('--crop_size', type=int, nargs='*', default=[28, 28])
    parser.add_argument('--transform', default=False, action='store_true')

    args = parser.parse_args()
    rdir = f'{args.model}_{si_prefix.si_format(args.tracked).strip()}_{args.extra}'
    try:
        os.makedirs(rdir)
    except Exception as e:
        pass
    finally:
        with open(os.path.join(rdir, 'args.json'), 'w') as arg_log:
            json.dump(vars(args), arg_log)
    # Set the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    train, valid = cifar.get_cifar10(scale=255.)


    if args.model == 'wrn':
        net = wrn.WideResNet(widen_factor=10, depth=28, n_classes=10)
    elif args.model == 'densenet':
        net = densenet.DenseNet(10)
    elif args.model == 'vgg':
        net = vgg.VGG(10)
    elif args.model == 'vd':
        net = vgg_vd.VGG16VD(10, warm_up=0.0001)
        net(train[0][0][None,])  # for setting in_channels automatically
        net.to_variational_dropout()
    if args.gpu > 0:
        # Enable autotuner of cuDNN
        chainer.config.autotune = True
        chainer.cuda.cupy.random.seed(args.seed)
        chainer.cuda.get_device_from_id(args.gpu).use()  # Make the GPU current
        net.to_gpu()

    if args.transform:
        mean = np.mean([x for x, _ in train], axis=(0, 2, 3))
        std = np.std([x for x, _ in train], axis=(0, 2, 3))
        train_transform = partial(
            transform, mean=mean, std=std, random_angle=args.random_angle,
            pca_sigma=args.pca_sigma, expand_ratio=args.expand_ratio,
            crop_size=args.crop_size, train=True)
        valid_transform = partial(transform, mean=mean, std=std, train=False)

        train = TransformDataset(train, train_transform)
        valid = TransformDataset(valid, valid_transform)

    run_training(
        net, train, valid, rdir, args.batchsize, args.gpu,
        args.training_epoch, args.initial_lr, args.lr_decay_rate,
        args.lr_decay_epoch, weight_decay=args.weight_decay,
        tracked=args.tracked, freeze=args.freeze, momentum=args.momentum,
        decay_init=args.decay_init, delay_lr=args.delay_lr, use_pruning=args.use_pruning)
    try:
        serializers.save_npz(os.path.join(rdir, 'cifar10.model'), net)
    except Exception as e:
        pass
