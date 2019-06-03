# -*- coding: utf-8 -*-
from __future__ import division
from data import *
from layers.modules import MultiBoxLoss
from layers.functions.detection import Detect
from utils.augmentations import SSDAugmentation
import os
import sys
import math
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
# should be python 2.7
# from eval import test_net
from eval_dir_coco import test_net


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--models', default='acd', choices=['ssd', 'acd'],
                    type=str, help='ssd or acd')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet',
                    default='weights/vgg16_reducedfc.pth',
                    # default='weights/resnet101_caffe.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder',
                    default='backup/ACD',
                    help='Directory for saving checkpoint models')
parser.add_argument('--exp_name',
                    default='0120_1/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input_size', default=300, type=int,
                    help='input size of images')
parser.add_argument('--warmup', default=4, type=int,
                    help='epoch of warm up')
parser.add_argument('--save_frequency', default=5, type=int,
                    help='frequency for saving model')
parser.add_argument('--test_epoch', default=160, type=int,
                    help='start testing on voc')
parser.add_argument('--test_frequency', default=5)
parser.add_argument('--debug', default=False, type=str2bool)
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

args.save_folder = args.save_folder + '_' + str(args.input_size)
args.save_folder = os.path.join(args.save_folder, args.dataset, args.exp_name)

if args.debug:
    args.save_folder = 'backup/debug/'

print('Save folder:{}'.format(args.save_folder))

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

if args.models == 'ssd':
    from models.ssd import build_net
elif args.models == 'acd':
    from models.acd import build_net
elif args.models == 'acd_res101':
    from models.acd_res101 import build_net
else:
    print('Error! Only support for SSD and ACD!')
    exit()


def train():
    if args.dataset == 'COCO':
        args.dataset_root = COCO_ROOT
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(args.input_size, MEANS))
        test_dataset = COCODetection(COCO_ROOT, 'val2017', BaseTransform(args.input_size, MEANS),
                                     COCOAnnotationTransform())
    elif args.dataset == 'VOC':
        args.dataset_root = VOC_ROOT
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(args.input_size, MEANS))
        test_dataset = VOCDetection(args.dataset_root, [('2007', 'test')],
                                    BaseTransform(args.input_size, MEANS),
                                    VOCAnnotationTransform())
    else:
        print('Only support VOC and COCO dataset!')
        return

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    num_classes = (21, 81)[args.dataset == 'COCO']
    det_net = build_net('train', args.input_size, num_classes)
    net = det_net

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        state_dict = torch.load(args.resume)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        det_net.load_state_dict(new_state_dict)
    else:
        if 'res101' in args.models:
            det_net.load_weights(args.basenet)
        else:
            base_weights = torch.load(args.basenet)
            print('Loading base network...')
            det_net.vgg.load_state_dict(base_weights)
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        det_net.extras.apply(weights_init)
        det_net.loc.apply(weights_init)
        det_net.conf.apply(weights_init)
        if hasattr(det_net, 'ACblock'):
            det_net.ACblock.apply(weights_init)

    if args.cuda:
        # net = torch.nn.DataParallel(det_net, device_ids=[0,1,2,3])
        cudnn.benchmark = True
        net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    # create log files
    log_file = open(args.save_folder + 'log.txt', 'a+')
    args_file = open(args.save_folder + 'args.txt', 'a+')
    args_file.write(str(args) + '\n')
    args_file.close()

    batch_iterator = iter(data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True))

    epoch_size = len(dataset) // args.batch_size

    lr_steps = net.cfg['lr_steps']
    stepvalues = (lr_steps[0] * epoch_size, lr_steps[1] * epoch_size, lr_steps[2] * epoch_size)
    max_epoch = net.cfg['max_epoch']
    max_iter = max_epoch * epoch_size

    print('Step Values:', stepvalues)

    start_iter = 0
    if args.resume is not None:
        start_iter = args.resume_epoch * epoch_size + 1
        epoch = args.resume_epoch

    epoch_timer = time.time()
    tic = time.time()
    print('Start Iter:{}, max iter:{}, epoch size:{}'.format(start_iter, max_iter, epoch_size))
    for iteration in range(start_iter, max_iter + 10):
        if iteration % epoch_size == 0 and iteration > 0:
            epoch += 1
            print('Epoch:{}, time:{}'.format(epoch, time.time() - epoch_timer))
            epoch_timer = time.time()
            # save temp checkpoint every epoch
            tmp_backup_path = os.path.join(args.save_folder, 'ssd' + str(args.input_size) +
                                           '_' + args.dataset + '_backup.pth')
            torch.save(net.state_dict(), tmp_backup_path)

            # save checkpoint
            if epoch % args.save_frequency == 0:
                print('Saving state, epoch:', epoch)
                torch.save(det_net.state_dict(), args.save_folder + 'ssd' +
                           str(args.input_size) + '_' + args.dataset + '_' +
                           str(epoch) + '.pth')

            if epoch >= args.test_epoch and epoch % args.test_frequency == 0:
                # evaluate
                print('Evaluate: epoch-{}'.format(epoch))
                net.eval()
                net.phase = 'test'
                m = 'ssd' + str(args.input_size) + '_' + args.dataset + '_' +\
                    repr(epoch) + '.pth'
                test_net(args.save_folder, net, test_dataset, model_name=m)
                net.phase = 'train'
                net.train()

                batch_iterator = iter(data.DataLoader(dataset, args.batch_size,
                                                      num_workers=args.num_workers,
                                                      shuffle=True, collate_fn=detection_collate,
                                                      pin_memory=True))
            torch.cuda.empty_cache()

        step_index = get_step_index(iteration, stepvalues)
        current_lr = adjust_learning_rate(optimizer, args.gamma, step_index,
                                          epoch, epoch_size, iteration)

        # load train data
        try:
            im_data, im_targets = next(batch_iterator)
        except:
            batch_iterator = iter(data.DataLoader(dataset, args.batch_size,
                                                  num_workers=args.num_workers,
                                                  shuffle=True, collate_fn=detection_collate,
                                                  pin_memory=True))
            im_data, im_targets = next(batch_iterator)

        if args.cuda:
            images = Variable(im_data.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in im_targets]
        else:
            images = Variable(im_data)
            targets = [Variable(ann, volatile=True) for ann in im_targets]

        # t0 = time.time()
        out = net(images)
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        if iteration % 10 == 0:
            toc = time.time()
            print('Timer:{:.4f}sec'.format(toc - tic))
            tic = time.time()
            training_info = 'iter ' + repr(iteration) +\
                            ', epoch: %.4f '%(iteration * 1.0 / epoch_size) +\
                            ', Loss: %.4f ' % (loss.data[0]) \
                            + ', conf_loss: %.4f ' % (loss_c.data[0]) +\
                            ', smoothl1_loss: %.4f ' % (loss_l.data[0]) + \
                            ', learning_rate: %.6f' % (current_lr)
            log_file.write(training_info + '\n')
            print(training_info)

    log_file.close()
    torch.save(det_net.state_dict(), 'ssd' + str(args.input_size) +
               '_' + repr(args.dataset) + '_final.pth')


def get_step_index(cur_iter, step_list):
    step_index = 0
    for s in step_list:
        if cur_iter >= s:
            step_index += 1

    return step_index


def adjust_learning_rate(optimizer, gamma, step, epoch, epoch_size, iteration):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < args.warmup:
        lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * args.warmup)
    else:
        lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        #init.kaiming_normal(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant(m.weight, 1)
        nn.init.constant(m.bias, 0)


if __name__ == '__main__':
    train()
