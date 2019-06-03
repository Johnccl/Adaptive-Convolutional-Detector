"""
"""

from __future__ import print_function
import torch

# If you meet a problem like this:
# It reports an error: 'module' object has no attribute '_rebuild_tensor_v2'
# then add the following code:
import torch._utils
try:
    torch._utils._rebuild_tensor_v2l
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import COCODetection, COCOAnnotationTransform, COCO_ROOT, BaseTransform

import torch.utils.data as data

# from ssd import build_net
from models.acd import build_net

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
import random
import gc
from collections import OrderedDict
from matplotlib import pyplot as plt

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='backup/acd300_VOC.pth',
                    type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--input_size', default=300, type=int,
                    help='input size of images')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--root_dir', default=COCO_ROOT,
                    help='Location of VOC root directory')

args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if args.dataset == 'VOC':
    from data import VOC_CLASSES as labelmap
    args.root_dir = VOC_ROOT
else:
    from data import COCO_CLASSES as labelmap
    args.root_dir = COCO_ROOT

dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def test_net(save_folder, net, dataset, model_name='acd300', retest=False):
    num_images = len(dataset)
    num_classes = dataset.num_classes
    print('number of images:{}, classes:{}'.format(num_images, num_classes))

    if retest:
        dataset.evaluate_detections(None, None, model_name, retest=retest)
        return

    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    start_time = time.time()
    inference_time = 0
    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)
        inference_time += detect_time

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.dim() == 0:
                continue
            if dets.size()[0] == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

        if i%200 == 0 and i > 0:
            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

    total_time = time.time() - start_time
    print('Inference time:{:.4f}, speed:{:.3f}'.format(inference_time, num_images / inference_time))
    print('Forward time:{:.4f}, speed:{:.3f}'.format(net.timer, num_images / net.timer))
    # net.timer_reset()
    print('Total time:{:.4f}, average time:{:.4f}, speed:{:.4f}'.format(total_time,
                                                                        total_time / num_images, num_images / total_time))
    # return all_boxes
    dataset.evaluate_detections(all_boxes, save_folder, model_name)
    # remove local var
    print('Remove local variables...')
    t0 = time.time()
    for x in locals().keys():
        del locals()[x]
    gc.collect()
    print('Delete down! Time cost:{:.4f}'.format(time.time() - t0))

def eval():
    # load net
    num_classes = len(labelmap) + 1  # +1 for background
    root_dir = os.path.dirname(args.trained_model)
    model_name = os.path.basename(args.trained_model)
    # load data
    if args.dataset == 'VOC':
        dataset = VOCDetection(args.root_dir, [('2007', set_type)],
                               BaseTransform(args.input_size, dataset_mean),
                               VOCAnnotationTransform(), only_test=True)
    else:
        dataset = COCODetection(args.root_dir, 'test-dev2017', BaseTransform(args.input_size, dataset_mean),
                            COCOAnnotationTransform())

    net = build_net('test', args.input_size, num_classes)

    state_dict = torch.load(args.trained_model)
    # create new OrderedDict that does not contain `module.`

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        if 'base' in name:
            name = name.replace('base', 'vgg')
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

    net.eval()
    print('Finished loading model:{}'.format(args.trained_model))
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True

    # evaluation
    test_net(root_dir, net, dataset, model_name, retest=False)

if __name__ == '__main__':
    eval()
