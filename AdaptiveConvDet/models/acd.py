import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco, voc_512, coco_512
import os
import time


class ACBlock(nn.Module):

    def __init__(self, in_channels, reduction, pool_size=4, out_channels=256,
                 groups=32, size=3, dilation=1):
        super(ACBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction
        self.size = size
        self.dilation = dilation
        self.padding = size//2
        self.groups = groups
        self.pool_size = pool_size
        self.weight_channel = int((self.out_channels**2 * size**2) / (groups * pool_size**2))

        if self.in_channels != self.out_channels:
            self.conv0 = nn.Conv2d(in_channels, self.out_channels,
                                   kernel_size=1, padding=0)
            # self.bn0 = nn.BatchNorm2d(self.out_channels)
        if pool_size == 1:
            kernel_size = 1
        else:
            kernel_size = 3

        self.index_split = int(self.out_channels / (self.pool_size * self.groups))**2

        self.avg_pool = nn.AdaptiveAvgPool2d(pool_size)
        self.conv1 = nn.Conv2d(out_channels, out_channels // reduction,
                               kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels // reduction, self.weight_channel,
                               kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.relu(self.conv0(x))
        num = x.size(0)
        w = self.avg_pool(x)
        w = self.conv1(w)
        # w = self.bn1(w)
        w = self.relu(w)
        w = self.conv2(w)
        # parameters of every group are computed by one specific region
        # example: group=64(g), out=256, weight_channel=64*9(w), pool_size=4(p), ignore(num)
        # (w, p, p)->(w, 16)->(16, w)->(16, 64, 9)->(16*64, 9)->(256, 4, 9)->(256, 4, 3, 3)
        if self.pool_size == 1:
            w = w.view(num, self.out_channels, self.out_channels // self.groups,
                       self.size, self.size)
        else:
            w = w.view(num, self.weight_channel, -1).transpose(2, 1).contiguous()
            w = w.view(num, self.pool_size ** 2, -1, self.size ** 2)
            w = w.view(num, -1, self.size ** 2)
            w = w.view(num, self.out_channels, self.out_channels // self.groups, self.size ** 2)
            w = w.view(num, self.out_channels, self.out_channels // self.groups,
                       self.size, self.size)
        m = []
        for i in range(num):
            tmp_x = F.conv2d(x[i].expand(1, -1, -1, -1), w[i], padding=self.padding,
                             dilation=self.dilation, groups=self.groups)
            m.append(tmp_x.expand(1, -1, -1, -1))

        out = torch.cat(m, 0)
        return self.bn2(self.relu(out))


class ACD(nn.Module):
    """Adaptive Convolutional Detector Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(ACD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        if num_classes==21:
            if size==512:
                self.cfg = voc_512
            else:
                self.cfg = voc
        else:
            if size == 512:
                self.cfg = coco_512
            else:
                self.cfg = coco
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.ACblock = nn.ModuleList(head[2])

        # if phase == 'test':
        self.softmax = nn.Softmax(dim=-1)

        if size==512:
            top_k = (400, 400)[num_classes == 81]
        else:
            top_k = (250, 200)[num_classes == 81]
        print('Top k:{}'.format(top_k))
        self.detect = Detect(num_classes, 0, top_k, 0.01, 0.45)
        self.timer = 0

        # self.timer = 0
        self.only_forward = False


    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        start_timer = time.time()
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        num_block = len(self.ACblock)
        count = 0
        # for (x, l, c, se) in zip(sources, self.loc, self.conf, self.ACblock):
        for (x, l, c) in zip(sources, self.loc, self.conf):
            if count < num_block:
                # x = F.relu(self.ACblock[count](x), inplace=True)
                x = self.ACblock[count](x)
                count += 1
            # x = se(x)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        self.timer += time.time() - start_timer
        if self.only_forward:
            return (loc.view(loc.size(0), -1, 4), self.softmax(conf.view(conf.size(0), -1,
                                                                         self.num_classes)),
                    self.priors.type(type(x.data)))
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def timer_reset(self):
        self.timer = 0

    def get_mid_layer(self, x, n):
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        num_block = len(self.ACblock)
        count = 0
        mid_layers = []
        for (x, l, c) in zip(sources, self.loc, self.conf):
            if count < num_block:
                if n == count:
                    mid_layers.append(x)

                x = self.ACblock[count](x)

                if n == count:
                    mid_layers.append(x)
                count += 1

        return mid_layers


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False, size=300):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v

    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    ACblocks = []
    vgg_source = [21, -2]
    in_channels = []
    for k, v in enumerate(vgg_source):
        in_channels.append(vgg[v].out_channels)
        loc_layers += [nn.Conv2d(256, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        in_channels.append(v.out_channels)
        loc_layers += [nn.Conv2d(256, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, cfg[k] * num_classes, kernel_size=3, padding=1)]

    if len(in_channels)==6:
        ACblocks += [ACBlock(in_channels[0], 4, pool_size=4, size=3)]
        ACblocks += [ACBlock(in_channels[1], 4, pool_size=4, size=3)]
        ACblocks += [ACBlock(in_channels[2], 4, pool_size=4, size=3)]
        ACblocks += [ACBlock(in_channels[3], 4, pool_size=2, size=1)]
        ACblocks += [ACBlock(in_channels[4], 4, pool_size=1, size=1)]
    else:
        # get AP=32.3 on COCO test-dev
        ACblocks += [ACBlock(in_channels[0], 2, pool_size=4, size=3, groups=64)]
        ACblocks += [ACBlock(in_channels[1], 2, pool_size=4, size=3, groups=64)]
        ACblocks += [ACBlock(in_channels[2], 2, pool_size=4, size=3, groups=64)]
        ACblocks += [ACBlock(in_channels[3], 4, pool_size=2, size=3, groups=64)]
        ACblocks += [ACBlock(in_channels[4], 4, pool_size=2, size=3, groups=256)]

        # get AP=32.2 on COCO test-dev
        # ACblocks += [ACBlock(in_channels[0], 4, pool_size=4, size=3)]
        # ACblocks += [ACBlock(in_channels[1], 4, pool_size=4, size=3)]
        # ACblocks += [ACBlock(in_channels[2], 4, pool_size=4, size=3)]
        # ACblocks += [ACBlock(in_channels[3], 4, pool_size=2, size=3)]
        # ACblocks += [ACBlock(in_channels[4], 4, pool_size=1, size=1)]
        # ACblocks += [ACBlock(in_channels[5], 4, pool_size=1, size=1)]
    return vgg, extra_layers, (loc_layers, conf_layers, ACblocks)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    # if size != 300:
    #     print("ERROR: You specified size " + repr(size) + ". However, " +
    #           "currently only SSD300 (size=300) is supported!")
    #     return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024, size=size),
                                     mbox[str(size)], num_classes)
    return ACD(phase, size, base_, extras_, head_, num_classes)
