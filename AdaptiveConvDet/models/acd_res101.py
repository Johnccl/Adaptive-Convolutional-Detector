import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc_321, coco_321, voc_513, coco_513
import os
import time
from resnet import resnet101, Bottleneck

"""
    Corresponding to ssd_res101_v2.py
"""


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
        base: VGG16 layers for input, size of either 321 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(ACD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # self.cfg = (coco, voc)[num_classes == 21]
        if num_classes==21:
            if size==513:
                self.cfg = voc_513
            else:
                self.cfg = voc_321
        else:
            if size == 513:
                self.cfg = coco_513
            else:
                self.cfg = coco_321
        self.cfg['aspect_ratios'][0] = [2, 3]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # ACD network
        # self.base = nn.ModuleList(base)
        self.base_stage1 = nn.Sequential(base.conv1, base.bn1, base.relu,
                                         base.maxpool, base.layer1, base.layer2)
        self.base_stage2 = base.layer3
        self.base_stage3 = base.layer4
        # Layer learns to scale the l2 normalized features from conv4_3
        base.inplanes = 512
        self.expand_stage1 = base._make_layer(Bottleneck, 128, 3, 1)
        self.L2Norm3 = L2Norm(512, 20)
        self.L2Norm4 = L2Norm(1024, 16)
        #self.L2Norm5 = L2Norm(2048, 15)
        # self.extras = nn.ModuleList(extras)
        self.extras = nn.ModuleList(extras[0])
        self.finals = nn.ModuleList(extras[1])
        self.ACblock = nn.ModuleList(head[2])

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        # self.ACblock = nn.ModuleList(head[2])

        # if phase == 'test':
        self.softmax = nn.Softmax(dim=-1)

        if size==513:
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
            x: input image or batch of images. Shape: [batch,3,321,321].

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
        norms = []

        # apply base up to conv3_4 relu
        x = self.base_stage1(x)
        # norms.append(self.get_norm(x))
        s = self.L2Norm3(x)
        s = self.expand_stage1(s)
        sources.append(s)

        # apply base up to conv4_23 relu
        x = self.base_stage2(x)
        # norms.append(self.get_norm(x))
        s = self.L2Norm4(x)
        sources.append(s)

        # apply base up to conv5_3 relu
        x = self.base_stage3(x)
        # norms.append(self.get_norm(x))
        # s = self.L2Norm5(x)
        # sources.append(s)

        for k, v in enumerate(self.extras):
            x = v(x)
        # norms.append(self.get_norm(x))
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.finals):
            x = v(x)
            if k % 6 == 5:
                sources.append(x)
                # norms.append(self.get_norm(x))
        
        # print('Norms:{}'.format(norms))
        # apply multibox head to source layers
        num_block = len(self.ACblock)
        count = 0
        for (x, l, c) in zip(sources, self.loc, self.conf):
            if count < num_block:
                x = self.ACblock[count](x)
                count += 1
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        self.timer += time.time() - start_timer
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
        self.init_modules()
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights {} into 3 base stage state dict...'.format(base_file))
            state_dict = torch.load(base_file)
            from collections import OrderedDict

            stage1_state_dict = OrderedDict()
            stage2_state_dict = OrderedDict()
            stage3_state_dict = OrderedDict()
            for k, v in state_dict.items():
                head = k[:7]
                if head == 'module.':
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                if 'fc' in name:
                    continue
                if 'layer3' in name:
                    name = name.replace('layer3.', '')
                    stage2_state_dict[name] = v
                    continue
                if 'layer4' in name:
                    name = name.replace('layer4.', '')
                    stage3_state_dict[name] = v
                    continue
                if 'layer1' in name:
                    name = name.replace('layer1', '4')
                elif 'layer2' in name:
                    name = name.replace('layer2', '5')
                else:
                    if name == 'conv1.weight':
                        name = '0.weight'
                    if name in ['bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var']:
                        name = name.replace('bn', '')
                stage1_state_dict[name] = v
            self.base_stage1.load_state_dict(stage1_state_dict)
            self.base_stage2.load_state_dict(stage2_state_dict)
            self.base_stage3.load_state_dict(stage3_state_dict)
            
            del stage1_state_dict
            del stage2_state_dict
            del stage3_state_dict
            del state_dict
            # det_net.load_state_dict(new_state_dict)

            # self.load_state_dict(torch.load(base_file,
            #                      map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def timer_reset(self):
        self.timer = 0

    def get_norm(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()
        mean = norm.mean().data[0]
        return mean

    def init_modules(self):
        print('Self init with Xavier...')
        import torch.nn.init as init

        def xavier(param):
            init.xavier_uniform(param)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                # m.bias.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant(m.weight, 1)
                nn.init.constant(m.bias, 0)


def add_extras(base, size=321):
    # Extra layers added to VGG for feature scaling
    layers = []
    finals = []
    if size == 321:
        layers += [nn.Conv2d(2048, 512, 1, padding=0, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(512, 128, 1, padding=0, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(128, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(512, 128, 1, padding=0, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(256, 128, 1, padding=0, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(128, 256, 3, padding=0, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(256, 128, 1, padding=0, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(128, 256, 3, padding=0, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
    else:
        layers += [nn.Conv2d(2048, 512, 1, padding=0, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(512, 128, 1, padding=0, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(128, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(512, 128, 1, padding=0, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(256, 128, 1, padding=0, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(256, 128, 1, padding=0, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(128, 256, 3, padding=0, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(256, 128, 1, padding=0, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
        finals += [nn.Conv2d(128, 256, 4, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
    return (layers, finals)


def multibox(base, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    ACblocks = []

    for c in cfg:
        loc_layers += [nn.Conv2d(256, c * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, c * num_classes, kernel_size=3, padding=1)]

    if len(cfg) == 6:
        ACblocks += [ACBlock(512, 4, pool_size=4, size=3)]
        ACblocks += [ACBlock(1024, 4, pool_size=4, size=3)]
        ACblocks += [ACBlock(512, 4, pool_size=4, size=3)]
        ACblocks += [ACBlock(256, 4, pool_size=2, size=1)]
        ACblocks += [ACBlock(256, 4, pool_size=1, size=1)]
    else:
        ACblocks += [ACBlock(512, 4, pool_size=4, size=3, groups=32)]
        ACblocks += [ACBlock(1024, 4, pool_size=4, size=3, groups=32)]
        ACblocks += [ACBlock(512, 4, pool_size=4, size=3, groups=32)]
        ACblocks += [ACBlock(256, 4, pool_size=4, size=3, groups=32)]
        ACblocks += [ACBlock(256, 4, pool_size=2, size=3, groups=32)]
        ACblocks += [ACBlock(256, 4, pool_size=1, size=1, groups=32)]
    return base, (loc_layers, conf_layers, ACblocks)


mbox = {
    '321': [6, 6, 6, 6, 6, 4],  # number of boxes per feature map location
    '513': [6, 6, 6, 6, 6, 6, 4],
}


def build_net(phase, size=321, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    base_, head_ = multibox(resnet101(),
                                     mbox[str(size)], num_classes)
    extras_ = add_extras(base_, size=size)
    return ACD(phase, size, base_, extras_, head_, num_classes)
