# Adaptive Convolutional Detector
Forked from [PyTorch-SSD](https://github.com/amdegroot/ssd.pytorch), which is a [PyTorch](http://pytorch.org/) implementation of Single Shot Multibox Detector.

Source code and models will be opened soon!

If you have any questions, please contact with me:johnchen@mail.ustc.edu.cn

### Table of Contents
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-ssd'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#performance'>Performance</a>
- <a href='#demos'>Demos</a>
- <a href='#todo'>Future Work</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
- Clone this repository.
  * Note: We currently only tested with Python 2.7.
- Run the following command.
  ```Shell
  # change path to this directory
  cd PathToThisDirectory/
  # build nms
  sh make.sh
  ```
- Then download the dataset by following the [instructions](#datasets) below.
- Note: For training, we currently support [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](http://mscoco.org/).

## Datasets
Now we only support for MS COCO and PASCAL VOC.

## Training ACD
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train ACD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them.

```Shell
python train.py
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * Before training, you need check the path to your dataset in '/data/coco.py' or '/data/voc0712.py'.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## Evaluation
To evaluate a trained network:

```Shell
python eval.py
```

You can specify the parameters listed in the `eval.py` file by flagging them or manually changing them.


## Performance

#### VOC2007 Test

##### mAP@0.5

| Method | Train | Test | BaseNet | fps | mAP |
|:-:|:-:|:-:|:-:|:-:|:-:|
| SSD300 | VOC0712 |VOC07|VGG| 46.0 | 77.4% |
| SSD512 | VOC0712 |VOC07|VGG| 19.0 | 79.5% |
| ACD300 | VOC0712 |VOC07|VGG| 43.7 | 79.3% |
| ACD512 | VOC0712 |VOC07|VGG| 20.6 | 81.6% |

#### COCO Test-dev

##### AP@[0.5:0.95]
| Method | Train | Test |BaseNet| AP |
|:-:|:-:|:-:|:-:|:-:|
| SSD300 | trainval35k |test-dev|VGG| 25.1% |
| SSD512 | trainval35k |test-dev|VGG| 28.8% |
| ACD300 | trainval35k |test-dev|VGG| 28.6% |
| ACD512 | trainval35k |test-dev|VGG| 32.3% |
| SSD321 | trainval35k |test-dev|Resnet101| 28.0% |
| SSD513 | trainval35k |test-dev|Resnet101| 31.2% |
| ACD321 | trainval35k |test-dev|Resnet101| 30.8% |
| ACD513 | trainval35k |test-dev|Resnet101| 34.5% |


## Demos

### Use a pre-trained ACD network for detection

#### Download a pre-trained network
- We are trying to provide PyTorch `state_dicts` (dict of weight tensors) of the ACD models trained on different datasets.  
- Currently, we provide the following PyTorch models:
    * ACD300 trained on VOC0712 (newest PyTorch weights)
      - model wil be opened soon!
    * SSD300 trained on VOC0712 (original Caffe weights)
      - https://s3.amazonaws.com/amdegroot-models/ssd_300_VOC0712.pth

### Try the demo notebook
- Make sure you have [jupyter notebook](http://jupyter.readthedocs.io/en/latest/install.html) installed.
- Two alternatives for installing jupyter notebook:
    1. If you installed PyTorch with [conda](https://www.continuum.io/downloads) (recommended), then you should already have it.  (Just  navigate to the ssd.pytorch cloned repo and run):
    `jupyter notebook`

    2. If using [pip](https://pypi.python.org/pypi/pip):

```Shell
# make sure pip is upgraded
pip install --upgrade pip
# install jupyter notebook
pip install jupyter
# Run this inside ssd.pytorch
jupyter notebook
```

- Now navigate to `demo/demo.ipynb` at http://localhost:8888 (by default) and have at it!

## TODO
We have accumulated the following to-do list, which we hope to complete in the near future
- Still to come:
  * [ ] Support for visdom

## Authors
- Chunlin Chen

## References
- Wei Liu, et al. "SSD: Single Shot MultiBox Detector." [ECCV2016](http://arxiv.org/abs/1512.02325).
- [Pytorch-SSD](https://github.com/amdegroot/ssd.pytorch).
