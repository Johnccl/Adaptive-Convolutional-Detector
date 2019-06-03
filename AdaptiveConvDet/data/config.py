# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (180, 220, 240),
    'max_epoch': 250,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

voc_512 = {
    'num_classes': 21,
    'lr_steps': (140, 170, 190),
    'max_epoch': 200,
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],
    'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC_512',
}

coco = {
    'num_classes': 81,
    'lr_steps': (140, 160, 170),
    'max_epoch': 175,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

coco_512 = {
    'num_classes': 81,
    'lr_steps': (140, 160, 170),
    'max_epoch': 175,
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],
    'max_sizes': [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

voc_321 = {
    'num_classes': 21,
    'lr_steps': (180, 220, 240),
    'max_epoch': 250,
    'feature_maps': [40, 20, 10, 5, 3, 1],
    'min_dim': 321,
    'steps': [8, 16, 32, 64, 107, 321],
    'min_sizes': [32.1, 64.2, 104.3, 152.5, 207, 269.6],
    'max_sizes': [64.2, 104.3, 152.5, 207, 269.6, 340],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

voc_513 = {
    'num_classes': 21,
    'lr_steps': (140, 170, 190),
    'max_epoch': 200,
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 513,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [35.91, 76.95, 153.9, 230.85, 307.6, 384.75, 461.7],
    'max_sizes': [76.95, 153.9, 230.85, 307.6, 384.75, 461.7, 538.65],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC_512',
}

coco_321 = {
    'num_classes': 81,
    'lr_steps': (140, 160, 170),
    'max_epoch': 175,
    'feature_maps': [40, 20, 10, 5, 3, 1],
    'min_dim': 321,
    'steps': [8, 16, 32, 64, 107, 321],
    'min_sizes': [22.47, 48.15, 96.3, 157.29, 218.28, 279.27],
    'max_sizes': [48.15, 96.3, 157.29, 218.28, 279.27, 340.26],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

coco_513 = {
    'num_classes': 81,
    'lr_steps': (140, 160, 170),
    'max_epoch': 175,
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 513,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],
    'max_sizes': [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}