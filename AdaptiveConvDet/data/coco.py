from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import pickle

COCO_ROOT = '/media/workserv/498ee660-1fc8-40e8-bb02-f0a626cbfe93/ccl/COCO/'
IMAGES = 'images'
ANNOTATIONS = 'annotations'
COCO_API = 'cocoapi/PythonAPI'
INSTANCES_SET = 'instances_{}.json'
TEST_SET = 'image_info_{}.json'

# Note: coco2014 classes is different with coco2017
# coco2014
# COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#                 'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
#                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
#                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                 'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
#                 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                 'teddy bear', 'hair drier', 'toothbrush')

# coco2017
COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush')



def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, root, image_set='train2017', transform=None,
                 target_transform=COCOAnnotationTransform(), dataset_name='MS COCO'):
        sys.path.append(osp.join(root, COCO_API))
        # sys.path.append(COCO_API)
        from pycocotools.coco import COCO
        self.root = osp.join(root, IMAGES, image_set)
        if 'test' not in image_set:
            self.reset_path = osp.join(root, ANNOTATIONS, INSTANCES_SET.format(image_set))
            self.coco = COCO(self.reset_path)
            self.ids = list(self.coco.imgToAnns.keys())
        else:
            self.reset_path = osp.join(root, ANNOTATIONS, TEST_SET.format(image_set))
            self.coco = COCO(self.reset_path)
            self.ids = self.coco.getImgIds()
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.coco_name = image_set

        cats = self.coco.loadCats(self.coco.getCatIds())
        self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats],
                                              self.coco.getCatIds()))

        indexes = self.coco.getImgIds()
        self.image_indexes = indexes
        self._classes = tuple(['__background__'] + [c['name'] for c in cats])
        self.num_classes = len(self._classes)

        self.devkit = root


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        if 'test' not in self.coco_name:
            img_id = self.ids[index]
            target = self.coco.imgToAnns[img_id]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

            target = self.coco.loadAnns(ann_ids)
            path = osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
            assert osp.exists(path), 'Image path does not exist: {}'.format(path)
            img = cv2.imread(osp.join(self.root, path))
            # src_img = img
            height, width, _ = img.shape
            if self.target_transform is not None:
                target = self.target_transform(target, width, height)
            if self.transform is not None:
                target = np.array(target)
                img, boxes, labels = self.transform(img, target[:, :4],
                                                    target[:, 4])
                # to rgb
                img = img[:, :, (2, 1, 0)]

                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        else:
            img_id = self.ids[index]
            target = None
            path = osp.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
            if 'test-dev2017' in path:
                path = path.replace('test-dev2017', 'test2017')
            assert osp.exists(path), 'Image path does not exist: {}'.format(path)
            img = cv2.imread(osp.join(self.root, path))
            # src_img = img
            height, width, _ = img.shape
            if self.transform is not None:
                img, boxes, labels = self.transform(img)
                # to rgb
                img = img[:, :, (2, 1, 0)]
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                             self.num_classes))
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                           coco_cat_id))
            '''
            if cls_ind ==30:
                res_f = res_file+ '_1.json'
                print('Writing results json to {}'.format(res_f))
                with open(res_f, 'w') as fid:
                    json.dump(results, fid)
                results = []
            '''
        # res_f2 = res_file+'_2.json'
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            # results = json.dumps(results)
            json.dump(results, fid)

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.ids):
            # dets = boxes[im_ind].astype(np.float)
            dets = boxes[im_ind]
            if dets == []:
                continue
            dets = dets.astype(np.float)
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
                [{'image_id': index,
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _do_detection_eval(self, res_file, output_dir, model_name='ssd300_vgg'):
        # sys.path.append(osp.join(self.root, COCO_API))
        self.reset()
        from pycocotools.cocoeval import COCOeval

        imgIds = sorted(self.coco.getImgIds())
        imgIds = imgIds[0:100]
        # imgId = imgIds[np.random.randint(100)]

        ann_type = 'bbox'
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, ann_type)
        coco_eval.params.imgIds = imgIds
        # coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        eval_file = os.path.join(output_dir, 'eval.txt')
        stat = coco_eval.stats
        eval_write(eval_file, model_name, stat)
        # print(type(stat))
        # print(stat)

    def evaluate_detections(self, all_boxes, output_dir, model_name='ssd300_vgg'):
        res_file = os.path.join(output_dir, ('detections_' +
                                             self.coco_name +
                                             '_results'))
        res_file += '.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self.coco_name.find('test') == -1:
            print('Eval on results:{}'.format(res_file))
            self._do_detection_eval(res_file, output_dir, model_name)
            # Optionally cleanup results json file

    def reset(self):
        # sys.path.append(osp.join(self.devkit, COCO_API))
        from pycocotools.coco import COCO
        print('COCO dataset reset!')
        self.coco = COCO(self.reset_path)


def eval_write(result_file, res_name, stat):
    f = open(result_file, 'a+')
    f.write(res_name + ':')
    for i in range(12):
        # f.write(str(stat[i]) + ' ')
        f.write('{:.3f}, '.format(stat[i]))
    f.write('\n')
    f.close()