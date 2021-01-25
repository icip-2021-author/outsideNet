import sys
import os
import logging
import re
import functools
import collections
import torch
import fnmatch
import numpy as np
import torch.nn as nn
from math import ceil


def async_copy_to(obj, dev, main_stream=None):
    if torch.is_tensor(obj):
        v = obj.cuda(dev, non_blocking=True)
        if main_stream is not None:
            v.data.record_stream(main_stream)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(o, dev, main_stream) for k, o in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [async_copy_to(o, dev, main_stream) for o in obj]
    else:
        return obj

def setup_logger(distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    return logger


def find_recursive(root_dir, extensions=('jpg', 'png'), names_only=False):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for ext in extensions:
            for filename in fnmatch.filter(filenames, '*' + ext):
                if names_only:
                    files.append(filename)
                else:
                    files.append(os.path.join(root, filename))
    return files


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def color_encode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        if label == 255:
            label = -1            
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label + 1],
                    (labelmap.shape[0], labelmap.shape[1], 1)).astype(np.uint8)

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersection_and_union(im_pred, im_label, num_class):
    im_label = np.asarray(im_label).copy()
    im_pred = np.asarray(im_pred).copy()

    im_pred += 1
    im_label += 1
    # Remove classes from unlabeled pixels in gt image.
    im_pred = im_pred * (im_label > 0)

    # Compute intersection:
    intersection = im_pred * (im_pred == im_label)
    (area_intersection, _) = np.histogram(
        intersection, bins=num_class, range=(1, num_class))

    # Compute union:
    (area_pred, _) = np.histogram(im_pred, bins=num_class, range=(1, num_class))
    (area_lab, _) = np.histogram(im_label, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


class NotSupportedCliException(Exception):
    pass


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end + 1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):

    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "{}"'.format(d))
    return ret


def create_spatial_mask(size=(10, 10), shape=(3, 3)):
    w, h = size
    max_val = (w * h) - 1
    mask = np.zeros((h, w, 1))
    h_b, w_b = (ceil(h / shape[0]), ceil(w / shape[1]))
    val = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            mask[i * h_b:(i + 1) * h_b, j * w_b:(j + 1) * w_b] = val / max_val
            val += 1
    return mask

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None, align_corners=False):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.align_corners = align_corners
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = nn.functional.interpolate(input=score, size=(h, w), mode='bilinear',
                                              align_corners=self.align_corners)
        loss = self.criterion(score, target)
        return loss

    def forward(self, score, target):
        weights = [0.4, 1]
        assert len(weights) == len(score)
        combined_loss = [w * self._forward(x, target) for (w, x) in zip(weights, score)]
        # print(combined_loss)
        return sum(combined_loss)
