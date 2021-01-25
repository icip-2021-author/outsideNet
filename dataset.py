import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
from torchvision import transforms, utils

from utils import create_spatial_mask


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # parse options
        self.img_sizes = opt.img_sizes
        self.img_max_size = opt.img_max_size
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.parse_input_list(odgt, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.normalizev2 = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.5],
            std=[0.229, 0.224, 0.225, 0.5])

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            with open(odgt, 'r') as listFile:
                self.list_sample = json.load(listFile)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# input images: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def img_transform_v2(self, img, mask):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        mask = np.float32(mask)
        img = np.append(img, mask, axis=2)
        img = img.transpose((2, 0, 1))
        img = self.normalizev2(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to num_class-1
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, batch_per_gpu=1, spatial_mask=False, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        # down sampling rate of segmentation label
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu
        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]
        self.cur_idx = 0
        self.if_shuffled = False
        self.rand_flip = opt.random_flip
        self.rand_crop = opt.random_crop
        self.spatial_mask = spatial_mask

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample)  # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample)  # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def rand_scale_crop(self, img, segm, batch_size, mask=None,):
        # find possible scales
        scales = np.arange(2, min(img.size / batch_size), 0.5)
        # choose scale
        scale = np.random.choice(scales)
        batch_size = (batch_size * scale).astype(int)
        # select corners
        left = np.random.randint(0, img.size[0] - batch_size[0])
        top = np.random.randint(0, img.size[1] - batch_size[1])
        # crop image and label
        img = img.crop((left, top, left + batch_size[0], top + batch_size[1]))
        segm = segm.crop((left, top, left + batch_size[0], top + batch_size[1]))
        if mask:
            mask = mask[top:(top + batch_size[0]), left:(left + batch_size[1])]
            return img, segm, mask
        else:
            return img, segm

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            #  np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.img_sizes, list) or isinstance(self.img_sizes, tuple):
            this_short_size = np.random.choice(self.img_sizes)
        else:
            this_short_size = self.img_sizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width),
                self.img_max_size / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsampling rate'
        batch_images = torch.zeros(self.batch_per_gpu, 3 + self.spatial_mask, batch_height, batch_width)
        batch_segms = torch.zeros(self.batch_per_gpu,
                                  batch_height // self.segm_downsampling_rate,
                                  batch_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]
            batch_size = np.array([batch_widths[i], batch_heights[i]])
            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

            img = Image.open(image_path).convert('RGB')
            segm = Image.open(segm_path)
            assert(segm.mode == "L"), 'Exception: segmentation file {} is not in mode L'.format(segm_path)
            assert(img.size[0] == segm.size[0])
            assert(img.size[1] == segm.size[1])
            assert(img.size[0] > 0)
            assert(img.size[1] > 0)

            # creating spatial_mask mask
            if self.spatial_mask:
                mask = create_spatial_mask(img.size, (3, 1))
                assert(img.size[0] == mask.shape[1])
                assert(img.size[1] == mask.shape[0])
            # random_crop
            if (self.rand_crop and max(img.size) > max(batch_size) * 2
                    and np.random.choice([0, 1], p=[0.7, 0.3])):
                if self.spatial_mask:
                    img, segm, mask = self.rand_scale_crop(img, segm, batch_size, mask)
                else:
                    img, segm = self.rand_scale_crop(img, segm, batch_size)
            # random_flip
            if self.rand_flip and np.random.choice([0, 1], p=[0.7, 0.3]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
                if self.spatial_mask:
                    mask = np.flip(mask, 1)
            assert(img.size[0] > 0)
            assert(img.size[1] > 0)
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')
            assert(img.size[0] == segm.size[0])
            assert(img.size[1] == segm.size[1])
            assert(img.size[0] > 0)
            assert(img.size[1] > 0)
            if self.spatial_mask:
                mask = zoom(mask.squeeze(), [batch_heights[i] / mask.shape[0],
                            batch_widths[i] / mask.shape[1]], mode='nearest')
                mask = np.expand_dims(mask, 2)
                assert(img.size[0] == mask.shape[1])
                assert(img.size[1] == mask.shape[0])

            #  further downsample seg label, to avoid seg label misalignment during loss calcualtion
            segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(segm_rounded,
                            (segm_rounded.size[0] // self.segm_downsampling_rate,
                             segm_rounded.size[1] // self.segm_downsampling_rate),
                            interp='nearest')

            if self.spatial_mask:
                # image transform, to torch float tensor 4xHxW
                img = self.img_transform_v2(img, mask)
            else:
                # image transform, to torch float tensor 3xHxW
                img = self.img_transform(img)
            # segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)
            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return self.num_sample


class TestDataset(BaseDataset):
    def __init__(self, odgt, opt, spatial_mask=False, multi_scale=False, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)
        self.spatial_mask = spatial_mask
        self.multi_scale = multi_scale

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image
        image_path = this_record['fpath_img']
        img = Image.open(image_path).convert('RGB')

        ori_width, ori_height = img.size
        if self.multi_scale:
            img_resized_list = []
            for this_short_size in self.img_sizes:
                # calculate target height and width
                scale = min(this_short_size / float(min(ori_height, ori_width)),
                            self.img_max_size / float(max(ori_height, ori_width)))
                target_height, target_width = int(ori_height * scale), int(ori_width * scale)
                print('scale: {}, height: {}, width: {}'.format(scale, target_height, target_width))
                # to avoid rounding in network
                target_width = self.round2nearest_multiple(target_width, self.padding_constant)
                target_height = self.round2nearest_multiple(target_height, self.padding_constant)

                # resize images
                img_resized = imresize(img, (target_width, target_height), interp='bilinear')
                if self.spatial_mask:
                    mask = create_spatial_mask((target_width, target_height))
                    assert(img_resized.size[0] == mask.shape[1])
                    assert(img_resized.size[1] == mask.shape[0])
                    img = self.img_transform_v2(img_resized, mask)
                else:
                    # image transform, to torch float tensor 3xHxW
                    img_resized = self.img_transform(img_resized)
                img_resized = torch.unsqueeze(img_resized, 0)
                img_resized_list.append(img_resized)
            output = dict()
            output['img_ori'] = np.array(img)
            output['img_data'] = [x.contiguous() for x in img_resized_list]
            output['info'] = this_record['fpath_img']
        else:
            if self.spatial_mask:
                mask = create_spatial_mask((img.size[1], img.size[0]))
                assert(img.size[0] == mask.shape[1])
                assert(img.size[1] == mask.shape[0])
                _img = self.img_transform_v2(img, mask)
            else:
                # image transform, to torch float tensor 3xHxW
                _img = self.img_transform(img)
            print('before: {}'.format(_img.size()))
            _img = torch.unsqueeze(_img, 0)
            print('after: {}'.format(_img.size()))
            output = dict()
            output['img_ori'] = np.array(img)
            output['img_data'] = _img.contiguous()
            output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample

class StatsDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, spatial_mask=False, **kwargs):
        super(StatsDataset, self).__init__(odgt, opt, **kwargs)
        self.spatial_mask = spatial_mask
        self.root_dataset = root_dataset

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        segm = Image.open(segm_path)
        assert(segm.mode == "L")
        segm_ = imresize(segm, (100, 100), interp='nearest')
        return np.array(segm), np.array(segm_)

    def __len__(self):
        return self.num_sample
