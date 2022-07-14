import numpy as np
import os
import csv
import cv2
import logging 
from pycocotools import mask as mutils
from mmcv import Config
import torch
import os
from mmdet.datasets.registry import DATASETS
import pickle
from mmdet.datasets.pipelines import Compose
from mmdet.datasets.custom import CustomDataset

@DATASETS.register_module
class DeepLesionDataset25d(CustomDataset):

    CLASSES = ('lesion')
    def __init__(self, 
                 ann_file, 
                 pipeline,
                 pre_pipeline,
                 dicm2png_cfg,
                 data_root=None, 
                 image_path='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False):
        self.data_path = data_root
        self.classes = ['__background__', 'lesion']
        self.num_classes = len(self.classes)
        self.load_annotations(ann_file)
        self.img_ids = [a['filename'] for a in self.ann]
        self.cat_ids = self.classes
        # self.image_fn_list, self.lesion_idx_grouped = self.load_split_index()
        # self.num_images = len(self.image_fn_list)
        self.cfg = Config(dicm2png_cfg)
        self.pipeline = Compose(pipeline)
        self.pre_pipeline = Compose(pre_pipeline)
        self.img_path = image_path
        self.seg_prefix = seg_prefix
        self.proposals = None
        if proposal_file is not None:
            self.proposals = None
        self.slice_num = self.cfg.NUM_SLICES
        self.is_train = not test_mode
        
        if self.is_train:
            self._set_group_flag()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, info).
        """
        # print(index)
        ann = self.ann[index]
        image_fn = ann['filename']
        boxes = np.array(ann['ann']['bboxes'], dtype=np.float32)
        masks=  np.array([mutils.decode(m) for m in ann['ann']['masks']], dtype=np.float32).transpose((1,2,0))
        # masks = masks.sum(0)>0
        slice_intv = ann['ann']['slice_intv']
        spacing = ann['ann']['spacing']
        label = ann['ann']['labels']
        recists = ann['ann']['recists']
        diameters = ann['ann']['diameters']
        gender = float(ann['ann']['gender'])
        age = float(ann['ann']['age'])
        z_coord = float(ann['ann']['z_coord'])

        im, im_scale = load_prep_img(self.img_path, image_fn, spacing, slice_intv,
                                           self.cfg, num_slice=self.slice_num, is_train=self.is_train)

        # im -= self.cfg.PIXEL_MEAN
        boxes = self.clip_to_image(boxes, im, False)

        masks = masks.transpose((2, 0, 1))
        boxes = boxes.astype(np.float32)
        results = dict()#img_info=ann, ann_info=infos
        results['filename'] = image_fn
        # results['flage'] = flage
        infos = {'recists': recists,
                 'diameters': diameters,
                 'spacing': spacing,
                 'thickness':slice_intv
                }
        # results['flage'] = flage
        results['img'] = im
        results['img_shape'] = im.shape
        results['ori_shape'] = im.shape#[ann['height'], ann['width']]
        if self.proposals is not None:
            results['proposals'] = self.proposals[index]

        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['gt_bboxes'] = boxes
        results['bbox_fields'].append('gt_bboxes')
        results['gt_labels'] = label.astype(np.int64)
        results['gt_masks'] = masks
        results['mask_fields'].append('gt_masks')
        results['thickness'] = slice_intv
        results = self.pre_pipeline(results)
        # results['gt_masks'] = masks_scaled
        # results['mask_fields'].append('gt_masks')
        
        return self.pipeline(results)


    def __len__(self):
        return len(self.ann)
        # return 160
    def clip_to_image(self, bbox, img, remove_empty=True):
        TO_REMOVE = 1
        bbox[:, 0] = bbox[:, 0].clip(min=0, max=img.shape[1] - TO_REMOVE)
        bbox[:, 1] = bbox[:, 1].clip(min=0, max=img.shape[0] - TO_REMOVE)
        bbox[:, 2] = bbox[:, 2].clip(min=0, max=img.shape[1] - TO_REMOVE)
        bbox[:, 3] = bbox[:, 3].clip(min=0, max=img.shape[0] - TO_REMOVE)
        if remove_empty:
            box = bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return bbox[keep]
        return bbox

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        if not self.cfg.GROUNP_ZSAPACING: return
        for i in range(len(self)):
            img_info = self.ann[i]
            if img_info['ann']['slice_intv'] < 2.0:
                self.flag[i] = 1
        logging.info(f'slice_intv grounped by 2.0: {sum(self.flag)}/{len(self)-sum(self.flag)}')

    def load_annotations(self, ann_file):
        """load annotations and meta-info from DL_info.csv"""
        with open(ann_file,'rb') as f:
            self.ann = pickle.load(f)
    


def load_prep_img(data_dir, imname, spacing, slice_intv, cfg, num_slice=3, is_train=False):
    """load volume, windowing, interpolate multiple slices, clip black border, resize according to spacing"""
    im = load_multislice_img_16bit_png(data_dir, imname, slice_intv, num_slice, norm_slice_intv=cfg.SLICE_INTV)

    im = windowing(im, cfg.WINDOWING)
    im_shape = im.shape[0:2]
    im_scale = 1.0

    return im, im_scale

def load_multislice_img_16bit_png(data_dir, imname, slice_intv, num_slice, norm_slice_intv):
    data_cache = {}
    def _load_data_from_png(imname, delta=0):
        imname1 = get_slice_name(data_dir, imname, delta)
        if imname1 not in data_cache.keys():
            data_cache[imname1] = cv2.imread(os.path.join(data_dir, imname1), -1)
            assert data_cache[imname1] is not None, 'file reading error: ' + imname1
            # if data_cache[imname1] is None:
            #     print('file reading error:', imname1)
        return data_cache[imname1]

    _load_data = _load_data_from_png
    im_cur = _load_data(imname)


    if norm_slice_intv == 0 or np.isnan(slice_intv) or slice_intv < 0:
        ims = [im_cur] * num_slice  # only use the central slice

    else:
        ims = [im_cur]
        # find neighboring slices of im_cure
        rel_pos = float(norm_slice_intv) / slice_intv
        a = rel_pos - np.floor(rel_pos)
        b = np.ceil(rel_pos) - rel_pos
        if a == 0:  # required SLICE_INTV is a divisible to the actual slice_intv, don't need interpolation
            for p in range(int((num_slice-1)/2)):
                im_prev = _load_data(imname, - rel_pos * (p + 1))
                im_next = _load_data(imname, rel_pos * (p + 1))
                ims = [im_prev] + ims + [im_next]
        else:
            for p in range(int((num_slice-1)/2)):
                intv1 = rel_pos*(p+1)
                slice1 = _load_data(imname, - np.ceil(intv1))
                slice2 = _load_data(imname, - np.floor(intv1))
                im_prev = a * slice1 + b * slice2  # linear interpolation

                slice1 = _load_data(imname, np.ceil(intv1))
                slice2 = _load_data(imname, np.floor(intv1))
                im_next = a * slice1 + b * slice2

                ims = [im_prev] + ims + [im_next]

    ims = [im.astype(float) for im in ims]
    im = cv2.merge(ims)
    im = im.astype(np.float32,
                       copy=False) - 32768  # there is an offset in the 16-bit png files, intensity - 32768 = Hounsfield unit


    return im


def get_slice_name(data_dir, imname, delta=0):
    """Infer slice name with an offset"""
    if delta == 0:
        return imname
    delta = int(delta)
    dirname, slicename = imname.split(os.sep)
    slice_idx = int(slicename[:-4])
    imname1 = '%s%s%03d.png' % (dirname, os.sep, slice_idx + delta)

    # if the slice is not in the dataset, use its neighboring slice
    while not os.path.exists(os.path.join(data_dir, imname1)):
        # print('file not found:', imname1)
        delta -= np.sign(delta)
        imname1 = '%s%s%03d.png' % (dirname, os.sep, slice_idx + delta)
        if delta == 0:
            break

    return imname1


def windowing(im, win):
    """scale intensity from win[0]~win[1] to float numbers in 0~255"""
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    im1 -= 50
    return im1


def windowing_rev(im, win):
    """backward windowing"""
    im1 = im.astype(float)#/255
    im1 *= win[1] - win[0]
    im1 += win[0]
    return im1


# def get_mask(im):
#     """use a intensity threshold to roughly find the mask of the body"""
#     th = 32000  # an approximate background intensity value
#     mask = im > th
#     mask = binary_opening(mask, structure=np.ones((7, 7)))  # roughly remove bed
#     # mask = binary_dilation(mask)
#     # mask = binary_fill_holes(mask, structure=np.ones((11,11)))  # fill parts like lung

#     if mask.sum() == 0:  # maybe atypical intensity
#         mask = im * 0 + 1
#     return mask.astype(dtype=np.int32)

     

def get_range(mask, margin=0):
    """Get up, down, left, right extreme coordinates of a binary mask"""
    idx = np.nonzero(mask)
    u = max(0, idx[0].min() - margin)
    d = min(mask.shape[0] - 1, idx[0].max() + margin)
    l = max(0, idx[1].min() - margin)
    r = min(mask.shape[1] - 1, idx[1].max() + margin)
    return [u, d, l, r]


def map_box_back(boxes, cx=0, cy=0, im_scale=1.):
    """Reverse the scaling and offset of boxes"""
    boxes /= im_scale
    boxes[:, [0,2]] += cx
    boxes[:, [1,3]] += cy
    return boxes

def crop_or_pading(img, fixsize):
    h,w,c = img.shape
    fh,fw = fixsize
    mh,mw = max(h, fh),max(w,fw)
    img_new = np.zeros((mh,mw,c))
    img_new[(mh-h)//2:(mh+h)//2, (mw-w)//2:(mw+w)//2, :] = img

    return img_new[(mh-fh)//2:(mh+fh)//2, (mw-fw)//2:(mw+fw)//2,:], [(mh-h)//2-(mh-fh)//2, (mw-w)//2-(mw-fw)//2]