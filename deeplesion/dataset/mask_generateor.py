import numpy as np
import csv
import cv2
import logging
import os
from pycocotools import mask as mutils
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='generate mask through')
    # parser.add_argument('config', help='train config file path')
    parser.add_argument('--dataroot')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'small'])
    parser.add_argument('--output_filename', default=None)
    args = parser.parse_args()

    return args

def load_prep_img(data_dir, imname, win):
    """load volume, windowing, interpolate multiple slices, clip black border, resize according to spacing"""
    im = load_multislice_img_16bit_png(data_dir, imname)
    im = windowing(im, win)
    return im

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

def load_multislice_img_16bit_png(data_dir, imname):
    data_cache = {}
    def _load_data_from_png(imname, delta=0):
        imname1 = get_slice_name(data_dir, imname, delta)
        if imname1 not in data_cache.keys():
            data_cache[imname1] = cv2.imread(os.path.join(data_dir, imname1), -1)
            assert data_cache[imname1] is not None, 'file reading error: ' + imname1
            # if data_cache[imname1] is None:
            #     print('file reading error:', imname1)
        return data_cache[imname1]

    def _load_data_from_nifti(imname, delta=0):
        # in this case, data_dir is the numpy volume and imname is the slice index
        vol = data_dir
        idx = min(vol.shape[2]-1, max(int(imname+delta), 0))
        return vol[:,:,idx]

    if isinstance(data_dir, str) and isinstance(imname, str):
        _load_data = _load_data_from_png
    elif isinstance(data_dir, np.ndarray) and isinstance(imname, int):
        _load_data = _load_data_from_nifti
    im_cur = _load_data(imname)
    im = im_cur.astype(np.float32,
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


class DeepLesionDataset(object):

    def __init__(
            self, split, data_dir, ann_file, transforms=None
    ):
        self.transforms = transforms
        self.split = split
        self.data_path = data_dir
        self.classes = ['__background__',  # always index 0
                        'lesion']
        self.num_classes = len(self.classes)
        self.loadinfo(ann_file)
        self.image_fn_list, self.lesion_idx_grouped = self.load_split_index()
        self.num_images = len(self.image_fn_list)
        self.logger = logging.getLogger(__name__)

        self.logger.info('DeepLesion %s num_images: %d' % (split, self.num_images))
        self.win = [-1024, 2050]
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, info).
        """
        image_fn = self.image_fn_list[index]
        lesion_idx_grouped = self.lesion_idx_grouped[index]
        boxes0 = self.boxes[lesion_idx_grouped]
        # slice_no = self.slice_idx[lesion_idx_grouped][0]
        slice_intv = self.slice_intv[lesion_idx_grouped][0]
        spacing = self.spacing[lesion_idx_grouped][0]
        recists = self.d_coordinate[lesion_idx_grouped]
        diameters = self.diameter[lesion_idx_grouped]
        window = self.DICOM_window[lesion_idx_grouped][0]
        gender = float(self.gender[lesion_idx_grouped][0] == 'M')
        age = self.age[lesion_idx_grouped][0] / 100
        patient_idx = self.patient_idx[lesion_idx_grouped]
        study_idx = self.study_idx[lesion_idx_grouped]
        series_idx = self.series_idx[lesion_idx_grouped]
        slice_idx = self.slice_idx[lesion_idx_grouped]
        type = self.type[lesion_idx_grouped]

        if np.isnan(age) or age == 0:
            age = .5
        z_coord = self.norm_location[lesion_idx_grouped[0], 2]

        im = load_prep_img(self.data_path, image_fn, self.win)


        boxes = np.array(boxes0, dtype=np.float32)

        # if self.transforms is not None:
        #     im, boxes = self.transforms(im, boxes)
        classes = np.ones(boxes.shape[0], dtype=np.int64)
        masks = self.generate_mask(im, boxes, recists)
        infos = {'labels': classes, 'bboxes': boxes, 'im_index': index,
                 'lesion_idxs': lesion_idx_grouped,
                 'image_fn': image_fn, 'diameters': diameters * spacing,
                 'recists': recists, 'window': window, 'spacing': spacing,
                 'gender': gender, 'age': age, 'z_coord': z_coord,
                 'patient_idx': patient_idx, 'series_idx': series_idx, 'study_idx': study_idx,
                 'slice_idx': slice_idx, 'type': type, "slice_intv":slice_intv, 'masks':masks}
        return {
            'filename': image_fn,
            # 'image': im,
            'width': im.shape[1],
            'height': im.shape[0],
            'ann': infos
        }

    def __call__(self, idx):
        return self.__getitem__(idx)

    def generate_mask(self, image, bboxes, recists):
        # img = ds.__getitem__(index)#['image']
        # image = img['image']
        # bboxes = img["ann"]["bboxes"]
        # recists = img['ann']['recists']
        masks = []
        image1 = np.expand_dims(image, axis=-1).repeat(3,axis=-1).astype(np.uint8)#/image.max()maskmask
    #     print(image1.shape,image.dtype)
        for i in range(bboxes.shape[0]):#bboxes.shape[0]-
            mask = np.zeros_like(image, dtype=np.uint8)[...,np.newaxis]
            x1,y1,x2,y2 = bboxes[i]
            x = int((x1+x2)//2)
            y = int((y1+y2)//2)
            h = int(x2 - x1)
            w = int(y2 - y1)
            pp_mask = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]).astype(np.int32)
            _ = cv2.fillConvexPoly(mask, pp_mask, 3)
            tp_mask = recists[i].astype(np.int32).reshape((4,2))
            tp_mask = tp_mask[[0,2,1,3]]
            _ = cv2.fillConvexPoly(mask, tp_mask, 1)
            _ = cv2.grabCut(image1, mask, (x, y, h, w), None, None, 3, cv2.GC_INIT_WITH_MASK)        
            mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
            masks += mutils.encode(np.asfortranarray(mask))
        # img['ann']['labels'].astype(np.int64)
        # img['ann']['masks'] = masks
        # del img['image']
        return masks

    def __len__(self):
        return len(self.image_fn_list)

    def load_split_index(self):
        """
        need to group lesion indices to image indices, since one image can have multiple lesions
        :return:
        """

        split_list = ['train', 'val', 'test', 'small']
        index = split_list.index(self.split)
        if self.split != 'small':
            lesion_idx_list = np.where((self.train_val_test == index + 1) & ~self.noisy)[0]
        else:
            lesion_idx_list = np.arange(30)
        fn_list = self.filenames[lesion_idx_list]
        fn_list_unique, inv_ind = np.unique(fn_list, return_inverse=True)
        lesion_idx_grouped = [lesion_idx_list[inv_ind == i] for i in range(len(fn_list_unique))]
        return fn_list_unique, lesion_idx_grouped

    def clip_to_image(self, bbox, img, remove_empty=True):
        TO_REMOVE = 1
        bbox[:, 0].clamp_(min=0, max=img.shape[1] - TO_REMOVE)
        bbox[:, 1].clamp_(min=0, max=img.shape[0] - TO_REMOVE)
        bbox[:, 2].clamp_(min=0, max=img.shape[1] - TO_REMOVE)
        bbox[:, 3].clamp_(min=0, max=img.shape[0] - TO_REMOVE)
        if remove_empty:
            box = bbox
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return bbox[keep]
        return bbox

    def loadinfo(self, path):
        """load annotations and meta-info from DL_info.csv"""
        info = []
        with open(path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                filename = row[0]  # replace the last _ in filename with / or \
                idx = filename.rindex('_')
                row[0] = filename[:idx] + os.sep + filename[idx + 1:]
                info.append(row)
        info = info[1:]

        # the information not used in this project are commented
        self.filenames = np.array([row[0] for row in info])
        self.d_coordinate = np.array([[float(x) for x in row[5].split(',')] for row in info])
        self.d_coordinate -= 1
        self.boxes = np.array([[float(x) for x in row[6].split(',')] for row in info])
        self.boxes -= 1  # coordinates in info file start from 1
        self.diameter = np.array([[float(x) for x in row[7].split(',')] for row in info])
        self.norm_location = np.array([[float(x) for x in row[8].split(',')] for row in info])
        self.noisy = np.array([int(row[10]) > 0 for row in info])
        # self.slice_range = np.array([[int(x) for x in row[11].split(',')] for row in info])
        self.spacing3D = np.array([[float(x) for x in row[12].split(',')] for row in info])
        self.spacing = self.spacing3D[:, 0]
        self.slice_intv = self.spacing3D[:, 2]  # slice intervals
        # self.image_size = np.array([[int(x) for x in row[13].split(',')] for row in info])
        self.DICOM_window = np.array([[float(x) for x in row[14].split(',')] for row in info])
        self.gender = np.array([row[15] for row in info])
        self.age = np.array([float(row[16]) for row in info])  # may be NaN
        self.train_val_test = np.array([int(row[17]) for row in info])

        self.patient_idx = np.array([int(row[1]) for row in info])
        self.study_idx = np.array([int(row[2]) for row in info])
        self.series_idx = np.array([int(row[3]) for row in info])
        self.slice_idx = np.array([int(row[4]) for row in info])
        self.type = np.array([int(row[9]) for row in info])




if __name__ == "__main__":

    args = parse_args()
    dataroot = args.dataroot
    split = args.split
    output_filename = args.output_filename if args.output_filename is None else split+"_ann.pkl"

    data_dir, ann_file = dataroot + "Images_png", dataroot + "DL_info.csv"
    transform = None
    ds = DeepLesionDataset(split, data_dir, ann_file, transform)
    
    from multiprocessing import Pool
    from tqdm import tqdm
    import pickle
    train = []
    with Pool(processes=16) as pool:
        res = pool.imap_unordered(ds, range(len(ds)))#len(ds))
        for i in tqdm(res, total = len(ds)):
            train.append(i)
    with open(dataroot+output_filename,'wb') as f:
        pickle.dump(train,f)
