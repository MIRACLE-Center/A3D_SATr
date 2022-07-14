import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from pycocotools import mask as mutils
import sys
# sys.path.append('/your alignshit path')
import os
# from dataset import DeepLesionDataset
import cv2
import random
import matplotlib.pyplot as plt
from mmcv import Config
import mmcv
from mmcv.runner import get_dist_info, load_checkpoint
from mmcv.parallel import MMDataParallel
import random
import pickle
import gc
import argparse
from mmdet.datasets.registry import DATASETS, PIPELINES
from mmdet.models.registry import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                       ROI_EXTRACTORS, SHARED_HEADS)
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
# from deeplesion.evaluation.visualize import draw_bounding_boxes_on_image_array
from deeplesion.evaluation.evaluation_metrics import sens_at_FP

from deeplesion.evaluation.evaluation_metrics import sens_at_FP, IOU
from skimage.measure import regionprops
from pycocotools.mask import decode

def parse_args():
    parser = argparse.ArgumentParser(description='eval deeplesion')
    # parser.add_argument('config', help='train config file path')
    parser.add_argument('--config', default=None, help='config path')
    parser.add_argument('--checkpoint', help='checkpoint path')
    args = parser.parse_args()

    return args

def generate_cfg(checkpoint):    
    d = torch.load(checkpoint, map_location=torch.device('cpu'))
    cfg_path = checkpoint.replace('.pth','.py')
    if not os.path.exists(cfg_path):
        with open(cfg_path,'w') as f:
            f.write(d['meta']['config'])
    return cfg_path

def get_model(cfg_path):
    cfg = Config.fromfile(cfg_path)
    cfg.data.imgs_per_gpu=1
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.data.val.test_mode = True
    # cfg.test_cfg.rcnn.score_thr=0.001
    #build dataset
    print(cfg.description)
    dataset = build_dataset(cfg.data.test)
    data_loadertest = build_dataloader(
                                        dataset,
                                        imgs_per_gpu=1,
                                        workers_per_gpu=0,
                                        dist=False,
                                        shuffle=False)

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    model = MMDataParallel(model, device_ids=[0])
    # from mmcv.parallel import  MMDistributedDataParallel#hanadd
    # model = MMDistributedDataParallel(model.cuda())#hanadd
    model.CLASSES = dataset.CLASSES
    return model, data_loadertest

# def single_gpu_test(model, data_loader):
#     model.eval()
#     results = []
# #     dataset = data_loader.dataset
#     prog_bar = mmcv.ProgressBar(len(data_loader))
#     with torch.no_grad():
#         for i, data in enumerate(data_loader):

#             gt_boxes = data.pop('gt_bboxes')
#             r = model(return_loss=False, rescale=False, **data)   
#             # inference_time.append(time.time() - start_time)
#             data['gt_boxes'] = gt_boxes
#             data['bboxes'] = r[0]
#             data['segs'] = r[1]
#             # data['img'] = data['img'].data[0][data['img'].data[0].shape[0]//2]
#             data.pop('img')
#             results.append(data)
#             prog_bar.update()
#     return results

def single_gpu_test(model, data_loader):
    model.eval()
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    with torch.no_grad():
        ann = data_loader.dataset.ann
        for i, data in enumerate(data_loader):
            gt_boxes = data.pop('gt_bboxes')
            r = model(return_loss=False, rescale=False, **data)
            j=0
            res_dict={}
            res_dict['gt_boxes'] = gt_boxes.data[0][j].numpy()
            res_dict['bboxes'] = r[0][j]
            res_dict['segs'] = r[1][j]
            # res_dict['img'] = data['img'].data[j][0][0]
            dd = ann[i+j]['ann']['diameters']
            res_dict['diameters'] = dd
            res_dict['img_meta'] = data['img_meta'].data[0][j]
            # data.pop('img')
            res_dict['gt_masks'] = data['gt_masks'].data[0][j]
            res_dict['spacing'] = ann[i]['ann']['spacing']
            res_dict['recists'] = ann[i]['ann']['recists']
            res_dict['thickness'] = ann[i]['ann']['slice_intv']
            # res_dict['diameter_erro'], res_dict['pred_mask'], res_dict['gt_mask_my'], res_dict['pred_mask_index'] = mask_matrics(res_dict)                
            results.append(res_dict)
                
            prog_bar.update()
    return results

def write_metrics(outputs, log_path, epoch):
    avgFP = [0.5, 1, 2, 4, 8, 16]
    iou_th = 0.5
    s1_box=[]
    s1_gt=[]
    s5_box=[]
    s5_gt=[]
    so_box=[]
    so_gt=[]
    so_seg_erro = []
    for d in outputs:
        if d['thickness']<=2.:
            s1_box.append(np.vstack(d['bboxes']))
            s1_gt.append(d['gt_boxes'])
        elif d['thickness']==5.:
            s5_box.append(np.vstack(d['bboxes']))
            s5_gt.append(d['gt_boxes']) 
        else:
            so_box.append(np.vstack(d['bboxes']))
            so_gt.append(d['gt_boxes']) 
        # so_seg_erro.extend(d['diameter_erro'])
    sens1 = sens_at_FP(s1_box, s1_gt, avgFP, iou_th)
    sens2 = sens_at_FP(s5_box, s5_gt, avgFP, iou_th)
    sens = sens_at_FP(s1_box+s5_box+so_box, s1_gt+s5_gt+so_gt, avgFP, iou_th)

    # so_seg_erro = np.array(so_seg_erro)
    # diameter_erro = so_seg_erro[so_seg_erro>-1].mean()
    s = str(epoch)+':\t'+str(sens)+'\t'+str(sens1)+'\t'+str(sens2)+f'\t \n'#diameter_erro:{diameter_erro}\n'
    print(s)
    # with open(log_path,'a+') as f:
    #     f.write(s)
    return s 

def mask_matrics(output, iou_thresh=0.5):
    erro = [-1] * len(output['gt_boxes'])
    pred_mask = []
    gt_mask = []
    pred_mask_index = []
    pred_mask_contours = []
    for i, box in enumerate(output['gt_boxes']):
        iou1 = IOU(box, output['bboxes'])
        if len(iou1)==0:
            continue
        indx = iou1.argmax()
        try:
            if iou1[indx] > iou_thresh:
                d_seg = decode(output['segs'][indx])
                pred_mask.append(output['segs'][indx])
                l_seg = output['gt_masks'][i].astype(np.uint8)
                gt_mask.append(l_seg)
                # _, cnts = cv2.findContours(l_seg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                pred_mask_index.append(indx)
                prop = regionprops(d_seg)[0]
                diameter = np.sqrt(prop.major_axis_lengt**2 + prop.minor_axis_length**2)/2
                recists = output['recists'][i].reshape((4, 2))
                l = np.linalg.norm(recists[0] - recists[1])
                s = np.linalg.norm(recists[2] - recists[3])
                diameter1 = np.sqrt(l**2 + s**2)/2

                erro[i] = (abs(prop.major_axis_lengt-max(l,s)) + abs(prop.minor_axis_length-min(l,s))) * output['spacing']
                # erro[i] = np.abs(diameter1 - diameter) * output['spacing']
        except Exception as e:
            print(e)
            print(len(output['segs']), indx, i, output['bboxes'].shape, decode(output['segs'][indx]).sum())

    return erro, pred_mask, gt_mask, pred_mask_index

def main(checkpoint, cfg_path=None):
    if cfg_path is None:
        cfg_path = generate_cfg(checkpoint)
    print(cfg_path)
    model, dl = get_model(cfg_path)
    log_path = './log/metrix_log.txt'
    load_checkpoint(model, checkpoint, map_location='cpu', strict=True)
    outputs = single_gpu_test(model, dl)
    r = write_metrics(outputs, log_path, 'N/A')
    # save_output(outputs, os.path.basename(os.path.dirname(checkpoint))+'.pkl')
    with open(log_path,'a+') as f:
        f.write(checkpoint+':\n'+r)
    with open(os.path.dirname(checkpoint)+'metrics_log.txt','a+') as f:
        f.write(r)        
        print(r)

if __name__ =='__main__':
    # checkpoint_path = f'/mnt/data3/deeplesion/dl/work_dirs/densenet_3d_acs_r2/latest.pth'
    args = parse_args()
    checkpoint = args.checkpoint
    cfg_path = args.config#generate_cfg(checkpoint)
    main(checkpoint, cfg_path)