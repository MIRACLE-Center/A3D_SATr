import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
# import xlwt
import random
import  json
from deeplesion.mconfigs.densenet_align import new_anchor
mask_root=r'E:\ai\newCov_dataset\PCL\mask\renamed_accurate_mask'
BM_save_root=r'E:\ai\newCov_dataset\PCL\mask\BM_accurate_mask'
BBox_save_root=r'E:\ai\newCov_dataset\PCL\mask\BBox_accurate_mask'
#same center box select
def match_proposal_to_gt(gt_boxes,proposals):
    np_gt_boxes=gt_boxes.detach().cpu().numpy()
    np_proposals=proposals.detach().cpu().numpy()
    num_gt=gt_boxes.shape[0]
    num_proposal=np_proposals.shape[0]
    if num_gt==1: return [0]* num_proposal
    gt_inds=[]
    for i in range(num_proposal):
        proposal_box=np_proposals[i]
        distance=[]
        for j in range (num_gt):
            gt_box=np_gt_boxes[j]
            delt=np.abs(proposal_box-gt_box)
            distance.append(np.power(delt[0],2)+np.power(delt[1],2))
        tem_ids=np.argsort(distance)[0]#np argsort是从小到大排列的
        gt_inds.append(tem_ids)
    return gt_inds


def match_an_to_box(boxlist1, anchors,topk,scale=1,anchor_xyxy=1):
    # device=boxlist1.device
    # boxlist1 = boxlist1.convert('xywh')
    # boxlist2 = (boxlist2.convert('xywh'))
    #一般是xyxy 格式。
    box1=boxlist1*scale

    box2=anchors
    if anchor_xyxy:
        c_x2 = (box2[:, 0] + box2[:, 2]) / 2
        c_y2 = (box2[:, 1] + box2[:, 3]) / 2
    else:
        c_x2 = box2[:, 0] + (box2[:, 2] / 2)
        c_y2 = box2[:, 1] + (box2[:, 3] / 2)
    ids=[]
    for box in box1:
        draw_box_xyxy=0
        if draw_box_xyxy:
            c_x1=(box[0]+box[2])/2
            c_y1=(box[1]+box[3])/2
            diff_w=max(box[2]-box[0],40)
            diff_h=max(box[3]-box[1],40)
        else:
            c_x1=box[0]+(box[2]/2)
            c_y1=box[1]+(box[3]/2)
            #一定范围内的中心点相同的
            diff_w=max(box[2],40)
            diff_h=max(box[3],40)



        c_dx=np.abs( c_x1-c_x2)
        c_dy=np.abs(c_y1-c_y2)
        w=c_dx<diff_w
        h=c_dy<diff_h
        #先在一定范围内
        # result=w*h
        # a=np.where(result==True)
        count=len(np.nonzero(w*h)[0])# np中的nonzero 返回的是一个人array([]) 里面放着一个list_
        #取最少的
        tem_topk=min(topk,count)
        incuda=1
        if incuda:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            cuda_c_dx=torch.from_numpy(c_dx).to(device)
            cuda_c_dy=torch.from_numpy(c_dy).to("cuda:0")
            distance=torch.pow(cuda_c_dx,2) + torch.pow(cuda_c_dy,2)
            _,cuda_tem_ids=distance.topk(tem_topk,largest=False)
            tem_ids=cuda_tem_ids.cpu().numpy()
        else:
            distance=np.power(c_dx,2)+np.power(c_dy,2)

            #再根据距离，取前topK个，当topk<count，才有用
            tem_ids=np.argsort(distance)[:tem_topk]#np argsort是从小到大排列的
            # w = torch.pow(c_x1 - c_x2,2)
            # h = torch.pow(c_y1 - c_y2,2)
            # wh=w+h

        ids.append(tem_ids)
        # test=box2[tem_ids]
    final_ids=np.concatenate(ids,axis=0)
    return  final_ids

def match_an_to_gtbox(boxlist1, anchors,topk,scale=1,anchor_xyxy=1,draw_box_xyxy=1):#用GT BM 来匹配
    # device=boxlist1.device
    # boxlist1 = boxlist1.convert('xywh')
    # boxlist2 = (boxlist2.convert('xywh'))
    #一般是xyxy 格式。
    box1=boxlist1*scale

    box2=anchors
    # np_anchors=anchors.detach().cpu().numpy()
    if anchor_xyxy:
        c_x2 = (box2[:, 0] + box2[:, 2]) / 2
        c_y2 = (box2[:, 1] + box2[:, 3]) / 2
    else:
        c_x2 = box2[:, 0] + (box2[:, 2] / 2)
        c_y2 = box2[:, 1] + (box2[:, 3] / 2)
    ids=[]
    for box in box1:

        if draw_box_xyxy:
            c_x1=(box[0]+box[2])/2
            c_y1=(box[1]+box[3])/2
            diff_w=max((box[2]-box[0])/2,4)
            diff_w=min(12,diff_w)
            diff_h=max((box[3]-box[1])/2,4)
            diff_h=min(12,diff_h)
        else:
            c_x1=box[0]+(box[2]/2)
            c_y1=box[1]+(box[3]/2)
            #一定范围内的中心点相同的
            diff_w=max(box[2],8)
            diff_h=max(box[3],8)



        c_dx=torch.abs( c_x1-c_x2)
        c_dy=torch.abs(c_y1-c_y2)
        np_cdx=c_dx.cpu().numpy()
        np_cdy=c_dy.cpu().numpy()
        w=c_dx<diff_w
        h=c_dy<diff_h
        #先在一定范围内
        # result=w*h
        # a=np.where(result==True)
        # count=len(torch.nonzero(w*h)) # np中的nonzero 返回的是一个人array([]) 里面放着一个list_
        #取最少的
        # tem_topk=min(topk,count)
        # incuda=1
        distance=torch.pow(c_dx,2) + torch.pow(c_dy,2)
        #找最近的5倍，然后下面
        compare_area=1
        if compare_area:
            _,tem_ids=distance.topk(topk*20,largest=False)
            nearby_acnhors=anchors[tem_ids]#只根据距离找的anchors
            nearby_anchors_area= (nearby_acnhors[:,2]-nearby_acnhors[:,0])*(nearby_acnhors[:,3]-nearby_acnhors[:,1])
            area_ratio=nearby_anchors_area/((box[2]-box[0])*(box[3]-box[1]))
            # tem_ids=tem_ids[(area_ratio<2) & (area_ratio>0.5)]
            pre_len=len(tem_ids)
            tem_ids=tem_ids[(area_ratio<4) & (area_ratio>0.25)]
            # print(len(tem_ids))
            if len(tem_ids)>0:#只要不是0就行，大不了重复
                try:
                    rand_idx=torch.tensor(random.sample(range(0, len(tem_ids)),topk))
                except:
                    rand_idx=torch.randint(low=0,high=len(tem_ids),size=(topk,))#从这里随机生成1倍的
                    print(topk)
                    print(pre_len)
                    print(len(tem_ids))
            else:
                _,tem_ids=distance.topk(topk*5,largest=False)
                rand_idx=torch.randint(low=0,high=5*topk,size=(topk,))#从这里随机生成1倍的
            tem_ids=tem_ids[rand_idx]#这个当成最后一个
        else:
            _,tem_ids=distance.topk(topk*5,largest=False)
            rand_idx=torch.randint(low=0,high=5*topk,size=(topk,))#从这里随机生成1倍的
            tem_ids=tem_ids[rand_idx]#这个当成最后一个

        ids.append(tem_ids)
        # test=box2[tem_ids]
    final_ids=torch.cat(ids,dim=0)
    draw_viso=0 #论文作图用
    if draw_viso:
        pos_anchors=anchors[final_ids]
        Draw_BBox(pos_anchors,'../display_img/rpn_BM_anchor_samples.png',shape=(512,512),gt_box=box1)

    return  final_ids

def objectness_Normalize(img):
    #img:numpy
    #output: normalized numpy
    max_val=img.max()
    min_val=img.min()

    n_img=(img-min_val)/(max_val-min_val)
    return n_img

def objectness_to_mathced_ids(one_batch_one_level_numpy_objectness,all_level_anchors,pre_nms_top_n):
    A=one_batch_one_level_numpy_objectness.shape[0]#anchor类
    W=one_batch_one_level_numpy_objectness.shape[-1]
    per_img=[]
    pre_nms_top_n=int(pre_nms_top_n/A)#因为后面match anchor to box时，有多少draw box就要match几个topk,这里假设一个图上只有一个drawbox
    for a in range(A):
        numpy_single_objectness=one_batch_one_level_numpy_objectness[a,:,:]
        # threshold=numpy_single_objectness.max+numpy_single_objectness.min
        nor_numpy_single_objectness=objectness_Normalize(numpy_single_objectness)
        bin_np_sing_object= np.where(nor_numpy_single_objectness>0.5,1,0)
        # if W==200:
        #     plt.imsave(r'../drawbox.png', bin_np_sing_object, cmap='gray')
        _,_,tem_BBox,_=cv2.connectedComponentsWithStats((bin_np_sing_object*255).astype(np.uint8),)

        BBox=tem_BBox[1:]
        removed_BBox=[]
        min_x=int(W/20)+1#CT很靠边的肯定不是病灶了
        max_x=W-min_x+2

        center_x=int(W/2)+1
        for i in range(len(BBox)):
            tem_BBox=BBox[i]
            if tem_BBox[4]>5 and tem_BBox[0]>min_x and tem_BBox[1]>min_x and tem_BBox[0]<max_x and tem_BBox[1]<max_x:
                removed_BBox.append(tem_BBox[:-1])
        if removed_BBox==[]:
            removed_BBox=[[center_x,center_x,min_x,min_x]] #这里加上，只是为了可以让append 进行，只能加一个xx ，不影响结果
        removed_BBox=np.array(removed_BBox)
        # if W==200:
        #     box_img=bin_np_sing_object*255+Draw_BBox(removed_BBox,w=W,is_xywh=True)
        #     plt.imsave(r'../one_RPN_selector_box.png', box_img.astype(np.uint8), cmap='gray')

        per_img.append(removed_BBox)

    per_img=np.concatenate(per_img,axis=0)
    num_draw_box=per_img.shape[0]
    if num_draw_box > 32:
        random_list=np.random.randint(0,num_draw_box,32,dtype=np.int).tolist()
        per_img=per_img[random_list]
    scale=800/W#draw的box是object尺寸的
    topk_ids = match_an_to_box(per_img,all_level_anchors,pre_nms_top_n,scale=scale)
    # if per_img.size>8:
    #     per_img=per_img[:8]
    box1_proposal=per_img*scale
    box1_proposal[:,2]+=box1_proposal[:,0]
    box1_proposal[:,3]+=box1_proposal[:,1]
    return topk_ids, box1_proposal

def BMgt_to_mathced_ids(gt_box,all_level_anchors,pre_nms_top_n):
    #如果有BM 来找anchor 就不用考虑channel了，直接用一个BM 全算了，
    pre_nms_top_n=int(pre_nms_top_n)#因为后面match anchor to box时，有多少draw box就要match几个topk,这里假设一个图上只有一个drawbox

    topk_ids = match_an_to_gtbox(gt_box,all_level_anchors,pre_nms_top_n)#用的是GT box 和anchor都是是原图的尺寸

    box1_proposal=None
    return topk_ids, box1_proposal

def one_img_gt_bbox_as_maps(target,w=800,c_18=1,size_aware=0):
    # the input should be GT box
    # w is the img size w.r.t GT box
    gtbox = target
    num_box = gtbox.shape[0]
    # 每个gtbox都画在map上
    box_map_x = torch.zeros((w, w))
    box_map_y = torch.zeros((w, w))
    box_map_xy = torch.zeros((w, w))
    max_area=0
    for i in range(num_box):
        tem_box_map_x = torch.zeros((w, w))
        tem_box_map_y = torch.zeros((w, w))
        box = gtbox[i]
        #MVP是5个数，应该是1234 其他是 0123
        x1 = box[0].item()-1
        y1 = box[1].item()-1
        x2 = box[2].item()-1
        y2 = box[3].item()-1
        area=(x2-x1)*(y2-y1)
        max_area=max(area,max_area)
        dx = 0.5 * (x2 - x1)
        dy = 0.5 * (y2 - y1)
        ctr_x = np.round(x1 + dx).astype(np.int)
        ctr_y = np.round(y1 + dy).astype(np.int)
        int_x1 = np.round(x1).astype(np.int)
        int_x2 = np.round(x2).astype(np.int)
        int_y1 = np.round(y1).astype(np.int)
        int_y2 = np.round(y2).astype(np.int)
        center_1=0
        size_adapt=0
        if center_1:
            assert  center_1==size_adapt #如果用center1 那就一定是sizeada
            if dx*dy <50:
                tem_box_map_y[int_y1:int_y2, int_x1:int_x2] = 1
                tem_box_map_x[int_y1:int_y2, int_x1:int_x2] = 1
            else:
                k_x = 0.5 / dx
                k_y = 0.5 / dy
                tem_box_map_x[int_y1:int_y2, ctr_x] = 1
                # tem_box_map_y[int_x1:int_x2, ctr_y] = 1
                tem_box_map_y[ctr_y, int_x1:int_x2] = 1

                dx_len = np.round(dx).astype(np.int)
                dy_len = np.round(dy).astype(np.int)

                for i_x in range(dx_len):
                    # tem_box_map_x[ctr_x - 1 - i_x, int_y1:int_y2] = 1 - k_x * (i_x + 1)
                    tem_box_map_x[int_y1:int_y2, ctr_x - 1 - i_x] = 1 - k_x * (i_x + 1)
                    # tem_box_map_x[ctr_x + 1 + i_x, int_y1:int_y2] = 1 - k_x * (i_x + 1)
                    tem_box_map_x[int_y1:int_y2, ctr_x + 1 + i_x] = 1 - k_x * (i_x + 1)

                for i_y in range(dy_len):
                    # tem_box_map_y[int_x1:int_x2, ctr_y - 1 - i_y] = 1 - k_y * (i_y + 1)
                    tem_box_map_y[ctr_y - 1 - i_y, int_x1:int_x2] = 1 - k_y * (i_y + 1)
                    # tem_box_map_y[int_x1:int_x2, ctr_y + 1 + i_y] = 1 - k_y * (i_y + 1)
                    tem_box_map_y[ctr_y + 1 + i_y, int_x1:int_x2] = 1 - k_y * (i_y + 1)
                tem_box_map_y[ctr_y - 1 - 7:ctr_y - 1 +7, ctr_x - 1 - 7:ctr_x - 1 +7] = 1
                tem_box_map_x[ctr_y - 1 - 7:ctr_y - 1 +7, ctr_x - 1 - 7:ctr_x - 1 +7] = 1


        else:
            if size_adapt:
                if dx*dy <50: #因为dy dx是半个长度，见上面，所以100 是400,50是200
                    k_x=0
                    k_y=0
                elif dx*dy <250:#250是1000
                    k_x = 0.5 / dx
                    k_y = 0.5 / dy
                else:
                    # k_x = 0.7 / dx
                    # k_y = 0.7 / dy
                    k_x = 0.5 / dx
                    k_y = 0.5 / dy
            else:
                k_x = 0.5 / dx if dx > 1 else 0.5 #if是备用的
                k_y = 0.5 / dy if dy > 1 else 0.5

            # tem_box_map_x[ctr_x, int_y1:int_y2] = 1
            tem_box_map_x[int_y1:int_y2, ctr_x] = 1
            # tem_box_map_y[int_x1:int_x2, ctr_y] = 1
            tem_box_map_y[ctr_y, int_x1:int_x2] = 1

            dx_len = np.round(dx).astype(np.int)
            dy_len = np.round(dy).astype(np.int)

            for i_x in range(dx_len):#tensor是HW，应该是yx
                # tem_box_map_x[ctr_x - 1 - i_x, int_y1:int_y2] = 1 - k_x * (i_x + 1)
                tem_box_map_x[int_y1:int_y2, ctr_x - 1 - i_x] = 1 - k_x * (i_x + 1)
                # tem_box_map_x[ctr_x + 1 + i_x, int_y1:int_y2] = 1 - k_x * (i_x + 1)
                tem_box_map_x[int_y1:int_y2, ctr_x + 1 + i_x] = 1 - k_x * (i_x + 1)

            for i_y in range(dy_len):
                # tem_box_map_y[int_x1:int_x2, ctr_y - 1 - i_y] = 1 - k_y * (i_y + 1)
                tem_box_map_y[ctr_y - 1 - i_y, int_x1:int_x2] = 1 - k_y * (i_y + 1)
                # tem_box_map_y[int_x1:int_x2, ctr_y + 1 + i_y] = 1 - k_y * (i_y + 1)
                tem_box_map_y[ctr_y + 1 + i_y, int_x1:int_x2] = 1 - k_y * (i_y + 1)

        tem_box_map_xy = np.sqrt(tem_box_map_x * tem_box_map_y)
        box_map_x += tem_box_map_x
        box_map_y += tem_box_map_y
        box_map_xy += tem_box_map_xy
    one = torch.ones_like(box_map_x)
    box_map_x=torch.where(box_map_x > 1,one,box_map_x)
    tem_num_BMx = (tem_box_map_x).cpu().numpy()
    box_map_y=torch.where(box_map_y > 1,one,box_map_y)
    tem_num_BMy = (tem_box_map_y).cpu().numpy()
    box_map_xy=torch.where(box_map_xy > 1,one,box_map_xy)
    # draw_box_map_x = box_map_x.numpy()
    draw=0
    if draw:
        # draw_box_map_y = box_map_y.numpy()
        draw_box_map_xy = box_map_xy.numpy()
        # plt.imsave('../BMXY_center1.png', np.floor(draw_box_map_xy * 255), cmap='gray')
        plt.imsave('../display_img/BMXY_RPN.png', np.floor(draw_box_map_xy * 255), cmap='gray')
    # writer = SummaryWriter('/home1/hli/nCov/semisupervised/maskrcnn-benchmark-master/tools/runs')
    # writer.add_image('mask_BM_GT', draw_box_map_y, dataformats='HW')
    # writer.close()
    #RPN阶段，MVP中，用的是三层，Align 是用的18层
    # box_map = torch.stack((box_map_xy.float(), box_map_xy.float(),box_map_xy.float()), dim=0)
    if c_18:
        if new_anchor:
            ss_area=256
            s_area=576
            m_area=1024
            l_area=2304
            xl_area=4096
        else:
            ss_area=256
            s_area=576
            m_area=1024
            l_area=2304
            xl_area=9216


        if size_aware:
            if max_area<ss_area:
                ids=[0,6,12]
            elif max_area<s_area:
                ids=[1,7,13]
            elif max_area<m_area:
                ids=[2,8,14]
            elif max_area<l_area:
                ids=[3,9,15]
            elif max_area<xl_area:
                ids=[4,10,16]
            else:
                ids=[5,11,17]
            Align_BM=torch.zeros((18,w, w))
            if size_aware==1:
                Align_BM[ids]=box_map_xy
            elif size_aware==2:
                all_ids=[i for i in range(18)]
                Align_BM[all_ids] = 0.5*box_map_xy
                Align_BM[ids] = box_map_xy
            elif size_aware==3:
                pass
            box_map=Align_BM.float()

            # np_box_map=box_map.numpy()
            # for i in range(18):
            #     # plt.imsave('../display_img/BMgt_%s_size_aware_RPN.png'%(str(i)), np.floor(np_box_map[i,:,:]* 255), cmap='gray')
            #     plt.imsave('../display_img/BMgt_%s_size_aware_RPN.png'%(str(i)), np_box_map[i,:,:], cmap='gray')

        else:
            Align_BM=[box_map_xy.float()]*18
            box_map = torch.stack(Align_BM, dim=0)
    else:
        box_map=[box_map_xy.float()]
    # box_map = torch.stack((box_map_x.float(), box_map_y.float(),box_map_xy.float()), dim=0)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    box_map= box_map.to(target.device)
    return box_map




def Box_as_maps(bbox_label,file_name):
    # the input should be GT box
    gtbox=bbox_label
    num_box=gtbox.shape[0]#cv2的背景也会算box
    # 每个gtbox都画在map上
    box_map_x = np.zeros((512, 512))
    box_map_y = np.zeros((512, 512))
    box_map_xy = np.zeros((512, 512))

    for i in range(1,num_box):
        tem_box_map_x = np.zeros((512, 512))
        tem_box_map_y = np.zeros((512, 512))
        # tem_box_map_xy = np.zeros((512, 512))
        box=gtbox[i]
        x1=box[0]
        y1=box[1]
        x2=x1+box[2]
        y2=y1+box[3]
        dx=0.5*(x2-x1)
        dy=0.5*(y2-y1)
        ctr_x=np.round(x1+dx).astype(np.int)
        ctr_y=np.round(y1+dy).astype(np.int)
        int_x1=np.round(x1).astype(np.int)
        int_x2=np.round(x2).astype(np.int)
        int_y1=np.round(y1).astype(np.int)
        int_y2=np.round(y2).astype(np.int)
        k_x=0.5/dx
        k_y=0.5/dy

        tem_box_map_x[ctr_x,int_y1:int_y2]=1
        tem_box_map_y[int_x1:int_x2,ctr_y]=1

        dx_len=np.round(dx).astype(np.int)
        dy_len=np.round(dy).astype(np.int)



        for i_x in range(dx_len):
            tem_box_map_x[ctr_x-1-i_x,int_y1:int_y2]=1-k_x*(i_x+1)
            tem_box_map_x[ctr_x+1+i_x,int_y1:int_y2]=1-k_x*(i_x+1)

        for i_y in range(dy_len):
            tem_box_map_y[int_x1:int_x2,ctr_y-1-i_y]=1-k_y*(i_y+1)
            tem_box_map_y[int_x1:int_x2,ctr_y+1+i_y]=1-k_y*(i_y+1)

        tem_box_map_xy=np.sqrt(tem_box_map_x*tem_box_map_y)
        box_map_x+=tem_box_map_x
        box_map_y+=tem_box_map_y
        box_map_xy+=tem_box_map_xy
    # draw_box_map_x=box_map_x.numpy()
    # draw_box_map_y=box_map_y.numpy()
    box_map_x[np.where(box_map_x>1)]=1
    box_map_y[np.where(box_map_y>1)]=1
    box_map_xy[np.where(box_map_xy>1)]=1
    abs_xBM_save_name=os.path.join(BM_save_root,'x',file_name)
    abs_yBM_save_name=os.path.join(BM_save_root,'y',file_name)
    abs_xyBM_save_name=os.path.join(BM_save_root,'xy',file_name)
    save_box_map_x=(np.floor(box_map_x*255)).transpose().astype(np.uint8)
    save_box_map_y=(np.floor(box_map_y*255)).transpose().astype(np.uint8)
    save_box_map_xy=(np.floor(box_map_xy*255)).transpose().astype(np.uint8)
    # plt.imshow((save_box_map_x * 255), cmap='gray')
    # plt.show()
    # plt.imsave(abs_xBM_save_name,save_box_map_x,cmap='gray')
    # plt.imsave(abs_yBM_save_name,save_box_map_y,cmap='gray')
    # plt.imsave(abs_xyBM_save_name,save_box_map_xy,cmap='gray')
    cv2.imwrite(abs_xBM_save_name, save_box_map_x)
    cv2.imwrite(abs_yBM_save_name, save_box_map_y)
    cv2.imwrite(abs_xyBM_save_name, save_box_map_xy)

    # box_map=torch.stack((box_map_x.float(),box_map_y.float()),dim=0)
    # return box_map
def Draw_BBox(bbox_label,file_name, image=None,shape=(800,800),gt_box=None):
    proposals = bbox_label#the gtbox is the sampled bbox in RPN,and the gt_box is the true gt box
    num_box = proposals.shape[0]   # cv2的背景也会算box
    # 每个gtbox都画在map上

    bbox_img = np.zeros(shape)
    if image is not None: bbox_img=image.astype(np.uint8)
    for i in range(0,num_box):# 0 is dropped for in some other cases, 0 is the backgroud
        box=proposals[i]
        x1=int(box[0])
        y1=int(box[1])
        x2=int(box[2])
        y2=int(box[3])
        zero_map = np.zeros(shape)
        tem_draw_tan=cv2.rectangle(zero_map,(x1,y1),(x2,y2),255, 1).astype(np.uint8)
#        （原图，左上坐标，右下，颜色，粗细）
        bbox_img+=tem_draw_tan
    # abs_bbox_save_name=os.path.join(BBox_save_root,file_name)
    if gt_box is not None: #in RPN, the gt_box is specific, we will draw it in another channel
        num_gt_box = gt_box.shape[0]

        gt_bbox_img = np.zeros(shape)
        for i in range(0, num_gt_box):
            box = gt_box[i]
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            zero_map = np.zeros(shape)
            gtbox_map = cv2.rectangle(zero_map, (x1, y1), (x2, y2), 255, 2)
            gt_bbox_img+=gtbox_map
        zero_map = np.zeros(shape)
        bbox_img=bbox_img.clip(min=0,max=255)
        plt.imsave(file_name, np.stack((bbox_img/bbox_img.max(), gt_bbox_img/gt_bbox_img.max(), zero_map), axis=2))
        #maybe there is more than one GT boxes, divide the max to avoid overflow
        # plt.imsave(file_name, np.stack((bbox_img/bbox_img.max(), gt_bbox_img/gt_bbox_img.max(), zero_map), axis=2))
    else:
        plt.imsave(file_name, bbox_img, cmap='gray')
    # plt.imshow(bbox_img,cmap='gray')
    # plt.show()
def find_bbox(mask):
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    # stats 是bounding box的信息，N*(连通域label数)的矩阵，行对应每个label，这里也包括背景lable 0，列为[x0, y0, width, height, area]
    # centroids 是每个域的质心坐标(非整数)

    """
    输入：
    [0, 255, 0, 0],
    [0, 0, 0, 255],
    [0, 0, 0, 255],
    [255, 0, 0, 0]
    labels:
    [[0 1 0 0]
     [0 0 0 2]
     [0 0 0 2]
     [3 0 0 0]]
     stats
    [[  0  64   0   0]
     [  0   0   0 191]
     [  0   0   0 191]
     [255   0   0   0]]
     centroids:
     [[1.41666667 1.5       ]
 [1.         0.        ]
 [3.         1.5       ]
 [0.         3.        ]]
    """

    stats = stats[stats[:,4].argsort()]
    return stats[:-1]

# mask = label[4]
# ax = plt.axes()
# plt.imshow(mask,cmap='bone')
# bboxs = find_bbox(mask)
# for j in bboxs:
#     rect = patches.Rectangle((j[0],j[1]),j[2],j[3],linewidth=1,edgecolor='r',facecolor='none')
#     ax.add_patch(rect)
# plt.show()
def To_string(array):
    list=array.tolist()
    new_list=[]
    for i in range(len(list)):
        new_row_list=[]
        for j in range(len(list[i])):
            new_row_list.append(str(list[i][j]))
        new_list.append(new_row_list)
    return  new_list

def BM_assign_and_sample():
    return 0

if __name__=='__main__':
    Workbook=xlwt.Workbook()
    sheet=Workbook.add_sheet('bbox')
    ALL_label_dict={}

    for f in os.listdir(mask_root):
        # f='CAO_KAI_WEI_P0482688_115.png'
        tem_label_dict={}
        abs_mask_name=os.path.join(mask_root,f)
        mask=cv2.imread(abs_mask_name,-1)
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))



        # labels=To_string(labels)
        # stats=To_string(stats)
        # centroids=To_string(centroids)

        # label_dict={'label':labels[1:].tolist()}
        BBox_dict={'BBox':stats[1:].tolist()}
        centroids_dict={'center':centroids[1:].tolist()}

        # tem_label_dict.update(label_dict)
        tem_label_dict.update(BBox_dict)
        tem_label_dict.update(centroids_dict)
        ALL_label_dict.update({f:tem_label_dict})
        # BBox.append(tem_BBox)
        # Box_as_maps(stats,f)
        # Draw_BBox(stats,f,mask)
        print(f)

    with open(r'.\BBox_label.json', 'w') as f:
        json.dump(ALL_label_dict, f)
    count=0
    # for i in range(len(BBox)):
    #     for j in range(len(BBox[i])):
    #         sheet.write(i,j,BBox[i][j])
    #     count+=1
    #     print(count)
    # Workbook.save('.\mask2BBox.xls')