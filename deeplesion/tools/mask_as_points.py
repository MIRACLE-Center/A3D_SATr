import numpy as np
import torch
import torchvision.transforms as t
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from maskrcnn_benchmark.layers import ROIAlign



def gt_bbox_as_maps(target):
    # the input should be GT box
    gtbox=target.bbox
    num_box=gtbox.shape[0]
    # 每个gtbox都画在map上
    box_map_x = torch.zeros((800, 800))
    box_map_y = torch.zeros((800, 800))

    for i in range(num_box):
        box=gtbox[i]
        x1=box[0].item()
        y1=box[1].item()
        x2=box[2].item()
        y2=box[3].item()
        dx=0.5*(x2-x1)
        dy=0.5*(y2-y1)
        ctr_x=np.round(x1+dx).astype(np.int)
        ctr_y=np.round(y1+dy).astype(np.int)
        int_x1=np.round(x1).astype(np.int)
        int_x2=np.round(x2).astype(np.int)
        int_y1=np.round(y1).astype(np.int)
        int_y2=np.round(y2).astype(np.int)

        k_x=0.5/dx if dx>1 else 0.5
        k_y=0.5/dy if dy>1 else 0.5

        box_map_x[ctr_x,int_y1:int_y2]=1
        box_map_y[int_x1:int_x2,ctr_y]=1

        dx_len=np.round(dx).astype(np.int)
        dy_len=np.round(dy).astype(np.int)



        for i_x in range(dx_len):
            box_map_x[ctr_x-1-i_x,int_y1:int_y2]=1-k_x*(i_x+1)
            box_map_x[ctr_x+1+i_x,int_y1:int_y2]=1-k_x*(i_x+1)

        for i_y in range(dy_len):
            box_map_y[int_x1:int_x2,ctr_y-1-i_y]=1-k_y*(i_y+1)
            box_map_y[int_x1:int_x2,ctr_y+1+i_y]=1-k_y*(i_y+1)
    draw_box_map_x=box_map_x.numpy()
    draw_box_map_y=box_map_y.numpy()
    # plt.imsave('/home1/hli/maskrcnn-benchmark-master/input0.png',np.floor(draw_box_map_y*255),cmap='gray')
    # writer=SummaryWriter('/home1/hli/nCov/semisupervised/maskrcnn-benchmark-master/tools/runs')
    # writer.add_image('mask_BM_GT',draw_box_map_y,dataformats='HW')
    # writer.close()
    box_map=torch.stack((box_map_x.float(),box_map_y.float()),dim=0)
    return box_map

def proposal_map_from_box(box_map,proposal_box):
    pro_boxs=proposal_box
    len_pro_boxs=pro_boxs.shape[0]
    proposal_box_maps=[]
    for j in range(len_pro_boxs):
        proposal_box=pro_boxs[j]
        x1 = proposal_box[0].item()
        y1 = proposal_box[1].item()
        x2 = proposal_box[2].item()
        y2 = proposal_box[3].item()
        int_x1 = np.round(x1).astype(np.int)
        int_x2 = np.round(x2).astype(np.int)
        int_y1 = np.round(y1).astype(np.int)
        int_y2 = np.round(y2).astype(np.int)
        proposal_box_map=box_map[:,int_x1:int_x2,int_y1:int_y2]
        # 这里的box_map基本不是正方形的，但是我们之前对特征进行了roi align,这里直接进行resize就行
        # ROI=ROIAlign([256,256])
        # resized_box_map=ROI(proposal_box_map)

        proposal_box_mapPIL=[t.ToPILImage()(proposal_box_map[0]),t.ToPILImage()(proposal_box_map[1])]

        resize=t.Compose([t.Resize((28,28))])
        resized_mapPIL=[resize(proposal_box_mapPIL[0]),resize(proposal_box_mapPIL[1])]
        a,b=t.ToTensor()(resized_mapPIL[0]),t.ToTensor()(resized_mapPIL[1])
        new_proposal_box_map=torch.cat((a,b),dim=0)
        proposal_box_maps.append(new_proposal_box_map)
        # if j ==0:
        #     proposal_box_maps=new_proposal_box_map
        # else:
        #
        #     proposal_box_maps=torch.stack([proposal_box_maps,new_proposal_box_map],dim=0)
    proposal_box_maps = torch.stack(proposal_box_maps, dim=0)
    return proposal_box_maps.float()


