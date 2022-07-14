import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.registry import LOSSES

def dice_loss(input, target):
    smooth = 0.01
    input = input.sigmoid()
    # intersection = ((input * target).sum(dim=(1,2)))#减少一个中间变量 是否可以减少内存#hyadd
    dice = ((2. * ((input * target).sum(dim=(1,2))) + smooth) /
                (input.sum(dim=(1,2)) + target.sum(dim=(1,2)) + smooth))
    return 1 - dice.mean()

def mask_cross_entropy(pred, target, label, reduction='mean', avg_factor=None):
    # TODO: handle these two reserved arguments
    # assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return dice_loss(pred_slice, target)[None]
    # return F.binary_cross_entropy_with_logits(
    #     pred_slice, target, reduction='mean')[None]

@LOSSES.register_module
class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 loss_weight=1.0):
        super(DiceLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight

        if self.use_mask:
            self.cls_criterion = mask_cross_entropy


    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        # assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls