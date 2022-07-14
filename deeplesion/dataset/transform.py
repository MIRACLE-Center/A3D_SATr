from mmdet.datasets.registry import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
import numpy as np

@PIPELINES.register_module
class ImageToTensor_3d(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(np.expand_dims(results[key].transpose(2, 0, 1), 0))
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)



@PIPELINES.register_module
class DefaultFormatBundle_3d(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        if 'img' in results:
            img = np.ascontiguousarray(np.expand_dims(results['img'].transpose(2, 0, 1), 0))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)
        if 'thickness' in results:
            results['thickness'] = DC(to_tensor(results['thickness']), stack=True, pad_dims=None)
        return results

    def __repr__(self):
        return self.__class__.__name__