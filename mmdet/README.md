# mmdetection modifications
## 1. add code for evaluation during training

in file [./mmdet/apis/train.py](./apis/train.py) import MyDeepLesionEval and register after epoch hook
```python
from deeplesion.evaluation.evaluation import MyDeepLesionEval
......
runner.register_hook(MyDeepLesionEval(val_dataset_cfg, **eval_cfg))

```

## 2. regist backbone in [./mmdet/models/backbones/__init__.py](./models/backbones/__init__.py)
```python
from deeplesion.models.truncated_densenet import DenseNetCustomTrunc
from deeplesion.models.truncated_densenet3d_tsm import DenseNetCustomTrunc3dTSM
from deeplesion.models.truncated_densenet3d_alignshift import DenseNetCustomTrunc3dAlign
```
## 3. regist align detector in [./mmdet/models/detectors/__init__.py](./models/detectors/__init__.py)

```python 
from deeplesion.models.detector_alignshift import AlignShiftMaskRCNN
```

## 4. regist dice loss in [./mmdet/models/losses/__init__.py](./models/losses/__init__.py)
```python
from deeplesion.losses.diceloss import DiceLoss
```