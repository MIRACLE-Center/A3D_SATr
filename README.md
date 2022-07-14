# DeepLesion Codebase for Universal Lesion Detection

This repository contains the codebase for our 2 papers A3D (MICCAI'21) and AlignShift (MICCAI'20), which achieves *state-of-the-art* performance on DeepLesion for universal lesion detection. 

* Asymmetric 3D Context Fusion for Universal Lesion Detection ([MICCAI'21](https://arxiv.org/abs/2109.08684))

* *AlignShift*: Bridging the Gap of Imaging Thickness in 3D Anisotropic Volumes ([MICCAI'20](https://arxiv.org/abs/2005.01969), early accepted)


## Code structure

* ``nn``
  The core implementation of AlignShift convolution and TSM convolution, including the operators, models, and 2D-to-3D/AlignShift/TSM model converters. 
  * ``operators``: A3DConv, AlignShiftConv, TSMConv.
  * ``converters.py``: include converters which convert 2D models to 3DConv/AlignShiftConv/TSMConv/A3DConv counterparts.
  * ``models``: Native AlignShift/TSM/A3DConv models. 
* ``deeplesion`` 
  The experiment code is based on [mmdetection](https://github.com/open-mmlab/mmdetection), this directory consists of compounents used in mmdetection.
* ``mmdet``: a duplication of [mmdetection](https://github.com/open-mmlab/mmdetection) with our new models registered.

## Installation

 * git clone this repository
 * pip install -e . 
 
The code requires only common Python environments for machine learning. Basically, it was tested with
Python 3 (>=3.6)
PyTorch==1.3.1
numpy==1.18.5, pandas==0.25.3, scikit-learn==0.22.2, Pillow==8.0.1, fire, scikit-image
Higher (or lower) versions should also work (perhaps with minor modifications).

## Convert a 2D model into 3D with a single line of code

```python
from converter import Converter
import torchvision
from alignshift import AlignShiftConv
# m is a standard pytorch model
m = torchvision.models.resnet18(True)
alignshift_conv_cfg = dict(conv_type=AlignShiftConv, 
                          n_fold=8, 
                          alignshift=True, 
                          inplace=True,
                          ref_spacing=0.2, 
                          shift_padding_zero=True)
m = Converter(m, 
              alignshift_conv_cfg, 
              additional_forward_fts=['thickness'], 
              skip_first_conv=True, 
              first_conv_input_channles=1)
# after converted, m is using AlignShiftConv and capable of processing 3D volumes
x = torch.rand(batch_size, in_channels, D, H, W)
thickness = torch.rand(batch_size, 1)
out = m(x, thickness)
```

## Usage of AlignShiftConv/TSMConv/A3DConv operators

```python
from nn.operators import AlignShiftConv, TSMConv, A3DConv
x = torch.rand(batch_size, 3, D, H, W)
thickness = torch.rand(batch_size, 1)
# AlignShiftConv to process 3D volumnes
conv = AlignShiftConv(in_channels=3, out_channels=10, kernel_size=3, padding=1, n_fold=8, alignshift=True, ref_thickness=2.0)
out = conv(x, thickness)
# TSMConv to process 3D volumnes
conv = TSMConv(in_channels=3, out_channels=10, kernel_size=3, padding=1, n_fold=8, tsm=True)
out = conv(x)
# A3DConv to process 3D volumnes
conv = A3DConv(in_channels=3, out_channels=10, kernel_size=3, padding=1, dimension=3)
out = conv(x)
```

## Usage of native  AlignShiftConv/TSMConv models

```python
from nn.models import DenseNetCustomTrunc3dAlign, DenseNetCustomTrunc3dTSM
net = DenseNetCustomTrunc3dAlign(num_classes=3)
B, C_in, D, H, W = (1, 3, 7, 256, 256)
input_3d = torch.rand(B, C_in, D, H, W)
thickness = torch.rand(batch_size, 1)
output_3d = net(input_3d, thickness)
```

## How to run the experiments

* Dataset

  * Download [DeepLesion dataset](https://nihcc.box.com/v/DeepLesion)
  * Before training, mask should be generated from bounding box and recists. [mask generation](./deeplesion/dataset/generate_mask_with_grabcut.md)

* Preparing mmdetection script

  * Specify input ct slices in [./deeplesion/mconfigs/densenet_align.py](./deeplesion/mconfigs/densenet_align.py) through modifing NUM_SLICES in dict dataset_transform
  
  * Specify data root in [./deeplesion/ENVIRON.py](./deeplesion/ENVIRON.py)
  
* Model weights
  Our trained weights available on: 
   * [Google Drive](https://drive.google.com/drive/folders/1_ApYs5vb_VzkdqK02lb861Psj-GSdznV?usp=sharing)
   * [百度网盘](https://pan.baidu.com/s/1NsCkvjZdAgi9navg3_ry3g) (h2wc)
   * [ ] [TODO] A3D models are coming soon!


* Training
  ```bash
  ./deeplesion/train_dist.sh ${mmdetection script} ${dist training GPUS}
  ```

  * Train AlignShiftConv models 
  ```bash
  ./deeplesion/train_dist.sh ./deeplesion/mconfig/densenet_align.py 2
  ```

  * Train TSMConv models 
  ```bash
  ./deeplesion/train_dist.sh ./deeplesion/mconfig/densenet_tsm.py 2
  ```
    * Train A3DConv models 
  ```bash
  ./deeplesion/train_dist.sh ./deeplesion/mconfig/densenet_a3d.py 2
  ```

 * Evaluation 
   ```bash
   ./deeplesion/eval.sh ${mmdetection script} ${checkpoint path}
      ```
   ```bash
   ./deeplesion/eval.sh ./deeplesion/mconfig/densenet_align.py ./deeplesion/model_weights/alignshift_7slice.pth
   ```

## Citation
If you find this project useful, please cite the following papers:

    Jiancheng Yang, Yi He, Kaiming Kuang, Zudi Lin, Hanspeter Pfister, Bingbing Ni. "Asymmetric 3D Context Fusion for Universal Lesion Detection". International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2021.
    
    Jiancheng Yang, Yi He, Xiaoyang Huang, Jingwei Xu, Xiaodan Ye, Guangyu Tao, Bingbing Ni. "AlignShift: Bridging the Gap of Imaging Thickness in 3D Anisotropic Volumes". International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2020.
    
or using bibtex:

    @inproceedings{yang2021asymmetric,
      title={Asymmetric 3D Context Fusion for Universal Lesion Detection},
      author={Yang, Jiancheng and He, Yi and Kuang, Kaiming and Lin, Zudi and Pfister, Hanspeter and Ni, Bingbing},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
      pages={571--580},
      year={2021},
      organization={Springer}
    }

    @inproceedings{yang2020alignshift,
      title={AlignShift: bridging the gap of imaging thickness in 3D anisotropic volumes},
      author={Yang, Jiancheng and He, Yi and Huang, Xiaoyang and Xu, Jingwei and Ye, Xiaodan and Tao, Guangyu and Ni, Bingbing},
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
      pages={562--572},
      year={2020},
      organization={Springer}
    }
