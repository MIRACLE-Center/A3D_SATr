
# Code for paper "SATr: Slice Attention with Transformer for Universal Lesion Detection"

This repository contains the *state-of-the-art* version (A3D-SATr) of our paper SATr (MICCAI'22). 
*SATr: Slice Attention with Transformer for Universal Lesion Detection ([MICCAI'22](https://arxiv.org/abs/2203.07373))

本项目为MICCAI22文章 "SATr: Slice Attention with Transformer for Universal Lesion Detection" 的开源代码，因为我们的方法具有通用性，我们仅开源SOAT的版本, A3D-SATr

## Code structure
* ``main structure``
The A3D-SATr version is heavily based on the A3D work ([MICCAI'21](https://github.com/M3DV/AlignShift)) and [mmdetection](https://github.com/open-mmlab/mmdetection)
The main structure please follow the above mentioned two repository.
* ``SATr structure``
Our code modifications mainly located in the *Class Trans_with_A3D* (line 753 in nn\models\truncated_densenet3d_a3d.py)

## Installation

 * git clone this repository
 * pip install -e . 
 
The code requires only common Python environments for machine learning. Basically, it was tested with
Python 3 (>=3.6)
PyTorch==1.3.1
numpy==1.18.5, pandas==0.25.3, scikit-learn==0.22.2, Pillow==8.0.1, fire, scikit-image
Higher (or lower) versions should also work (perhaps with minor modifications).
