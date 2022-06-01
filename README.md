#<h1 align="left">
    TIDEE: Novel Room Reorganization using Visuo-Semantic Common Sense Priors
</h1>

# TIDEE
This repo contains code and data for TIDEE. If you like this paper, please cite us:
```
```

### Contents
<!--
# To create the table of contents, move the [TOC] line outside of this comment
# and then run the below Python block.
[TOC]
import markdown
with open("README.md", "r") as f:
    a = markdown.markdown(f.read(), extensions=["toc"])
    print(a[:a.index("</div>") + 6])
-->
<div class="toc">
<ul>
<li><a href="#installation"> Installation </a></li>
<li><a href="#tidy-task"> Tidy Task </a><ul>
<li><a href="#out-of-place-detector"> Out-of-place Detector</a></li>
<li><a href="#visual-memex"> Visual Memex</a></li>
<li><a href="#visual-search-network"> Visual Search Network</a></li>
</ul>
<li><a href="#rearrangement-task"> Rearrangement Task </a></li>
</ul>
</div>

## Installation 
(1) For training and running all networks for the tidy task and room rearrangement, start by cloning the repository:
```bash
git clone git@github.com:Gabesarch/TIDEE.git
```
(1a) (optional) If you are using conda, create an environment: 
```bash
conda create -n TIDEE python=3.8
```

(2) Install [PyTorch](https://pytorch.org/get-started/locally/) with the CUDA version you have. For example, run the following for CUDA 11.1: 
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

(3) Install additional requirements: 
```bash
pip install -r requirements.txt
```

(4) Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) with correct PyTorch and CUDA version. 
For example, run the following for PyTorch 1.8.1 & CUDA 11.1:
```bash
pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-sparse==0.6.12      -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-geometric 
```

(5) Install [Detectron 2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) (needed for SOLQ detector) with correct PyTorch and CUDA version. 
E.g. for PyTorch 1.8 & CUDA 11.1:
```bash
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```

(6) Build SOLQ deformable attention:
```bash
cd ./SOLQ/models/ops &&  rm -rf build && sh make.sh && cd ../../..
```


## Tidy Task

## Out-of-place Detector

## Visual Memex

## Rearrangement Task

