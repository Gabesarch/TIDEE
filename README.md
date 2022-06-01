<h1 align="center">
    TIDEE: Novel Room Reorganization using Visuo-Semantic Common Sense Priors
</h1>

<p align="left">
<!--     <a href="//github.com/allenai/ai2thor-rearrangement/blob/main/LICENSE">
        <!-- ai2thor-rearrangement wasn't identifiable by GitHub (on the day this was added), so using the same one as ai2thor -->
        <img alt="License" src="https://img.shields.io/github/license/allenai/ai2thor.svg?color=blue">
    </a> -->
<!--     <a href="//ai2thor.allenai.org/rearrangement/" target="_blank">
        <img alt="Documentation" src="https://img.shields.io/website/https/ai2thor.allenai.org?down_color=red&down_message=offline&up_message=online">
    </a> -->
<!--     <a href="//github.com/allenai/ai2thor-rearrangement/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/allenai/ai2thor-rearrangement.svg">
    </a> -->
    <a href="//arxiv.org/" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2103.16544-<COLOR>">
    </a>
<!--     <a href="//arxiv.org/abs/2103.16544" target="_blank">
        <img src="https://img.shields.io/badge/venue-CVPR 2021-blue">
    </a> -->
    <a href="//www.gabesarch.me/" target="_blank">
        <img src="https://img.shields.io/badge/video-YouTube-red">
    </a>
<!--     <a href="https://join.slack.com/t/ask-prior/shared_invite/zt-oq4z9u4i-QR3kgpeeTAymEDkNpZmCcg" target="_blank">
        <img src="https://img.shields.io/badge/questions-Ask PRIOR Slack-blue">
    </a> -->
</p>

This repo contains code and data for running and training TIDEE. 

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
<li><a href="#citation"> Citation </a></li>
</ul>
</div>

## Installation 
**(1)** For training and running all networks for the tidy task and room rearrangement, start by cloning the repository:
```bash
git clone git@github.com:Gabesarch/TIDEE.git
```
**(1a)** (optional) If you are using conda, create an environment: 
```bash
conda create -n TIDEE python=3.8
```

**(2)** Install [PyTorch](https://pytorch.org/get-started/locally/) with the CUDA version you have. For example, run the following for CUDA 11.1: 
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**(3)** Install additional requirements: 
```bash
pip install -r requirements.txt
```

**(4)** Install [PyG](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) with correct PyTorch and CUDA version. 
For example, run the following for PyTorch 1.8.1 & CUDA 11.1:
```bash
pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-sparse==0.6.12      -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch-geometric 
```

**(5)** Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) (needed for SOLQ detector) with correct PyTorch and CUDA version. 
E.g. for PyTorch 1.8 & CUDA 11.1:
```bash
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```

**(6)** Build SOLQ deformable attention:
```bash
cd ./SOLQ/models/ops &&  rm -rf build && sh make.sh && cd ../../..
```

## Tidy Task

## Out-of-place Detector

## Visual Memex

## Rearrangement Task

## Citation
If you like this paper, please cite us:
```
```

