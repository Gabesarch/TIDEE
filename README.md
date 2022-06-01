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
For training and running all networks for the tidy task and room rearrangement, start by cloning the repository:
```bash
git clone git@github.com:allenai/ai2thor-rearrangement.git
```
(optional) If you are using conda, create an environment: 
```bash
conda create -n TIDEE python=3.8
```

Install [torch](https://pytorch.org/get-started/locally/) with the CUDA version you have. For example, for CUDA 11.1, you might install via: 
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Next, install additional requirements: 
```bash
pip install -r requirements.txt
```



## Tidy Task

## Out-of-place Detector

## Visual Memex

## Rearrangement Task

