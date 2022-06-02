<h1 align="center">
    TIDEE: Novel Room Reorganization using Visuo-Semantic Common Sense Priors
</h1>

<p align="left">
<!--     <a href="//github.com/allenai/ai2thor-rearrangement/blob/main/LICENSE">
        <!-- ai2thor-rearrangement wasn't identifiable by GitHub (on the day this was added), so using the same one as ai2thor -->
<!--         <img alt="License" src="https://img.shields.io/github/license/allenai/ai2thor.svg?color=blue">
    </a> -->
    <a href="//www.gabesarch.me/" target="_blank">
        <img alt="Website" src="https://img.shields.io/badge/website-TIDEE-orange">
    </a>
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
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f \ 
    https://download.pytorch.org/whl/torch_stable.html
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

### Remote Server Setup
To run the Ai2THOR simulator on a headless machine, you must either stat an X-server or use Ai2THOR's new headless mode. 
To start an X-server with any of the scripts, you can simply append `--startx` to the arguments. 
Alternatively, you can use [Ai2THOR's new headless rendering](https://ai2thor.allenai.org/ithor/documentation/#headless-setup) by appending `--do_headless_rendering` to the arguments. 

## Tidy Task
The Tidy Task involves detecting and moving out of place objects to plausible places within the scene without any instructions. You can see `task_base/messup.py` for our data generation code to move objects out of place. See `task_base/example.py` for an example script of running the task with random actions. 

### Dataset
Our tidy task dataset contains `8000` training scenes, `200` validation scenes, and `100` testing scenes with five objects in each scene moved out of place. To run the tidy task with the dataset, download the scene metadata from [here]() and place the extracted folder inside the `data/` folder.  

### Out of Place Detector
For our out of place detectors, we first train [SOLQ](https://github.com/megvii-research/SOLQ) with two prediction heads (one for category, one for out of place). See `models/aithor_solq.py` and `models/aithor_solq_base.py` for code details, and `arguments.py` for training argument details. 

```
python main.py --mode solq --S 5 --data_batch_size 5 --lr_drop 7 --run_val --load_val_agent --val_load_dir ./data/val_data/aithor_tidee_oop --plot_boxes --plot_masks --randomize_scene_lighting_and_material --start_startx --do_predict_oop --load_base_solq --mess_up_from_loaded --log_freq 250 --val_freq 250 --set_name TIDEE_solq_oop
```

To train the visual and language detector, you can run the following (see `models/aithor_bert_oop_visual.py` and `models/aithor_solq_base.py` for details): 
```
python main.py --mode visual_bert_oop --do_visual_and_language_oop --S 3 --data_batch_size 3 --run_val --load_val_agent \ --val_load_dir ./data/val_data/aithor_tidee_oop_VL --n_val 3 --load_train_agent --train_load_dir ./data/train_data/aithor_tidee_oop_VL --n_train 50 \ --randomize_scene_lighting_and_material --start_startx --do_predict_oop --mess_up_from_loaded  --save_freq 2500 --log_freq 250 --val_freq 250 --max_iters 25000 --keep_latest 5 --start_one --score_threshold_oop 0.0 --score_threshold_cat 0.0 --set_name TIDEE_oop_vis_lang
```

The above will generate training and validation data from the simulator. If you would like to use our training data from the paper, Please contact us and we would be happy to share it. 

### Visual Memex

### Visual Search Network

### Pretrained networks

## Rearrangement Task

## Citation
If you like this paper, please cite us:
```
```

