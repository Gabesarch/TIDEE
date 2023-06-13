<h1 align="center">
    TIDEE: Tidying Up Novel Rooms using Visuo-Semantic Commonsense Priors
</h1>

<p align="left">
<!--     <a href="//github.com/allenai/ai2thor-rearrangement/blob/main/LICENSE">
        <!-- ai2thor-rearrangement wasn't identifiable by GitHub (on the day this was added), so using the same one as ai2thor -->
<!--         <img alt="License" src="https://img.shields.io/github/license/allenai/ai2thor.svg?color=blue">
    </a> -->
    <a href="https://tidee-agent.github.io/" target="_blank">
        <img alt="Website" src="https://img.shields.io/badge/website-TIDEE-orange">
    </a>
<!--     <a href="//github.com/allenai/ai2thor-rearrangement/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/allenai/ai2thor-rearrangement.svg">
    </a> -->
    <a href="https://arxiv.org/abs/2207.10761" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2103.16544-<COLOR>">
    </a>
<!--     <a href="//arxiv.org/abs/2103.16544" target="_blank">
        <img src="https://img.shields.io/badge/venue-CVPR 2021-blue">
    </a> -->
    <a href="https://youtu.be/wXJuVKeWZmk" target="_blank">
        <img src="https://img.shields.io/badge/video-YouTube-red">
    </a>
<!--     <a href="https://join.slack.com/t/ask-prior/shared_invite/zt-oq4z9u4i-QR3kgpeeTAymEDkNpZmCcg" target="_blank">
        <img src="https://img.shields.io/badge/questions-Ask PRIOR Slack-blue">
    </a> -->
</p>

This branch contains code and data for running TIDEE on the [2023 Room Rearrangement benchmark](https://github.com/allenai/ai2thor-rearrangement#-whats-new-in-the-2023-challenge). 

Note: this is a self-contained branch for running the 2023 Rearrangement Challenge only. Please see the main branch for running the tidy task.

**This version of TIDEE also contains improved mapping and planning!**

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
<li><a href="#installation"> Installation </a></li>
<li><a href="#room-rearrangement-task"> Room Rearrangement Task </a></li><ul>
<li><a href="#pretrained-networks"> Pretrained Networks</a></li>
<li><a href="#running-tidee-on-room-rearrangement"> Running TIDEE on Room Rearrangement</a></li>
<li><a href="#evaluation--videos"> Evaluation & Videos</a></li>
</ul>
<li><a href="#citation"> Citation </a></li>
</div>

## Installation 
**Note:** We have tested this on a remote cluster with CUDA versions 10.2 and 11.1. The dependencies are for running the full TIDEE system. A reduced environment can be used if only running the tidy task and not the TIDEE networks. 

**(1)** For training and running all TIDEE networks for the tidy task and room rearrangement, start by cloning the repository:
```bash
git clone git@github.com:Gabesarch/TIDEE.git
```
**(1a)** (optional) If you are using conda, create an environment: 
```bash
conda create -n TIDEE python=3.8
```

You also will want to set CUDA paths. For example (on our tested machine with CUDA 11.1): 
```bash
export CUDA_HOME="/opt/cuda/11.1.1"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
```

**(2)** Install [PyTorch](https://pytorch.org/get-started/locally/) with the CUDA version you have. For example, run the following for CUDA 11.1: 
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**(3)** Install additional requirements: 
```bash
pip install -r requirements.txt
```

**(4)** Install [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) (needed for SOLQ detector) with correct PyTorch and CUDA version. 
E.g. for PyTorch 1.8 & CUDA 11.1:
```bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```

**(5)** Build SOLQ deformable attention:
```bash
cd ./SOLQ/models/ops && sh make.sh && cd ../../..
```

**(5)** clone [Room Rearrangement](https://github.com/allenai/ai2thor-rearrangement#-whats-new-in-the-2023-challenge) repository and add in custom task base
```bash
sh setup.sh
```

### Remote Server Setup
To run the Ai2THOR simulator on a headless machine, you must either stat an X-server or use Ai2THOR's new headless mode. 
To start an X-server with any of the scripts, you can simply append `--start_startx` to the arguments. You can specify the X-server port use the `--server_port` argument.
Alternatively, you can use [Ai2THOR's new headless rendering](https://ai2thor.allenai.org/ithor/documentation/#headless-setup) by appending `--do_headless_rendering` to the arguments. 

# Room Rearrangement Task

## Pretrained networks
All pretrained model checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1KpTL6Kp5Hk_paFXTPMtfO4H6Ur8LTGph?usp=sharing). 

For use with the tidy task or room rearrangement, place all checkpoints directly in the `checkpoints` folder. 

## Running TIDEE on Room Rearrangement 
The evaluation code for the rearrangement challenge task is taken from [Visual Room Rearrangement](https://github.com/allenai/ai2thor-rearrangement) and is included in the current repo in `rearrangement` modified to include estimated depth, noisy pose, noisy depth, and TIDEE config.

To run TIDEE on the 2023 rearrangement benchmark combined set (train, val, test), run (for example) the following: 
```
python main.py --mode rearrangement --tag TIDEE_rearrengement_2023 --OT_dist_thresh 1.0 --thresh_num_dissimilar -1 --match_relations_walk --HORIZON_DT 30 --log_every 25 --dataset 2023 --eval_split combined
```

All metrics will be saved in the folder `metrics` every `log_every` episodes (specified by arguments). 

Note: TIDEE does not support open/close for the 2023 version of the rearrangement benchmark.

Noisy measurements: 
(1) To run using estimated depth, append `--estimate_depth`.

(2) To run using noisy pose, append `--noisy_pose`.

(3) To run using noisy depth, append `--noisy_depth`.

## Evaluation & Videos
Evaluation images can be logged by adding (for example) the following to the arguments:
```
--log_every 1 --save_object_images --image_dir tidy_task
```

And an .mp4 movie of each episode can be logged by adding (for example) the following to the arguments:
```
--create_movie --movie_dir tidy_task
```

# Citation
If you like this paper, please cite us:
```
@inproceedings{sarch2022tidee,
            title = "TIDEE: Tidying Up Novel Rooms using Visuo-Semantic Common Sense Priors",
            author = "Sarch, Gabriel and Fang, Zhaoyuan and Harley, Adam W. and Schydlo, Paul and Tarr, Michael J. and Gupta, Saurabh and Fragkiadaki, Katerina", 
            booktitle = "European Conference on Computer Vision",
            year = "2022"}
```

