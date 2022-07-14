<h1 align="center">
    TIDEE: Novel Room Reorganization using Visuo-Semantic Common Sense Priors
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
    <a href="https://tidee-agent.github.io/" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2103.16544-<COLOR>">
    </a>
<!--     <a href="//arxiv.org/abs/2103.16544" target="_blank">
        <img src="https://img.shields.io/badge/venue-CVPR 2021-blue">
    </a> -->
    <a href="https://tidee-agent.github.io/" target="_blank">
        <img src="https://img.shields.io/badge/video-YouTube-red">
    </a>
<!--     <a href="https://join.slack.com/t/ask-prior/shared_invite/zt-oq4z9u4i-QR3kgpeeTAymEDkNpZmCcg" target="_blank">
        <img src="https://img.shields.io/badge/questions-Ask PRIOR Slack-blue">
    </a> -->
</p>

This repo contains code and data for running TIDEE and the tidy task. 

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
<li><a href="#running-the-task"> Running the task</a></li>
<li><a href="#dataset"> Dataset</a></li>
</ul>
<li><a href="#TIDEE"> TIDEE </a><ul>
<li><a href="#running-tidee-on-the-tidy-task"> Running TIDEE on the tidy task</a></li>
<li><a href="#evaluation--videos"> Evaluation & Videos</a></li>
<li><a href="#out-of-place-detector"> Out-of-place Detector</a></li>
<li><a href="#visual-memex"> Visual Memex</a></li>
<li><a href="#visual-search-network"> Visual Search Network</a></li>
<li><a href="#pretrained-networks"> Pretrained Networks</a></li>
</ul>
<li><a href="#room-rearrangement-task"> Room Rearrangement Task </a></li>
<li><a href="#citation"> Citation </a></li>
</ul>
</div>

## Installation 
**Note:** A reduced environment can be used if only running the tidy task and not the TIDEE networks. 

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
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```

**(6)** Build SOLQ deformable attention:
```bash
cd ./SOLQ/models/ops && sh make.sh && cd ../../..
```

### Remote Server Setup
To run the Ai2THOR simulator on a headless machine, you must either stat an X-server or use Ai2THOR's new headless mode. 
To start an X-server with any of the scripts, you can simply append `--start_startx` to the arguments. You can specify the X-server port use the `--server_port` argument.
Alternatively, you can use [Ai2THOR's new headless rendering](https://ai2thor.allenai.org/ithor/documentation/#headless-setup) by appending `--do_headless_rendering` to the arguments. 

# Tidy Task

## Running the task
The Tidy Task involves detecting and moving out of place objects to plausible places within the scene without any instructions. You can see `task_base/messup.py` for our data generation code to move objects out of place. See `task_base/example.py` for an example script of running the task with random actions. To run the tidy task, the tidy task dataset must be downloaded (see <a href="#dataset"> Dataset</a>)

## Dataset
Our tidy task dataset contains `8000` training scenes, `200` validation scenes, and `100` testing scenes with five objects in each scene moved out of place. To run the tidy task with the generated scenes, download the scene metadata from [here](https://drive.google.com/file/d/1KFUxxL8KU4H8dxBpjhp1SGAf3qnTtEBM/view?usp=sharing) and place the extracted contents inside of the `data` folder.  

# TIDEE

## Running TIDEE on the tidy task
To run the full TIDEE pipeline on the tidy task, do the following: 

(1) Download all model checkpoints (see <a href="#pretrained-networks"> Pretrained Networks</a>) and add them to `checkpoints`. Then, download the tidy task dataset (see <a href="#dataset"> Dataset</a>) and add it to the `data` folder. 

(2) Download the visual memex graph data from [here](https://drive.google.com/file/d/1W6jMJOCZVSFOslKqhZKWxsKtyX6Y05pb/view?usp=sharing), and place the pickle file in the `data` folder.

(3) Run TIDEE on the tidy task using the following command: 
```
python main.py --mode TIDEE --do_predict_oop --eval_split test --do_visual_memex --do_vsn_search --do_visual_oop --do_add_semantic
```

## Evaluation & Videos
Evaluation images can be logged by adding (for example) the following to the arguments:
```
--log_every 1 --save_object_images --image_dir tidy_task
```

And an .mp4 movie of each episode can be logged by adding (for example) the following to the arguments:
```
--create_movie --movie_dir tidy_task
```

## Out of Place Detector
This section details how to train the Out of Place Detector.

We first train [SOLQ](https://github.com/megvii-research/SOLQ) with two prediction heads (one for category, one for out of place). See `models/aithor_solq.py` and `models/aithor_solq_base.py` for code details, and `arguments.py` for training argument details. 

```
python main.py --mode solq --S 5 --data_batch_size 5 --lr_drop 7 --run_val --load_val_agent --val_load_dir ./data/val_data/aithor_tidee_oop --plot_boxes --plot_masks --randomize_scene_lighting_and_material --start_startx --do_predict_oop --load_base_solq --mess_up_from_loaded --log_freq 250 --val_freq 250 --set_name TIDEE_solq_oop
```

To train the visual and language detector, you can run the following (see `models/aithor_bert_oop_visual.py` and `models/aithor_solq_base.py` for details): 
```
python main.py --mode visual_bert_oop --do_visual_and_language_oop --S 3 --data_batch_size 3 --run_val --load_val_agent --val_load_dir ./data/val_data/aithor_tidee_oop_VL --n_val 3 --load_train_agent --train_load_dir ./data/train_data/aithor_tidee_oop_VL --n_train 50 --randomize_scene_lighting_and_material --start_startx --do_predict_oop --mess_up_from_loaded  --save_freq 2500 --log_freq 250 --val_freq 250 --max_iters 25000 --keep_latest 5 --start_one --score_threshold_oop 0.0 --score_threshold_cat 0.0 --set_name TIDEE_oop_vis_lang
```
The above will generate training and validation data from the simulator if the data does not already exist. 

## Neural Associative Memory Graph Network
This section details how to train the Neural Associative Memory Graph Network.

To train the visual memex, the following steps are required: 

(1) Make sure you have the SOLQ checkpoint (see <a href="#pretrained-networks"> Pretrained Networks</a>) in `checkpoints`. 

(2) (skip if already done for Visual Search Network) first generate some observations of the mapping phase to use for the scene graph features. These can be generated and saved by running the following command:
```
python main.py --mode generate_mapping_obs --start_startx --do_predict_oop --mapping_obs_dir ./data/mapping_obs
```
This will generate the mapping observations to `mapping_obs_dir` (Note: this data will be ~200GB). 
Or, alternatively, download the mapping observations from [here](https://drive.google.com/file/d/1LW4zUDRkirtiDuQQzOgMgJrvvlsDvxIC/view?usp=sharing) and place the extracted contents in `./data/`.

(3) Train the graph network (see `models/aithor_visrgcn.py` and `models/aithor_visrgcn_base.py` for details):
```
python main.py --mode visual_memex --run_val --load_val_agent --do_predict_oop --radius_max 3.0 --num_mem_houses 5 --num_train_houses 15 --load_visual_memex --do_load_oop_nodes_and_supervision --vmemex_supervision_dir /projects/katefgroup/project_cleanup/tidee_final/vmemex_supervision_dir --only_include_receptacle --objects_per_scene 15 --scenes_per_batch 10 --mapping_obs_dir ./data/mapping_obs --load_model --load_model_path ./checkpoints/vrgcn-00002000.pth --set_name tidee_vmemex07 
```

## Visual Search Network
This section details how to train the Visual Search Network.

To train the Visual Search Network, the following steps are required: 

(1) Make sure you have the SOLQ checkpoint (see <a href="#pretrained-networks"> Pretrained Networks</a>) in `checkpoints`. 

(2) (skip if already done for Neural Associative Memory Graph Network) first generate some observations of the mapping phase. These can be generated and saved by running the following command:
```
python main.py --mode generate_mapping_obs --start_startx --do_predict_oop --mapping_obs_dir ./data/mapping_obs
```
This will generate the mapping observations to `mapping_obs_dir` (Note: this data will be ~200GB).
Or, alternatively, download the mapping observations from [here](https://drive.google.com/file/d/1LW4zUDRkirtiDuQQzOgMgJrvvlsDvxIC/view?usp=sharing) and place the extracted contents in `./data/`.

(3) Train the graph network (see `models/aithor_visualsearch.py` and `models/aithor_visualsearch_base.py` for details):
```
python main.py --mode visual_search_network --run_val --objects_per_scene 3 --scenes_per_batch 6 --n_val 8 --objects_per_scene_val 2 --mapping_obs_dir ./data/mapping_obs --do_add_semantic --log_freq 250 --val_freq 250 --set_name tidee_vsn
```

### Object goal navigation evaluation
To run the object goal navigation evaluation from the paper using the Visual Search Network, run: 
```
python main.py --mode visual_search_network --eval_object_nav --object_navigation_policy_name vsn_search --load_model --load_model_path ./checkpoints/vsn-00013500.pth --tag tidee_object_nav_vsn --do_predict_oop --detector_threshold_object_nav 0.5 --visibilityDistance 1.0 --max_steps_object_goal_nav 200 --nms_threshold 0.5 
```
To run the object goal navigation evaluation from the paper without the Visual Search Network, run: 
```
python main.py --mode visual_search_network --eval_object_nav --object_navigation_policy_name random --tag tidee_object_nav_novsn --do_predict_oop --detector_threshold_object_nav 0.5 --visibilityDistance 1.0 --max_steps_object_goal_nav 200 --nms_threshold 0.5
```

## Pretrained networks
All pretrained model checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1KpTL6Kp5Hk_paFXTPMtfO4H6Ur8LTGph?usp=sharing). 

For use with the tidy task or room rearrangement, place all checkpoints directly in the `checkpoints` folder. 

# Room Rearrangement Task

## Running TIDEE on Room Rearrangement 
The evaluation code for the rearrangement challenge task is taken from [Visual Room Rearrangement](https://github.com/allenai/ai2thor-rearrangement) and is included in the current repo in `rearrangement` modified to include estimated depth, noisy pose, noisy depth, and TIDEE config.

To run TIDEE on the 2022 rearrangement benchmark combined set (train, val, test), run (for example) the following: 
```
python main.py --mode rearrangement --tag TIDEE_rearrengement_2022 --OT_dist_thresh 1.0 --thresh_num_dissimilar -1 --match_relations_walk --HORIZON_DT 30 --log_every 25 --dataset 2022 --eval_split combined
```

To run TIDEE on the 2021 rearrangement benchmark combined set (train, val, test), run (for example) the following: 
```
python main.py --mode rearrangement --tag TIDEE_rearrengement_2021 --OT_dist_thresh 1.0 --thresh_num_dissimilar -1 --match_relations_walk --HORIZON_DT 30 --log_every 25 --dataset 2021 --eval_split combined
```

All metrics will be saved in the folder `metrics` every `log_every` episodes (specified by arguments). 

To run with the open and close, append `--do_open`.

Noisy measurements: 
(1) To run using estimated depth, append `--estimate_depth`.

(2) To run using noisy pose, append `--noisy_pose`.

(3) To run using noisy depth, append `--noisy_depth`.

# Citation
If you like this paper, please cite us:
```
```

