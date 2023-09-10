import argparse
import numpy as np
import os
parser = argparse.ArgumentParser()


parser.add_argument("--seed", type=int, default=43, help="Random seed")
parser.add_argument("--mode", type=str, help="mode to run, see main.py")



###########%%%%%%% agent parameters %%%%%%%###########
parser.add_argument("--start_startx", action="store_true", default=False, help="start x server upon calling main")
parser.add_argument("--do_headless_rendering", action="store_true", default=False, help="render in headless mode with new Ai2thor version")
parser.add_argument("--HORIZON_DT", type=int, default=30, help="pitch movement delta")
parser.add_argument("--DT", type=int, default=90, help="yaw movement delta")
parser.add_argument("--STEP_SIZE", type=int, default=0.25, help="yaw movement delta")
parser.add_argument("--fov", type=int, default=90, help="field of view")
parser.add_argument("--W", type=int, default=480, help="image width")
parser.add_argument("--H", type=int, default=480, help="image height")
parser.add_argument("--visibilityDistance", type=float, default=1.5, help="visibility NOTE: this will not change rearrangement visibility")
parser.add_argument("--dont_use_controller", action="store_true", default=False, help="set to True if do not want to init the controller")
parser.add_argument("--dpi", type=int, default=100, help="dpi for matplotlib. higher value will give higher res movies but slower to create.")

parser.add_argument("--max_traj_steps", type=int, default=1000, help="maximum trajectory steps")
parser.add_argument("--max_api_fails", type=int, default=30, help="maximum allowable api failures")
parser.add_argument("--metrics_dir", type=str, default="./metrics", help="where to output rendered movies")
parser.add_argument('--skip_if_exists', default=False, action='store_true', help='skip if file exists in teach metrics')
parser.add_argument("--episode_in_try_except", action="store_true", default=False, help="Continue to next episode if assertion error occurs? ")


### WANDB
parser.add_argument("--group", type=str, default="default", help="group name")
parser.add_argument("--wandb_directory", type=str, default='./wandb', help="Path to wandb metadata")

###########%%%%%%% splits %%%%%%%###########
parser.add_argument("--num_mem_houses", type=int, default=0, help="num memory houses")
parser.add_argument("--num_train_houses", type=int, default=20, help="num train houses")
parser.add_argument("--num_val_houses", type=int, default=5, help="num val houses")
parser.add_argument("--num_test_houses", type=int, default=5, help="num test houses")
parser.add_argument("--do_kitchen", type=bool, default=True, help="use kitchens")
parser.add_argument("--do_living_room", type=bool, default=True, help="use kitchens")
parser.add_argument("--do_bedroom", type=bool, default=True, help="use kitchens")
parser.add_argument("--do_bathroom", type=bool, default=True, help="use kitchens")
parser.add_argument("--eval_split", type=str, default="test", help="evaluation mode: combined (rearrange), train, test, val")

###########%%%%%%% TIDEE task %%%%%%%###########
parser.add_argument("--save_object_images", action="store_true", default=False, help="save object images after each phase (used for MTurk)")
parser.add_argument("--do_visual_memex", action="store_true", default=False, help="use visual memex")
parser.add_argument("--do_most_common_memory", action="store_true", default=False, help="most common memory baseline")
parser.add_argument("--do_vsn_search", action="store_true", default=False, help="use visual search network for searching for objects")
parser.add_argument("--do_visual_oop", action="store_true", default=False, help="use visual oop detector for oop detection")
parser.add_argument("--verbose", action="store_true", default=False, help="print out actions + other logs during task")
parser.add_argument("--vsn_checkpoint", type=str, default="./checkpoints/vsn-00013500.pth", help="checkpoint for visual search network")
parser.add_argument("--visual_memex_checkpoint", type=str, default="./checkpoints/vrgcn-00002000.pth", help="checkpoint for visual memex network")
parser.add_argument("--num_search_locs_object", type=int, default=3, help="number of search locations when searching for an object by visual search network")
parser.add_argument("--num_related_objects", type=int, default=3, help="number of top k related objects to try if cannot find the initial ones")
parser.add_argument("--num_search_locs_oop", type=int, default=5, help="number of search locations when searching for an OOP object")
parser.add_argument("--max_episode_steps", type=int, default=1000, help="maximum steps for an episode")
parser.add_argument("--score_threshold_oop", type=float, default=0.6, help="out of place detector threshold")
parser.add_argument("--score_threshold_interaction", type=float, default=0.3, help="score threshold for redetection for getting 2D point for interaction")
parser.add_argument("--num_pipeline_attempts", type=int, default=10, help="how many times to attempt to go through TIDEE pipeline before stopping - this can speed up episodes")
parser.add_argument("--visualize_memex", action="store_true", default=False, help="visualize memex inference in the video")
parser.add_argument("--visualize_vsn", action="store_true", default=False, help="visualize VSN inference in the video")
parser.add_argument("--visualize_masks", action="store_true", default=False, help="visualize object tracker masks")

###########%%%%%%% rearrangement challenge %%%%%%%###########
parser.add_argument("--noisy_pose", action="store_true", default=False, help="add guassian noise to movements based on Locobot measurements")
# parser.add_argument("--movementGaussianSigma", type=float, default=None, help="add guassian noise to translation if noisy pose")
# parser.add_argument("--rotateGaussianSigma", type=float, default=None, help="add guassian noise to rotation if noisy pose")
parser.add_argument("--estimate_depth", action="store_true", default=False, help="use estimated depth maps")
parser.add_argument("--noisy_depth", action="store_true", default=False, help="use noisy depth for rearrangement")
parser.add_argument("--max_relations", type=int, default=50, help="maximum number of object relations")
parser.add_argument("--dissimilar_threshold", type=float, default=0.35, help="threshold for percent of relations for object to be out of place")
parser.add_argument("--tag", type=str, default="test00", help="tag for metric file")
parser.add_argument("--thresh_num_dissimilar", type=int, default=-1, help="threshold for minimum number of dissimilar for object to be out of place")
parser.add_argument("--use_solq", type=bool, default=True, help="use SOLQ detector")
parser.add_argument("--use_masks", type=bool, default=True, help="use masks > boxes")
parser.add_argument("--rearrange_interval_min", type=int, default=None, help="interval range for rearrange iterations")
parser.add_argument("--rearrange_interval_max", type=int, default=None, help="interval range for rearrange iterations")
parser.add_argument("--shuffle_maps", action="store_true", default=False, help="shuffle maps during eval")
parser.add_argument("--load_submission", action="store_true", default=False, help="pick up where left off")
parser.add_argument("--server_port", type=int, default=0, help="x server port")
parser.add_argument("--dataset", type=str, default="2022", help="OPTIONS: 2021, 2022")
parser.add_argument("--get_pose_change_from_GT", action="store_true", default=False, help="get pose changes from GT")
parser.add_argument("--use_GT_action_success", action="store_true", default=False, help="use GT for action success")
parser.add_argument("--use_GT_masks", action="store_true", default=False, help="use GT masks")
parser.add_argument("--use_GT_centroids", action="store_true", default=False, help="use GT centroids for objects in view")
parser.add_argument("--use_GT_centroids_from_meta", action="store_true", default=False, help="get object centroids from meta data (all objects in the scene)")
parser.add_argument("--do_open", action="store_true", default=False, help="also consider open and close")
parser.add_argument("--only_one_obj_per_cat", action="store_true", default=False, help="only use one object per category in object tracker")
parser.add_argument("--loop_through_cat", action="store_true", default=False, help="loop through all category instances")
parser.add_argument("--match_relations_walk", action="store_true", default=False, help="assume that ranked scoring detections match between stages")
parser.add_argument("--generate_rearrangement_images", action="store_true", default=False, help="generate images of the objects to be rearranged")
parser.add_argument("--log_relations_txt", action="store_true", default=False, help="generate txt log of relations")


###########%%%%%%% navigation %%%%%%%###########
parser.add_argument("--dist_thresh", type=int, default=0.5, help="navigation distance threshold to point goal")
parser.add_argument("--SOLQ_checkpoint", type=str, default="./checkpoints/solq-00018500.pth", help="checkpoint for SOLQ")
parser.add_argument("--depth_checkpoint_45", type=str, default="./checkpoints/model-2000-best_silog_10.13741", help="Depth checkpoint trained at pitch 45 degrees")
parser.add_argument("--depth_checkpoint_0", type=str, default="./checkpoints/model-102500-best_silog_17.00430", help="Depth checkpoint trained at pitch 0 degrees")

###########%%%%%%% object tracker %%%%%%%###########
parser.add_argument("--OT_dist_thresh", type=float, default=1.0, help="distance threshold for NMS for object tracker")
parser.add_argument("--confidence_threshold", type=float, default=0.4, help="confidence threshold for detections [0, 0.1]")
parser.add_argument("--confidence_threshold_searching", type=float, default=0.2, help="confidence threshold for detections when searching for a target object class [0, 0.1]")
parser.add_argument("--nms_threshold", type=float, default=0.5, help="NMS threshold for object tracker")

###########%%%%%%% logging %%%%%%%###########
parser.add_argument("--log_every", type=int, default=5, help="how often to log movies, etc.")
parser.add_argument("--create_movie", action="store_true", default=False, help="create mp4 movie")
parser.add_argument("--movie_dir", type=str, default="./images", help="where to output rendered movies")
parser.add_argument("--MAX_QUEUE", type=int, default=10, help="max queue for tensorboard")
parser.add_argument("--image_dir", type=str, default="./images", help="where to output rendered images")

parser.add_argument("--set_name", type=str, default="test00", help="name of experiment")
parser.add_argument("--data_path", type=str, default="./data", help="path to data")
parser.add_argument('--batch_size', default=None, type=int, help="batch size for model training. If None, will be S*data_batch_size. batch_size must be <= S*data_batch_size.")
parser.add_argument('--keep_latest', default=3, type=int, help="number of checkpoints to keep at one time")

###########%%%%%%% data generation parameters for detector training %%%%%%%###########
parser.add_argument("--data_mode", type=str, default="solq", help="mode for detector (only option currently: SOLQ)")
parser.add_argument("--radius_min", type=float, default=0.0, help="radius min to spawn near target object")
parser.add_argument("--radius_max", type=float, default=7.0, help="radius max to spawn near target object")
parser.add_argument("--nbins", type=int, default=30, help="Number of yaw bins to consider around object")
parser.add_argument("--S", type=int, default=5, help="Number of views per trajectory")
parser.add_argument("--visibility_threshold", type=float, default=0.005, help="minimum visibility/occlusion for object to be used as supervision")
parser.add_argument("--min_percent_points", type=float, default=0.0001, help="minimum percent of image points for object to be used as supervision")
parser.add_argument("--views_to_attempt", type=int, default=8, help="max views to attempt for getting trajectory")
parser.add_argument("--amodal", type=bool, default=True, help="Train with amodal boxes (masks are always modal)")
parser.add_argument("--do_masks", type=bool, default=True, help="train with masks")
parser.add_argument("--fail_if_no_objects", type=bool, default=True, help="fail view if no objects in view")
parser.add_argument("--movement_mode", type=str, default="forward_first", help="movement mode for action sampling for getting trajectory (forward_first, random); forward_first: always try to move forward")
parser.add_argument("--data_batch_size", type=int, default=5, help="number of trajectories per data generation batch")
parser.add_argument("--randomize_object_state", action="store_true", default=False, help="randomize object states during data generation (dirty, cooked, filled, toggle on/off, etc.)")
parser.add_argument("--randomize_object_placements", action="store_true", default=False, help="randomize object locations in the room")
# parser.add_argument("--openness_increments", type=int, default=0.1, help="degree of openness intervals if randomize_object_state is TRUE")
parser.add_argument("--randomize_scene_lighting_and_material", action="store_true", default=False, help="randomize room lighting and object material")
parser.add_argument("--log_freq", type=int, default=250, help="how often to log to tensorboard in iterations")
parser.add_argument("--lr_scheduler_freq", type=int, default=500, help="how often to step LR scheduler in iterations")
parser.add_argument("--run_val", action="store_true", default=False, help="run validation every val_freq iters")
parser.add_argument("--val_freq", type=int, default=250, help="how often to run validation")
parser.add_argument("--save_freq", type=int, default=500, help="how often to save a checkpoint")
parser.add_argument("--plot_boxes", action="store_true", default=False, help="plot boxes to tensorboard during log iters")
parser.add_argument("--plot_masks", action="store_true", default=False, help="plot masks to tensorboard during log iters")
parser.add_argument("--score_threshold", type=float, default=0.5, help="score threshold for plotting boxes")

###########%%%%%%% DDETR/SOLQ out of place detector %%%%%%%###########
# parser.add_argument("--two_pred_heads", action="store_true", default=False, help="add a second prediction head to SOLQ for out of place prediction")
parser.add_argument("--do_predict_oop", action="store_true", default=False, help="train for out of place prediction")
parser.add_argument("--mess_up_from_loaded", action="store_true", default=False, help="create mess up scene from loaded file")
parser.add_argument("--mess_up_dir", type=str, default="./data/messup/", help="create mess up scene from loaded file")
parser.add_argument("--num_objects", type=int, default=5, help="number of objects to move out of place per scene")
parser.add_argument("--n_train", type=int, default=100, help="maximum number of samples to save per room for TRAINING")
parser.add_argument("--n_val", type=int, default=10, help="maximum number of samples to save per room for VALIDATION")
parser.add_argument("--n_test", type=int, default=5, help="maximum number of samples to save per room for TESTING")
parser.add_argument("--n_train_messup", type=int, default=100, help="maximum number of messup configurations to save per room for TRAINING")
parser.add_argument("--n_val_messup", type=int, default=10, help="maximum number of messup configurations to save per room for VALIDATION")
parser.add_argument("--n_test_messup", type=int, default=3, help="maximum number of messup configurations to save per room for TESTING")

###########%%%%%%% visual bert out of place detector %%%%%%%###########
parser.add_argument("--SOLQ_oop_checkpoint", type=str, default="./checkpoints/solq_oop-00010500.pth", help="checkpoint for out of place SOLQ detector")
parser.add_argument("--do_visual_and_language_oop", action="store_true", default=False, help="train language and visual")
parser.add_argument("--do_visual_only_oop", action="store_true", default=False, help="train language and visual")
parser.add_argument("--do_language_only_oop", action="store_true", default=False, help="train language and visual")
parser.add_argument("--freeze_layers", type=int, default=0, help="number of BERT laayers to freeze")
parser.add_argument("--num_each_oop", type=int, default=2, help="number of OOP per batch (to get class balance)")
parser.add_argument("--use_gt_centroids_and_labels", action="store_true", default=False, help="use GT centroids and labels for training")
parser.add_argument("--visualize_relations", action="store_true", default=False, help="visualize relations in overhead view")
parser.add_argument("--finetune_on_one_object", action="store_true", default=False, help="augmented training for changing priors")
parser.add_argument("--eval_test_set", action="store_true", default=False, help="evaluate detector on test set")
parser.add_argument("--test_load_dir", type=str, default="./data/TIDEE_test/", help="where to save and load test data")
parser.add_argument("--iou_det_thresh", type=float, default=0.3, help="iou threshold for getting GT match to detected match for supervision")
parser.add_argument("--lr_vboop", type=float, default=2e-7, help="learning rate visual bert oop detector")
parser.add_argument("--run_data_ordered", action="store_true", default=False, help="run training data in order (as opposed to randomly choosing)")
parser.add_argument("--score_threshold_cat", type=float, default=0.4, help="confidence threshold for detectionsof category [0, 0.1]")
parser.add_argument("--load_object_tracker", action="store_true", default=False, help="load object tracker during training")
parser.add_argument("--explore_env", action="store_true", default=False, help="expore env to populate object_tracker")
parser.add_argument("--do_GT_relations", action="store_true", default=False, help="use GT relations as input")

###########%%%%%%% visual MEMEX %%%%%%%###########
parser.add_argument("--num_tries_memory", type=int, default=20, help="number of times to visit object in memory to get features")
parser.add_argument("--in_channels", type=int, default=1024, help="feature size for rgcn in")
parser.add_argument("--visual_feat_size_after_proj", type=int, default=512, help="visual feature size after linear projection. must be < in_channels.")
# parser.add_argument("--out_channels", type=int, default=1024, help="feature size for rgcn out")
parser.add_argument("--lr_vrgcn", type=float, default=2e-5, help="learning rate visual rgcn")
parser.add_argument('--weight_decay_vrgcn', default=1e-4, type=float)
# parser.add_argument("--remove_connect_mem_with_scene", action="store_true", default=False, help="connect memory graph with scene graph")
parser.add_argument("--without_memex", action="store_true", default=False, help="ablation to not use memex in the rGCN (only scene graph + oop)")
parser.add_argument("--remove_sg_layers", action="store_true", default=False, help="remove GCN layers over fully-connected scene graph")
parser.add_argument("--visual_memex_path", type=str, default="./data/visual_memex.p", help="location where visual memex is saved")
parser.add_argument("--load_visual_memex", action="store_true", default=False, help="load visual memex from memory (NOTE: includes ddetr visual deatures)")
parser.add_argument("--backbone_layer_ddetr", type=int, default=0, help="ddetr resnet backbone layer to get visual features from (options: 0, 1, 2)")
parser.add_argument("--only_include_receptacle", action="store_true", default=False, help="only supervise with receptacle classes")
parser.add_argument("--do_load_oop_nodes_and_supervision", action="store_true", default=False, help="load oop nodes and supervision from memory")
parser.add_argument("--do_save_oop_nodes_and_supervision", action="store_true", default=False, help="save oop nodes and supervision to memory. NOTE: needed for training.")
parser.add_argument("--vmemex_supervision_dir", type=str, default="./data/vmemex_supervision_dir/", help="directory for saving oop nodes and supervision to memory")
parser.add_argument("--objects_per_scene", type=int, default=5, help="objects per scene in the batch")
parser.add_argument("--scenes_per_batch", type=int, default=5, help="scenes per training batch")
parser.add_argument("--max_views_scene_graph", type=int, default=50, help="upper bound on mapping views to use for getting scene graph")
parser.add_argument("--confidence_threshold_scene_graph", type=float, default=0.3, help="confidence threshold for detecting ovjects in scene graph")

###########%%%%%%% Visual Search Network %%%%%%%###########
# parser.add_argument("--predict_obj_locations", action="store_true", default=False, help="predict location of object")
parser.add_argument("--do_add_semantic", action="store_true", default=True, help="add semantic map to input")
parser.add_argument("--include_rgb", action="store_true", default=False, help="include RGB featurization to input")
parser.add_argument("--do_masked_pos_loss", action="store_true", default=False, help="do loss with balanced CE loss")
parser.add_argument("--num_positive_on_average", type=int, default=5, help="number of positives on average for CE loss - to overweight positives")
parser.add_argument("--eval_object_nav", action="store_true", default=False, help="evaluate network on object navigation to all object classes")
parser.add_argument("--num_views_mapping_sample", type=int, default=20, help="number of mapping views to sample for making input map")
parser.add_argument("--keep_target_aithor_ref", action="store_true", default=False, help="keep object target in aithor reference (and do not convert to camX0 reference frame for supervision)")
parser.add_argument("--lr_vsn", type=float, default=5e-6, help="visual search network learning rate")
parser.add_argument("--X", type=int, default=128, help="Width of 3D voxel grid")
parser.add_argument("--Y", type=int, default=64, help="Height of 3D voxel grid")
parser.add_argument("--Z", type=int, default=128, help="Length of 3D voxel grid")
parser.add_argument("--voxel_min", type=float, default=-4, help="voxel grid minimum in meter")
parser.add_argument("--voxel_max", type=float, default=4, help="voxel grid maximum in meter")
parser.add_argument("--vsn_threshold", type=float, default=0.8, help="threshold for sigmoid output during inference to get sparse search points")
parser.add_argument("--erosion_iters", type=int, default=3, help="binary erosion iterations for VSN output during inference")
#####%%% Object nav evaluation %%%%####
parser.add_argument("--max_steps_object_goal_nav", type=int, default=200, help="object goal navigation evaluation maximum steps allowed for each object")
parser.add_argument("--object_navigation_policy_name", type=str, default="vsn_search", help="policy to use for searching (options: vsn_search, random)")
# parser.add_argument("--num_search_locs_object", type=int, default=3, help="number of search locations when searching for an object by visual search network")
parser.add_argument("--objects_per_scene_val", type=int, default=5, help="objects per scene in the batch validation")
parser.add_argument("--detector_threshold_object_nav", type=float, default=0.5, help="keep detector threshold constant at this threshold throughout episodes")

###########%%%%%%% saving mapping observations for memex and visual search network %%%%%%%###########
parser.add_argument("--n_train_mapping_obs", type=int, default=10, help="maximum number of mapping obs to save per room for TRAINING - used in visual memex and visual search network training")
parser.add_argument("--n_val_mapping_obs", type=int, default=5, help="maximum number of mapping obs to save per room for VAL - used in visual memex and visual search network validation")
parser.add_argument("--n_test_mapping_obs", type=int, default=5, help="maximum number of mapping obs to save per room for TEST - used in visual memex and visual search network testing")
parser.add_argument("--mapping_obs_dir", type=str, default="./data/mapping_obs/", help="create mess up scene from loaded file")

###########%%%%%%% hyperparameters for DDETR/SOLQ training %%%%%%%###########
# parser.add_argument("--lr_drop", type=int, default=5, help="how often to drop learning rate for LR scheduler")
parser.add_argument("--max_iters", type=int, default=50000, help="maximum iterations to train")
parser.add_argument("--load_base_solq", action="store_true", default=False, help="Load pretrained solq weights trained on coco")
parser.add_argument("--load_model", action="store_true", default=False, help="Load an existing checkpoint")
parser.add_argument("--load_model_path", type=str, default="", help="Path to existing checkpoint")
parser.add_argument("--load_strict_false", action="store_true", default=False, help="do not load strict checkpoint")
parser.add_argument("--lr_scheduler_from_scratch", action="store_true", default=False, help="do not load LR scheduler from checkpoint if True")
parser.add_argument("--optimizer_from_scratch", action="store_true", default=False, help="do not load optimizer from checkpoint if True")
parser.add_argument("--start_one", action="store_true", default=False, help="start from iteration 0")
parser.add_argument("--load_val_agent", action="store_true", default=False, help="load saved data for validation (rather than generating on the fly from simulator)")
parser.add_argument("--val_load_dir", type=str, default="./data/TIDEE_val/", help="where to save and load val data")
parser.add_argument("--load_train_agent", action="store_true", default=False, help="load saved data for validation (rather than generating on the fly from simulator)")
parser.add_argument("--train_load_dir", type=str, default="./data/TIDEE_train/", help="where to save and load train data")
# parser.add_argument("--n_train", type=int, default=40, help="maximum number of trajectories to save per room for TRAINING")
# parser.add_argument("--n_val", type=int, default=7, help="maximum number of trajectories to save per room for VALIDATION")
parser.add_argument("--data_load_dir", type=str, default="./data/TIDEE_solq_load_data", help="where to save generated data")
parser.add_argument("--save_output", action="store_true", default=False, help="just do data generation")
# SOLQ hyperparams
# parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
parser.add_argument('--lr_backbone', default=2e-5, type=float)
parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)

parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr_drop', default=40, type=int)
parser.add_argument('--save_period', default=10, type=int)
parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')
parser.add_argument('--meta_arch', default='solq', type=str)
parser.add_argument('--sgd', action='store_true')
# Variants of Deformable DETR
parser.add_argument('--with_box_refine', default=True, action='store_true')
parser.add_argument('--two_stage', default=True, action='store_true')
# VecInst
parser.add_argument('--with_vector', default=True, action='store_true')
parser.add_argument('--n_keep', default=256, type=int,
                    help="Number of coeffs to be remained")
parser.add_argument('--gt_mask_len', default=128, type=int,
                    help="Size of target mask")
parser.add_argument('--vector_loss_coef', default=3, type=float)
parser.add_argument('--vector_hidden_dim', default=1024, type=int,
                    help="Size of the vector embeddings (dimension of the transformer)")
parser.add_argument('--no_vector_loss_norm', default=False, action='store_true')
parser.add_argument('--activation', default='relu', type=str, help="Activation function to use")
parser.add_argument('--checkpoint', default=False, action='store_true')
parser.add_argument('--vector_start_stage', default=0, type=int)
parser.add_argument('--num_machines', default=1, type=int)
parser.add_argument('--loss_type', default='l1', type=str)
parser.add_argument('--dcn', default=False, action='store_true')
# Model parameters
parser.add_argument('--frozen_weights', type=str, default=None,
                    help="Path to the pretrained model. If set, only the mask head will be trained")
parser.add_argument('--pretrained', default=None, help='resume from checkpoint')
# * Backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'rel'),
                    help="Type of positional embedding to use on top of the image features")
# parser.add_argument('--position_embedding', default='coord', type=str, choices=('sine', 'learned', 'coord'),
#                     help="Type of positional embedding to use on top of the image features")
parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                    help="position / size * scale")
parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')
# * Transformer
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=1024, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=384, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=300, type=int,
                    help="Number of query slots")
parser.add_argument('--dec_n_points', default=4, type=int)
parser.add_argument('--enc_n_points', default=4, type=int)
# * Segmentation
parser.add_argument('--masks', action='store_true', default=True,
                    help="Train segmentation head if the flag is provided")
# Loss
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help="Disables auxiliary decoding losses (loss at each layer)")
# * Matcher
parser.add_argument('--set_cost_class', default=2, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")
# * Loss coefficients
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--cls_loss_coef', default=2, type=float)
parser.add_argument('--bbox_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--focal_alpha', default=0.25, type=float)
# dataset parameters
parser.add_argument('--dataset_file', default='coco')
parser.add_argument('--coco_path', default='./data/coco', type=str)
parser.add_argument('--coco_panoptic_path', type=str)
parser.add_argument('--remove_difficult', action='store_true')
parser.add_argument('--alg', default='instformer', type=str)
parser.add_argument('--output_dir', default='',
                    help='path where to save, empty for no saving')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
# parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
# distributed
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--dist-url', default=None, type=str, help='url used to set up distributed training')
parser.add_argument('--rank', default=None, type=int, help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--num-machines', default=None, type=int)

args = parser.parse_args()

args.metrics_dir = os.path.join(args.metrics_dir, args.set_name)
args.movie_dir = os.path.join(args.movie_dir, args.set_name)
args.image_dir = os.path.join(args.image_dir, args.set_name)

if args.batch_size is None:
   args.batch_size = args.S*args.data_batch_size

if args.noisy_pose:
   # add pose noise similare to LoCoBot
   # Use args to alter rearrange/environment movement amounts
   args.movementGaussianSigma = 0.005
   args.rotateGaussianSigma = 0.5
else:
   args.movementGaussianSigma = None
   args.rotateGaussianSigma = None


if args.estimate_depth: # or self.noisy_depth:
      args.keep_head_down = True
      args.keep_head_straight = False
      args.search_pitch_explore = False
      args.min_depth = 0.0
      args.max_depth = 20.0
elif args.noisy_depth:
      args.keep_head_down = False
      args.keep_head_straight = False
      args.search_pitch_explore = True
      args.min_depth = 0.00
      args.max_depth = 20.0
else:
      args.keep_head_down = False
      args.keep_head_straight = False
      args.search_pitch_explore = True
      args.min_depth = None
      args.max_depth = None

if args.mode=="visual_search_network" and args.eval_object_nav:
   args.confidence_threshold = args.detector_threshold_object_nav
   args.confidence_threshold_searching = args.detector_threshold_object_nav
   args.visibilityDistance = 1.0