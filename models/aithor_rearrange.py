import os
import sys
from models.aithor_rearrange_base import Ai2Thor_Base
import ipdb
st = ipdb.set_trace
import torch
import random
import numpy as np
from arguments import args
import traceback
import pickle
import sys
from tidee.navigation import Navigation, Depth
from task_base.animation_util import Animation
from tidee.object_tracker import ObjectTrack
from backend import saverloader
from PIL import Image
import utils.geom
import utils.aithor
import matplotlib.pyplot as plt
from PIL import Image
from nets.alfred_perception_models import AlfredSegmentationAndDepthModel
from collections import Counter, OrderedDict
import cv2
from nets.segmentation.segmentation_helper import SemgnetationHelper
import math
"""Inference loop for the AI2-THOR object rearrangement task."""
from allenact.utils.misc_utils import NumpyJSONEncoder
sys.path.append('rearrangement')
from rearrangement.baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from rearrange.constants import (
    OBJECT_TYPES_WITH_PROPERTIES,
    THOR_COMMIT_ID,
)

# from rearrangement.baseline_configs.one_phase.one_phase_rgb_base import (
#     OnePhaseRGBBaseExperimentConfig,
# # )
# from rearrangement.baseline_configs.two_phase.two_phase_rgb_base import (
#     TwoPhaseRGBBaseExperimentConfig,
# )
from rearrangement.baseline_configs.two_phase.two_phase_tidee_base import (
    TwoPhaseTIDEEExperimentConfig,
)
from rearrangement.rearrange.sensors import (
    RGBRearrangeSensor,
    InWalkthroughPhaseSensor,
    DepthRearrangeSensor,
    ClosestUnshuffledRGBRearrangeSensor,
)
from allenact.base_abstractions.sensor import SensorSuite, Sensor

try:
    from allenact.embodiedai.sensors.vision_sensors import DepthSensor
except ImportError:
    raise ImportError("Please update to allenact>=0.4.0.")

from rearrangement.rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask
from utils.wctb import Utils, Relations_CenterOnly
import re
import time
import json
import gzip
import os
from utils.noise_models.sim_kinect_noise import add_gaussian_shifts, filterDisp

# print("NOTE: SUPPRESSING WARNINGS!!!!!")
# import warnings
# warnings.filterwarnings("ignore")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def format_a(a):
    a_split = a.split('_')[1:]
    for phrase_i in range(len(a_split)):
        a_split[phrase_i] = a_split[phrase_i].capitalize()
    a_formatted = ''.join(a_split)
    return a_formatted

def undo_format_a(a):
    formatted = re.sub(r"(?<=\w)([A-Z])", r" \1", a).lower()
    formatted = formatted.replace(' ', '_')
    return formatted

# Note: parameters changed in rearrange/rearrange_base.py
# Note: altered self.physics_step_kwargs in rearrange/environment.py to include correct DT and HORIZON_DT


# from abc import ABC

# class TwoPhaseTIDEEExperimentConfig(TwoPhaseRGBBaseExperimentConfig, ABC):

#     SCREEN_SIZE = args.W
#     THOR_CONTROLLER_KWARGS = {
#         "rotateStepDegrees": args.DT,
#         "snapToGrid": False,
#         "quality": "Ultra",
#         "width": SCREEN_SIZE,
#         "height": SCREEN_SIZE,
#         "commit_id": THOR_COMMIT_ID,
#         "fastActionEmit": True,
#     }

#     EGOCENTRIC_RGB_UUID = "rgb"
#     UNSHUFFLED_RGB_UUID = "unshuffled_rgb"
#     EGOCENTRIC_RGB_RESNET_UUID = "rgb_resnet"
#     UNSHUFFLED_RGB_RESNET_UUID = "unshuffled_rgb_resnet"
#     EGOCENTRIC_DEPTH_UUID = "depth"
#     UNSHUFFLED_DEPTH_UUID = "unshuffled_depth"
#     EGOCENTRIC_DEPTH_RESNET_UUID = "depth_resnet"
#     UNSHUFFLED_DEPTH_RESNET_UUID = "unshuffled_depth_resnet"

#     SENSORS = [
#         RGBRearrangeSensor(
#             height=SCREEN_SIZE,
#             width=SCREEN_SIZE,
#             use_resnet_normalization=False,
#             uuid=EGOCENTRIC_RGB_UUID,
#         ),
#         DepthRearrangeSensor(
#             height=SCREEN_SIZE,
#             width=SCREEN_SIZE,
#             uuid=EGOCENTRIC_DEPTH_UUID,
#         ),
#         ClosestUnshuffledRGBRearrangeSensor(
#             height=SCREEN_SIZE,
#             width=SCREEN_SIZE,
#             use_resnet_normalization=True,
#             uuid=UNSHUFFLED_RGB_UUID,
#         ),
#         InWalkthroughPhaseSensor(),
#     ]

    


class Ai2Thor(Ai2Thor_Base):
    def __init__(self):   

        super(Ai2Thor, self).__init__()

        self.actions = [
            'done', 'move_ahead', 'move_left', 'move_right', 'move_back', 'rotate_right', 'rotate_left', 
            'stand', 'crouch', 'look_up', 'look_down', 'drop_held_object_with_snap', 'open_by_type_blinds', 
            'open_by_type_cabinet', 'open_by_type_drawer', 'open_by_type_fridge', 'open_by_type_laundry_hamper', 
            'open_by_type_microwave', 'open_by_type_safe', 'open_by_type_shower_curtain', 'open_by_type_shower_door', 
            'open_by_type_toilet', 'pickup_alarm_clock', 'pickup_aluminum_foil', 'pickup_apple', 'pickup_baseball_bat', 
            'pickup_basket_ball', 'pickup_book', 'pickup_boots', 'pickup_bottle', 'pickup_bowl', 'pickup_box', 'pickup_bread', 
            'pickup_butter_knife', 'pickup_c_d', 'pickup_candle', 'pickup_cell_phone', 'pickup_cloth', 'pickup_credit_card', 
            'pickup_cup', 'pickup_dish_sponge', 'pickup_dumbbell', 'pickup_egg', 'pickup_footstool', 'pickup_fork', 
            'pickup_hand_towel', 'pickup_kettle', 'pickup_key_chain', 'pickup_knife', 'pickup_ladle', 
            'pickup_laptop', 'pickup_lettuce', 'pickup_mug', 'pickup_newspaper', 'pickup_pan', 'pickup_paper_towel_roll', 
            'pickup_pen', 'pickup_pencil', 'pickup_pepper_shaker', 'pickup_pillow', 'pickup_plate', 'pickup_plunger', 'pickup_pot', 
            'pickup_potato', 'pickup_remote_control', 'pickup_salt_shaker', 'pickup_scrub_brush', 'pickup_soap_bar', 'pickup_soap_bottle', 
            'pickup_spatula', 'pickup_spoon', 'pickup_spray_bottle', 'pickup_statue', 'pickup_table_top_decor', 'pickup_teddy_bear', 
            'pickup_tennis_racket', 'pickup_tissue_box', 'pickup_toilet_paper', 'pickup_tomato', 'pickup_towel', 'pickup_vase', 
            'pickup_watch', 'pickup_watering_can', 'pickup_wine_bottle'
            ]

        self.nav_action_to_rearrange_action = {
            'MoveAhead':'move_ahead', 'MoveLeft':'move_left', 'MoveRight':'move_right', 'MoveBack':'move_back',
            'RotateRight':'rotate_right', 'RotateLeft':'rotate_left', 'LookUp':'look_up', 'LookDown':'look_down', 'Pass':'done',
            }

        self.rearrange_pickupable = []
        self.action_to_ind = {}
        self.ind_to_action = {}
        for a_i in range(len(self.actions)):
            a = self.actions[a_i]
            self.action_to_ind[a] = a_i
            self.ind_to_action[a_i] = a
            if 'pickup' in a:
                a_formatted = format_a(a)
                self.rearrange_pickupable.append(a_formatted)

        self.current_mode = args.eval_split # "test"
        self.tag = args.tag # "arxiv"
        self.verbose = False
        self.search_V2 = True
        self.search_receptacles = False

        self.log_every = args.log_every
        self.create_movie = args.create_movie

        ##### ESTIMATED DEPTH AND POSE #########
        # args.noisy_pose = False
        # args.estimate_depth = False
        # args.noisy_depth = False
        self.estimate_depth = args.estimate_depth
        self.noisy_depth = args.noisy_depth
        self.noisy_pose = args.noisy_pose
        assert(not (self.noisy_depth and self.estimate_depth) )
        # if self.estimate_depth: 
        #     assert(args.HORIZON_DT==45) # depth estimator works best at 45 degrees
            # args.HORIZON_DT = 45
            # args.DT = 45
        ########################################

        self.load_submission = args.load_submission
        self.start_at_500 = False
        if self.start_at_500: # run from 500- for multi-gpu 
            add_to_filename = '_500on'
        else:
            add_to_filename = ''
        
        self.submission_file = f"./metrics/submission_{self.current_mode}_original{add_to_filename}_estimateDepth={self.estimate_depth}_noisyDepth={self.noisy_depth}_noisyPose={self.noisy_pose}_{self.tag}.json.gz"
        
        # self.task_sampler_params = TwoPhaseRGBBaseExperimentConfig.stagewise_task_sampler_args(
        #     stage=self.current_mode, process_ind=0, total_processes=1,
        # )
        # self.two_phase_rgb_task_sampler: RearrangeTaskSampler = TwoPhaseRGBBaseExperimentConfig.make_sampler_fn(
        #     **self.task_sampler_params,
        #     force_cache_reset=True,  # cache used for efficiency during training, should be True during inference
        #     only_one_unshuffle_per_walkthrough=False,  # used for efficiency during training, should be False during inference
        #     epochs=1,
        #     # shuffle=args.shuffle_maps,
        # )
        self.task_sampler_params = TwoPhaseTIDEEExperimentConfig.stagewise_task_sampler_args(
            stage=self.current_mode, process_ind=0, total_processes=1,
        )
        self.two_phase_rgb_task_sampler: RearrangeTaskSampler = TwoPhaseTIDEEExperimentConfig.make_sampler_fn(
            **self.task_sampler_params,
            force_cache_reset=True,  # cache used for efficiency during training, should be True during inference
            only_one_unshuffle_per_walkthrough=False,  # used for efficiency during training, should be False during inference
            epochs=1,
        )

        self.how_many_unique_datapoints = self.two_phase_rgb_task_sampler.total_unique
        self.num_tasks_to_do = self.how_many_unique_datapoints #5

        print(
            f"Sampling {self.num_tasks_to_do} tasks from the Two-Phase {self.current_mode} dataset"
            f" ({self.how_many_unique_datapoints} unique tasks)"
        )

        self.controller_walkthrough = self.two_phase_rgb_task_sampler.walkthrough_env.controller
        self.controller_unshuffle = self.two_phase_rgb_task_sampler.unshuffle_env.controller

        # initialize detector
        args.data_mode = "solq"
        from nets.solq import DDETR
        load_pretrained = False
        self.ddetr = DDETR(len(self.include_classes), load_pretrained).cuda()

        SOLQ_checkpoint = args.SOLQ_checkpoint #"/projects/katefgroup/viewpredseg/checkpoints/TEACH_solq_aithor05/model-00023000.pth"
        print("...found checkpoint %s"%(SOLQ_checkpoint))
        checkpoint = torch.load(SOLQ_checkpoint)
        pretrained_dict = checkpoint['model_state_dict']
        self.ddetr.load_state_dict(pretrained_dict, strict=True)
        self.ddetr.eval().cuda()

        self.utils = Utils(args.H, args.W)
        self.relations_util = Relations_CenterOnly(args.H, args.W)

        # relations used for determining differences
        self.relations_executors_pairs = {
            # 'above': self.relations_util._is_above,
            # 'below': self.relations_util._is_below,
            'next-to': self.relations_util._is_next_to,
            'supported-by': self.relations_util._is_supported_by,
            # 'similar-height-to': self.relations_util._is_similar_height,
            # 'farthest-to': self.relations_util._farthest,
            'closest-to': self.relations_util._closest,
        }

        self.rel_to_id = {list(self.relations_executors_pairs.keys())[i]:i for i in range(len(self.relations_executors_pairs))}

        self.max_relations = args.max_relations

        # self.OT_dist_thresh = 0.05 #args.OT_dist_thresh #0.05
        self.dist_thresh = args.dist_thresh #0.5

        self.iter_interval = [] #args.rearrange_interval
        if args.rearrange_interval_min is not None or args.rearrange_interval_max is not None:
            self.iter_interval = [0, self.num_tasks_to_do]
            if args.rearrange_interval_min is not None: # option to start at 0
                self.iter_interval[0] = args.rearrange_interval_min
            if args.rearrange_interval_max is not None: # option to end at last iter
                self.iter_interval[1] = args.rearrange_interval_max

        self.oop_id_to_label = {0:'NOT oop', 1:'oop'}

        self.receptacles = ['Cabinet', 'CounterTop', 'Sink', 'TowelHolder',
            'GarbageCan', 
            'SinkBasin', 'Bed', 
            'Drawer', 'SideTable', 'Chair', 'Desk', 'Dresser',  
            'Ottoman', 'ArmChair', 'Sofa', 'DogBed', 'ShelvingUnit', 
            'Shelf', 'StoveBurner', 'Microwave', 'CoffeeMachine', 'Fridge', 
            'Toaster', 'DiningTable',  
            'LaundryHamper', 'Stool', 'CoffeeTable', 'Bathtub', 'Footstool', 'BathtubBasin', 
            'TVStand', 'Safe']

        if self.estimate_depth:
            self.depth_estimator = Depth()
        else:
            self.depth_estimator = None

        if self.noisy_depth:
            # print("REMOVED GUASSIAN SHIFT FROM NOISY DEPTH - ADD BACK.")
            # from utils.noise_models.redwood_depth_noise_model import RedwoodDepthNoiseModel
            # self.depth_noise_model = RedwoodDepthNoiseModel()
            self.dot_pattern_ = cv2.imread("utils/noise_models/data/kinect-pattern_3x3.png", 0)

        if self.noisy_pose:
            # add pose noise similare to LoCoBot
            # Use hyp to alter rearrange/environment movement amounts
            args.movementGaussianSigma = 0.005
            args.rotateGaussianSigma = 0.5
        else:
            args.movementGaussianSigma = None
            args.rotateGaussianSigma = None

        if not os.path.exists(args.movie_dir):
            os.mkdir(args.movie_dir)

        # self.main()

    def main(self):
        my_leaderboard_submission = {}

        if self.load_submission:
            if os.path.exists(self.submission_file):
                with gzip.open(self.submission_file, 'rt', encoding='UTF-8') as zipfile:
                    my_leaderboard_submission = json.load(zipfile)
        
        for i_task in range(self.num_tasks_to_do):
            
            print(args.DT, args.HORIZON_DT, args.STEP_SIZE)
            print(f"\nStarting task {i_task}")

            print("PHASE:", self.current_mode, "Estimated depth?", self.estimate_depth, "Noisy depth?", self.noisy_depth, "Noisy pose?", self.noisy_pose)            

            print("SUBMISSION NAME:", self.submission_file)

            walkthrough_task = self.two_phase_rgb_task_sampler.next_task()

            # if i_task==0:
            #         walkthrough_task.step(action=0)
            #         unshuffle_task: UnshuffleTask = self.two_phase_rgb_task_sampler.next_task()
            #         unshuffle_task.step(action=0)
            #         print("skipping..")
            #         continue

            # try:
            #     id_ = int(self.two_phase_rgb_task_sampler.current_task_spec.unique_id[-2:])
            # except:
            #     id_ = int(self.two_phase_rgb_task_sampler.current_task_spec.unique_id[-1:])
            # if id_>10:
            #     walkthrough_task.step(action=0)
            #     unshuffle_task: UnshuffleTask = self.two_phase_rgb_task_sampler.next_task()
            #     unshuffle_task.step(action=0)
            #     print(f"skipping {id_} ..")
            #     continue
                

            if len(self.iter_interval)>0:
                if i_task<self.iter_interval[0] or i_task>self.iter_interval[1]:
                    walkthrough_task.step(action=0)
                    unshuffle_task: UnshuffleTask = self.two_phase_rgb_task_sampler.next_task()
                    unshuffle_task.step(action=0)
                    print("skipping..")
                    continue

            unique_id = self.two_phase_rgb_task_sampler.walkthrough_env.current_task_spec.unique_id
            if unique_id in my_leaderboard_submission:
                walkthrough_task.step(action=0)
                unshuffle_task: UnshuffleTask = self.two_phase_rgb_task_sampler.next_task()
                unshuffle_task.step(action=0)
                print("in submission already.. skipping..")
                continue

            print(
                f"Sampled task is from the "
                f" '{self.two_phase_rgb_task_sampler.current_task_spec.stage}' stage and has"
                f" unique id '{self.two_phase_rgb_task_sampler.current_task_spec.unique_id}'"
            )

            # assert isinstance(walkthrough_task, WalkthroughTask)

            if self.estimate_depth: # or self.noisy_depth:
                keep_head_down = True
                keep_head_straight = False
                search_pitch_explore = False
                min_depth = 0.0
                max_depth = 20.0
            elif self.noisy_depth:
                keep_head_down = False
                keep_head_straight = False
                search_pitch_explore = True
                min_depth = 0.00
                max_depth = 20.0
            else:
                keep_head_down = False
                keep_head_straight = False
                search_pitch_explore = True
                min_depth = None
                max_depth = None
            navigation = Navigation(keep_head_down=keep_head_down, keep_head_straight=keep_head_straight, search_pitch_explore=search_pitch_explore)

            rgb, depth = self.get_obs(walkthrough_task, head_tilt=0)
            self.success_checker = CheckSuccessfulAction(rgb_init=rgb, H=self.H, W=self.W, controller=self.controller_walkthrough)
            self.update_navigation_obs(rgb,depth, True, navigation)

            if self.create_movie and i_task%self.log_every==0:
                print("LOGGING THIS ITERATION")
                vis = Animation(self.W, self.H, navigation=navigation, name_to_id=self.name_to_id)
                print('Height:', self.H, 'Width:', self.W)
            else:
                vis = None
            
            # for walkthrough task, explore env and track objects 
            object_tracker_walkthrough, inds_explore_walkthrough, _ = self.explore_env(
                walkthrough_task, navigation, self.controller_walkthrough, 
                vis=vis, phase="WALKTHROUGH",
                steps_max=RearrangeBaseExperimentConfig.MAX_STEPS["walkthrough"]*3,
                )

            object_tracker_walkthrough.filter_centroids_out_of_bounds()

            walkthrough_obj_rels = self.get_relations_pickupable(object_tracker_walkthrough)

            print("WALKTHROUGH")

            if not walkthrough_task.is_done():
                walkthrough_task.step(action=0)

            if vis is not None:
                for _ in range(5):
                    vis.add_frame(self.controller_unshuffle.last_event.frame, text="DONE.")
                if args.generate_rearrangement_images:
                    print("Warning: suggested to turn off rearrangement images for running full task")
                    vis_rearrange = Visualize_Rearrange()
                    vis_rearrange.get_walkthrough_images(walkthrough_task)
            
            unshuffle_task: UnshuffleTask = self.two_phase_rgb_task_sampler.next_task()

            if vis is not None:
                if args.generate_rearrangement_images:
                    vis_rearrange.get_unshuffle_images(unshuffle_task)

            navigation = Navigation(keep_head_down=keep_head_down, keep_head_straight=keep_head_straight, search_pitch_explore=search_pitch_explore)
            
            if vis is not None:
                vis.navigation = navigation
            
            rgb, depth = self.get_obs(unshuffle_task, head_tilt=0)
            self.success_checker = CheckSuccessfulAction(rgb_init=rgb, H=self.H, W=self.W, controller=self.controller_unshuffle)
            self.update_navigation_obs(rgb,depth, True, navigation)

            print("STARTING:", self.controller_unshuffle.last_event.metadata['cameraPosition'], self.controller_unshuffle.last_event.metadata['agent']['cameraHorizon'], self.controller_unshuffle.last_event.metadata['agent']['rotation'])

            object_tracker_unshuffle, _, camX0_T_origin = self.explore_env(
                unshuffle_task, navigation, self.controller_unshuffle, 
                vis=vis, steps_max=RearrangeBaseExperimentConfig.MAX_STEPS["walkthrough"], 
                phase="UNSHUFFLE", inds_explore=inds_explore_walkthrough,
                )

            object_tracker_unshuffle.filter_centroids_out_of_bounds()

            tracker_centroids, tracker_labels = object_tracker_unshuffle.get_centroids_and_labels()

            print("UNSHUFFLE")

            if args.get_pose_change_from_GT:
                out_of_place = self.get_GT_pose_changes(unshuffle_task, camX0_T_origin)
            else: # all estimated
                unshuffle_obj_rels = self.get_relations_pickupable(object_tracker_unshuffle)
                out_of_place, object_dict = self.get_out_of_place(walkthrough_obj_rels, unshuffle_obj_rels)

            for key in list(out_of_place.keys()):
                if unshuffle_task.is_done():
                    break
                print("ATTEMPTING TO REARRANGE", out_of_place[key]['label'])
                obj_dict = out_of_place[key]
                self.move_object_state(
                    obj_dict['label'], 
                    obj_dict['unshuffle_state'], 
                    obj_dict['walkthrough_state'], 
                    obj_dict['action'],
                    navigation, 
                    self.controller_unshuffle, 
                    unshuffle_task, 
                    vis=vis
                    )

            if not unshuffle_task.is_done():
                unshuffle_task.step(action=0)

            if vis is not None:
                for _ in range(5):
                    vis.add_frame(self.controller_unshuffle.last_event.frame, text="DONE.")
                if args.generate_rearrangement_images:
                    vis_rearrange.get_rearranged_images(unshuffle_task)

            metrics = unshuffle_task.metrics()
            
            task_info = metrics["task_info"]
            del metrics["task_info"]
            my_leaderboard_submission[task_info["unique_id"]] = {**task_info, **metrics}
            print(f"Both phases complete, metrics: '{metrics}'")
            # print(my_leaderboard_submission)
            
            if vis is not None:
                env = unshuffle_task.unshuffle_env
                ips, gps, cps = env.poses

                start_energies = unshuffle_task.start_energies
                end_energies = env.pose_difference_energy(gps, cps)
                start_energy = start_energies.sum()
                end_energy = end_energies.sum()

                start_misplaceds = start_energies > 0.0
                end_misplaceds = end_energies > 0.0

                num_broken = sum(cp["broken"] for cp in cps)
                num_initially_misplaced = start_misplaceds.sum()
                num_fixed = num_initially_misplaced - (start_misplaceds & end_misplaceds).sum()
                num_newly_misplaced = (end_misplaceds & np.logical_not(start_misplaceds)).sum()
                # if num_newly_misplaced>0:
                #     st()

                newly_misplaced = end_misplaceds & np.logical_not(start_misplaceds)
                where_misplaced = np.where(end_misplaceds)[0]
                types = [gps[i]['type']+f'_nm={newly_misplaced[i]}' for i in list(where_misplaced)]

                log_dir = os.path.join(args.movie_dir, str(self.two_phase_rgb_task_sampler.current_task_spec.unique_id))
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
                vis.render_movie(log_dir,i_task, tag=f'{self.two_phase_rgb_task_sampler.current_task_spec.unique_id}_rearrange_suc={metrics["unshuffle/success"]}_pfs={metrics["unshuffle/prop_fixed_strict"]}_typesmissed={types}')
                
                if args.generate_rearrangement_images:
                    vis_rearrange.save_final_images(log_dir)

                if args.log_relations_txt:
                    self.generate_relations_text_log(object_dict, log_dir, gps, newly_misplaced, start_misplaceds)


            if i_task % self.log_every == 0:
                save_path = self.submission_file # f"/home/gsarch/repo/project_cleanup/ai2thor-rearrangement/metrics/submission_{self.current_mode}_original_searchR={self.search_receptacles}_searchSame={self.search_same_spots}_searchV2={self.search_V2}_estimateDepth={self.estimate_depth}_noisyDepth={self.noisy_depth}_noisyPose={self.noisy_pose}.json.gz"
                if os.path.exists(os.path.dirname(save_path)):
                    print(f"Saving submission file to {save_path}")
                    submission_json_str = json.dumps(my_leaderboard_submission, cls=NumpyJSONEncoder)
                    with gzip.open(save_path, "w") as f:
                        f.write(submission_json_str.encode("utf-8"))

            # except Exception as e:
            #     print(e)
            #     print("Task failed.. skipping to next one")
            #     walkthrough_task.step(action=0)
            #     unshuffle_task: UnshuffleTask = self.two_phase_rgb_task_sampler.next_task()
            #     unshuffle_task.step(action=0)
            #     metrics = unshuffle_task.metrics()
            #     task_info = metrics["task_info"]
            #     del metrics["task_info"]
            #     my_leaderboard_submission[task_info["unique_id"]] = {**task_info, **metrics}
            #     print(f"Both phases complete, metrics: '{metrics}'")
            #     continue

            

        save_path = self.submission_file #f"/home/gsarch/repo/project_cleanup/ai2thor-rearrangement/metrics/submission_{self.current_mode}_original_searchR={self.search_receptacles}_searchSame={self.search_same_spots}_searchV2={self.search_V2}_estimateDepth={self.estimate_depth}_noisyDepth={self.noisy_depth}_noisyPose={self.noisy_pose}.json.gz"
        if os.path.exists(os.path.dirname(save_path)):
            print(f"Saving submission file to {save_path}")
            submission_json_str = json.dumps(my_leaderboard_submission, cls=NumpyJSONEncoder)
            with gzip.open(save_path, "w") as f:
                f.write(submission_json_str.encode("utf-8"))

        self.two_phase_rgb_task_sampler.close()

        print(f"\nFinished {num_tasks_to_do} One-Phase tasks.")

    def get_GT_pose_changes(self, unshuffle_task, camX0_T_origin):

        unshuffle_start_poses, walkthrough_start_poses, current_poses =  unshuffle_task.env.poses

        # ips, gps, cps = unshuffle_task.env.poses
        start_energies = unshuffle_task.start_energies
        # end_energies = unshuffle_task.env.pose_difference_energy(gps, cps)
        # start_energy = start_energies.sum()
        # end_energy = end_energies.sum()

        start_misplaceds = start_energies > 0.0
        # end_misplaceds = end_energies > 0.0

        where_misplaces = np.where(start_misplaceds)[0]

        pos_diff = []
        open_diff = []
        out_of_place = {}
        id_ = 0
        eps_pos = 1e-3
        eps_rot = 0.05
        eps_open = 0.05
        for o_i in list(where_misplaces):
            # if not start_misplaceds[o_i]:
            #     continue

            unshuffle_obj = unshuffle_start_poses[o_i]
            walkthrough_obj = walkthrough_start_poses[o_i]
            pose_diff = unshuffle_task.env.compare_poses(unshuffle_obj, walkthrough_obj)

            # different_check = pose_diff['iou']>0.0 or pose_diff['openness_diff']>0.0:
            # if pose_diff['iou'] is not None and (pose_diff['position_dist']>eps_pos or pose_diff['rotation_dist']>eps_rot and pose_diff['iou']<1.0) and unshuffle_obj['type'] in self.PICKUPABLE_OBJECTS:
            if unshuffle_obj['type'] in self.PICKUPABLE_OBJECTS:
                obj_center_walkthrough = torch.from_numpy(np.array(list(walkthrough_obj['position'].values()))).unsqueeze(0).cuda()
                obj_center_walkthrough_camX0 = utils.geom.apply_4x4(camX0_T_origin.unsqueeze(0).cuda().float(), obj_center_walkthrough.unsqueeze(1).cuda().float()).squeeze(1)
                obj_center_walkthrough_camX0[:,1] = -obj_center_walkthrough_camX0[:,1]
                obj_center_walkthrough_camX0 = obj_center_walkthrough_camX0.squeeze().cpu().numpy()
                obj_center_unshuffle = torch.from_numpy(np.array(list(unshuffle_obj['position'].values()))).unsqueeze(0)
                obj_center_unshuffle_camX0 = utils.geom.apply_4x4(camX0_T_origin.unsqueeze(0).cuda().float(), obj_center_unshuffle.unsqueeze(1).cuda().float()).squeeze(1)
                obj_center_unshuffle_camX0[:,1] = -obj_center_unshuffle_camX0[:,1]
                obj_center_unshuffle_camX0 = obj_center_unshuffle_camX0.squeeze().cpu().numpy()
                out_of_place[id_] = {}
                out_of_place[id_]['label'] = unshuffle_obj['type']
                out_of_place[id_]['walkthrough_state'] = obj_center_walkthrough_camX0
                out_of_place[id_]['unshuffle_state'] = obj_center_unshuffle_camX0
                out_of_place[id_]['action'] = 'pickup'
                id_ += 1
                pos_diff.append(pose_diff)
            # elif pose_diff['openness_diff'] is not None and pose_diff['openness_diff']>eps_open and args.do_open and unshuffle_obj['type'] in self.OPENABLE_OBJECTS:
            elif unshuffle_obj['type'] in self.OPENABLE_OBJECTS and args.do_open:
                obj_center_unshuffle = torch.from_numpy(np.array(list(unshuffle_obj['position'].values()))).unsqueeze(0)
                obj_center_unshuffle_camX0 = utils.geom.apply_4x4(camX0_T_origin.unsqueeze(0).cuda().float(), obj_center_unshuffle.unsqueeze(1).cuda().float()).squeeze(1)
                obj_center_unshuffle_camX0[:,1] = -obj_center_unshuffle_camX0[:,1]
                obj_center_unshuffle_camX0 = obj_center_unshuffle_camX0.squeeze().cpu().numpy()
                out_of_place[id_] = {}
                out_of_place[id_]['label'] = unshuffle_obj['type']
                out_of_place[id_]['walkthrough_state'] = None
                out_of_place[id_]['unshuffle_state'] = obj_center_unshuffle_camX0
                out_of_place[id_]['action'] = 'open'
                id_ += 1
                open_diff.append(pose_diff)
        
        return out_of_place

    def generate_relations_text_log(self, object_dict, save_dir, gps, newly_misplaced, start_misplaced):

        text = ''

        text += f'############################################################################\n'
        text += f'################################# PREDICTED ################################\n'
        text += f'############################################################################\n'
        for k in object_dict.keys():
            obj = object_dict[k]
            text += f'############%%%%%%%%%%% Object {k}: {obj["label"]} %%%%%%%%%%%##############\n'
            text += f'Predicted label: {obj["label"]}\n'
            text += f'Similar: \n'
            for s in obj["similar"]:
                text += f'     {s}\n'
            text += f'Different: \n'
            for d in obj["different"]:
                text += f'     {d}\n'
            text += f'Predicted rearranged?: {obj["out_of_place"]}\n'
            text += f'\n\n\n'

        text += f'############################################################################\n'
        text += f'################################# GT #######################################\n'
        text += f'############################################################################\n'
        text += f'Start misplaced: \n'
        for i in range(len(start_misplaced)):
            if start_misplaced[i]:
                text += f'     {gps[i]["type"]} \n'
        text += f'\n\n'
        text += f'Newly misplaced: \n'
        for i in range(len(newly_misplaced)):
            if newly_misplaced[i]:
                text += f'     {gps[i]["type"]} \n'
        
        text_file = open(os.path.join(save_dir, "relations_txt.txt"), "w")
        n = text_file.write(text)
        text_file.close()

class Visualize_Rearrange():
    '''
    Util for visualizing the placement locations in the walkthrough, unshuffle, and end of unshuffle phases
    '''

    def __init__(self):
        pass

    def save_final_images(self, save_dir):

        obj_names = list(self.rearrange_images.keys())

        for obj_name in obj_names:
            walkthrough_image = self.walkthrough_images[obj_name]['rgb']
            unshuffle_image = self.rearrange_images[obj_name]['rgb']
            rearrange_image = self.rearrange_images[obj_name]['rgb']
            img = Image.fromarray(np.uint8(np.concatenate([walkthrough_image, unshuffle_image, rearrange_image], axis=1)))
            walkthrough_rec = self.walkthrough_images[obj_name]['receptacle']
            unshuffle_rec = self.rearrange_images[obj_name]['receptacle']
            rearrange_rec = self.rearrange_images[obj_name]['receptacle']
            path = os.path.join(save_dir, f'{obj_name}_walkrec={walkthrough_rec}_unshuffrec={unshuffle_rec}_rearrec={rearrange_rec}.jpeg')
            print(f"saving {path}")
            img.save(path)


    def get_rearranged_images(self, unshuffle_task):

        controller = unshuffle_task.env.controller

        unshuffle_start_poses, walkthrough_start_poses, current_poses =  unshuffle_task.env.poses

        start_energies = unshuffle_task.start_energies
        start_misplaceds = start_energies > 0.0
        where_misplaces = np.where(start_misplaceds)[0]

        ordered_obj_names = []
        for o_i in list(where_misplaces):

            unshuffle_obj = unshuffle_start_poses[o_i]
            ordered_obj_names.append(unshuffle_obj["name"])

        objs = controller.last_event.metadata["objects"]
        objs_dict = {}
        for obj in objs:
            objs_dict[obj["name"]] = obj

        target_objs = []
        for k in ordered_obj_names:
            if k not in objs_dict.keys():
                continue
            target_objs.append(objs_dict[k])

        self.rearrange_images = self.get_images(controller, target_objs, args.H, args.W, args.fov)

    def get_unshuffle_images(self, unshuffle_task):

        controller = unshuffle_task.env.controller

        unshuffle_start_poses, walkthrough_start_poses, current_poses =  unshuffle_task.env.poses

        start_energies = unshuffle_task.start_energies
        start_misplaceds = start_energies > 0.0
        where_misplaces = np.where(start_misplaceds)[0]

        ordered_obj_names = []
        for o_i in list(where_misplaces):

            unshuffle_obj = unshuffle_start_poses[o_i]
            ordered_obj_names.append(unshuffle_obj["name"])

        objs = controller.last_event.metadata["objects"]
        objs_dict = {}
        for obj in objs:
            objs_dict[obj["name"]] = obj

        target_objs = []
        for k in ordered_obj_names:
            if k not in objs_dict.keys():
                continue
            target_objs.append(objs_dict[k])

        self.unshuffle_images = self.get_images(controller, target_objs, args.H, args.W, args.fov)


    def get_walkthrough_images(self, walkthrough_task):
        controller = walkthrough_task.env.controller

        ordered_obj_names = list(walkthrough_task.env.obj_name_to_walkthrough_start_pose.keys())

        objs = controller.last_event.metadata["objects"]
        objs_dict = {}
        for obj in objs:
            objs_dict[obj["name"]] = obj

        target_objs = []
        for k in ordered_obj_names:
            if k not in objs_dict.keys():
                continue
            target_objs.append(objs_dict[k])
        
        self.walkthrough_images = self.get_images(controller, target_objs, args.H, args.W, args.fov)

    def get_images(self, controller, target_objects, H, W, fov):


        position_start = controller.last_event.metadata["agent"]["position"]
        rotation_start = controller.last_event.metadata["agent"]["rotation"]
        head_tilt = controller.last_event.metadata["agent"]["cameraHorizon"]

        event = controller.step(
                        action="GetReachablePositions"
                    ) #.metadata["actionReturn"]
        nav_pts = event.metadata["actionReturn"]
        nav_pts = np.array([list(d.values()) for d in nav_pts])

        # if do_zoom_in_video or do_third_party_image:
        #     event_test = controller.step(
        #         action="UpdateThirdPartyCamera",
        #         thirdPartyCameraId=0,
        #         position=dict(x=-1.25, y=1, z=-1),
        #         rotation=dict(x=90, y=0, z=0),
        #         fieldOfView=90
        #     )
        #     if not event_test.metadata["lastActionSuccess"]:
        #         third_party_event = controller.step(
        #             action="AddThirdPartyCamera",
        #             position=dict(x=-1.25, y=1, z=-1),
        #             rotation=dict(x=90, y=0, z=0),
        #             fieldOfView=90
        #         )

        image_dict = {}
        for obj in target_objects:
            # if obj['name'] not in target_names:
            #     continue

            obj_center = np.array(list(obj['axisAlignedBoundingBox']['center'].values()))

            print(f"Getting image for {obj['name']}")
            # print(obj['axisAlignedBoundingBox']['center'])

            dists = np.sqrt(np.sum((nav_pts - obj_center)**2, axis=1))
            argmin_pos = np.argmin(dists)
            closest_pos= nav_pts[argmin_pos] 

            # YAW calculation - rotate to object
            agent_to_obj = np.squeeze(obj_center) - (closest_pos + np.array([0.0, 0.675, 0.0]))
            agent_local_forward = np.array([0, 0, 1.0]) 
            flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
            flat_dist_to_obj = np.linalg.norm(flat_to_obj)
            flat_to_obj /= flat_dist_to_obj

            det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
            turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

            # # add noise so not right in the center
            # noise = np.random.normal(0, 2, size=2)

            turn_yaw = np.degrees(turn_angle) #+ noise[0]

            turn_pitch = -np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj)) #+ noise[1]

            event = controller.step('TeleportFull', position=dict(x=closest_pos[0], y=closest_pos[1], z=closest_pos[2]), rotation=dict(x=0.0, y=turn_yaw, z=0.0), horizon=turn_pitch, standing=True, forceAction=True)
            origin_T_camX = utils.aithor.get_origin_T_camX(controller.last_event, False)
            # move agent far away

            argmax_pos = np.argmax(np.sqrt(np.sum((nav_pts - obj_center)**2, axis=1)))
            farthest_pos = nav_pts[argmax_pos] 
            # controller.step('TeleportFull', position=dict(x=farthest_pos[0], y=farthest_pos[1], z=farthest_pos[2]), rotation=dict(x=0.0, y=turn_yaw, z=0.0), horizon=turn_pitch, standing=True, forceAction=True)

            # pos_visit = [closest_pos]
            pos_visit = []
            select = dists<=1.5
            dists2 = dists[select]
            nav_pts2 = nav_pts[select]
            if len(nav_pts2)==0:
                pos_visit += [closest_pos]
            else:
                argmin_pos = np.argsort(dists2)[len(dists2)//2]
                closest_pos = nav_pts2[argmin_pos]
                pos_visit += [closest_pos]
            # select = dists<=3.0
            # dists2 = dists[select]
            # nav_pts2 = nav_pts[select]
            # if len(nav_pts2)==0:
            #     pos_visit += [closest_pos]
            # else:
            #     argmin_pos = np.argmax(dists2)
            #     closest_pos = nav_pts2[argmin_pos]
            #     pos_visit += [closest_pos]
                # pos_visit = pos_visit[[2,1,0]]
            # print("visited:", pos_visit)


            rgbs = []
            for p_i in range(len(pos_visit)):

                closest_pos = pos_visit[p_i]

                # YAW calculation - rotate to object
                agent_to_obj = np.squeeze(obj_center) - (closest_pos + np.array([0.0, 0.675, 0.0]))
                agent_local_forward = np.array([0, 0, 1.0]) 
                flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
                flat_dist_to_obj = np.linalg.norm(flat_to_obj)
                flat_to_obj /= flat_dist_to_obj

                det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
                turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

                # # add noise so not right in the center
                # noise = np.random.normal(0, 2, size=2)

                turn_yaw = np.degrees(turn_angle) #+ noise[0]

                turn_pitch = -np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj)) #+ noise[1]

                controller.step('TeleportFull', position=dict(x=closest_pos[0], y=closest_pos[1]+0.675, z=closest_pos[2]), rotation=dict(x=0.0, y=turn_yaw, z=0.0), horizon=turn_pitch, standing=True, forceAction=True)
                origin_T_camX = utils.aithor.get_origin_T_camX(controller.last_event, False)
                # controller.step('TeleportFull', position=dict(x=farthest_pos[0], y=farthest_pos[1], z=farthest_pos[2]), rotation=dict(x=0.0, y=turn_yaw, z=0.0), horizon=turn_pitch, standing=True, forceAction=True)
            
                # rgbs = []
                # fovs = [120, 100, 90]
                # for fov in list(fovs):
                # fov = 100
                # third_party_event = controller.step(
                #     action="UpdateThirdPartyCamera",
                #     thirdPartyCameraId=0,
                #     position=dict(x=closest_pos[0], y=closest_pos[1]+0.675, z=closest_pos[2]),
                #     rotation=dict(x=turn_pitch, y=turn_yaw, z=0),
                #     fieldOfView=fov,
                # )
                rgb = controller.last_event.frame

                hfov = float(fov) * np.pi / 180.
                pix_T_camX = np.array([
                    [(W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
                    [0., (H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
                    [0., 0.,  1, 0],
                    [0., 0., 0, 1]])
                pix_T_camX[0,2] = W/2.
                pix_T_camX[1,2] = H/2.

                obj_3dbox_origin = utils.aithor.get_3dbox_in_geom_format(obj)
                # get amodal box
                # origin_T_camX = get_origin_T_camX(controller.last_event, False)
                boxlist2d_amodal, obj_3dbox_camX = utils.aithor.get_amodal2d(origin_T_camX.cuda(), obj_3dbox_origin.cuda(), torch.from_numpy(pix_T_camX).unsqueeze(0).cuda(), H, W)
                boxlist2d_amodal = boxlist2d_amodal.cpu().numpy()
                boxlist2d_amodal[[0,1]] = boxlist2d_amodal[[0,1]] - 5
                boxlist2d_amodal[[2,3]] = boxlist2d_amodal[[2,3]] + 5

                rect_th = 1
                img = rgb.copy()
                cv2.rectangle(img, (int(boxlist2d_amodal[0]), int(boxlist2d_amodal[1])), (int(boxlist2d_amodal[2]), int(boxlist2d_amodal[3])),(0, 255, 0), rect_th)

                img2 = np.zeros((img.shape[0]+5*2, img.shape[1]+5*2, 3)).astype(int)
                for i_i in range(3):
                    img2[:,:,i_i] = np.pad(img[:,:,i_i], pad_width=5, constant_values=255)
                rgbs.append(img2)

            # st()
            img = np.concatenate(rgbs, axis=1)

            # st()
            # plt.figure()
            # plt.imshow(img)
            # plt.savefig('images/test.png')

            name = obj['name']
            if obj['parentReceptacles'] is None:
                receptacle = 'Floor'
            else:
                receptacle = obj['parentReceptacles'][-1]
            image_dict[name] = {}
            image_dict[name]['rgb'] = img
            image_dict[name]['receptacle'] = receptacle.split('|')[0]

        controller.step('TeleportFull', position=dict(x=position_start["x"], y=position_start["y"], z=position_start["z"]), rotation=dict(x=rotation_start["x"], y=rotation_start["y"], z=rotation_start["z"]), horizon=head_tilt, standing=True, forceAction=True)

        return image_dict

        
class CheckSuccessfulAction():
    '''
    (Hack) Check action success by comparing RGBs. 
    TODO: replace with a network
    '''
    def __init__(self, rgb_init, H, W, perc_diff_thresh = 0.05, controller=None):
        '''
        rgb_init: the rgb image from the spawn viewpoint W, H, 3
        This class does a simple check with the previous image to see if it completed the action 
        '''
        self.rgb_prev = rgb_init
        self.perc_diff_thresh = perc_diff_thresh
        self.H = H
        self.W = W
        self.controller = controller

    def update_image(self, rgb):
        self.rgb_prev = rgb

    def check_successful_action(self, rgb):
        if args.use_GT_action_success:
            success = self.controller.last_event.metadata["lastActionSuccess"]
        else:
            num_diff = np.sum(np.sum(self.rgb_prev.reshape(self.W*self.H, 3) - rgb.reshape(self.W*self.H, 3), 1)>0)
            # diff = np.linalg.norm(self.rgb_prev - rgb)
            # print(num_diff)
            if num_diff < self.perc_diff_thresh*self.W*self.H:
                success = False
            else:
                success = True
            # self.rgb_prev = rgb
        return success

if __name__ == '__main__':
    Ai2Thor()
        
    


