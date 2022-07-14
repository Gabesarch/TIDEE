from time import sleep
from arguments import args
import numpy as np
from map_and_plan.mess.utils import Foo
from map_and_plan.mess.explore import Explore
from argparse import Namespace
import logging
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
import utils.aithor
import torch
import ipdb
st = ipdb.set_trace
import sys
from PIL import Image, ImageDraw
import os
import cv2

class Navigation():

    def __init__(
        self, 
        z=None, 
        keep_head_down=False, 
        keep_head_straight=False, 
        controller=None, 
        estimate_depth=True, 
        add_obs_when_navigating_if_explore_fail=False, 
        on_aws=False, 
        max_steps=500, 
        search_pitch_explore=False,
        pix_T_camX=None,
        task=None,
        ):  

        self.on_aws = on_aws
        print("Initializing NAVIGATION...")
        print("NOTICE: Make sure snapToGrid=False in the Ai2Thor Controller.")  
        self.action_fail_count = 0
        self.controller = controller

        self.W, self.H = args.W, args.H

        self.max_steps = max_steps

        if z is None:
            self.z = [0.05, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
            # self.z = [0.05, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
            # self.z = [0.05, 2.0]
            # self.z = [0.1, 2.0]
        else:
            self.z = z
        
        self.obs = Namespace()
        # self.verbose = True
        self.pix_T_camX = pix_T_camX
        self.obs.STEP_SIZE = args.STEP_SIZE #args.STEP_SIZE
        self.obs.DT = args.DT #args.DT
        self.obs.HORIZON_DT = args.HORIZON_DT #args.HORIZON_DT
        self.obs.camera_height = 1.5759992599487305 # fixed when standing up 
        self.obs.camera_aspect_ratio = [args.H, args.W]
        self.obs.camera_field_of_view = args.fov
        self.obs.head_tilt_init = 0 #event.metadata['agent']['cameraHorizon']
        self.obs.reward = 0
        self.obs.goal = Namespace(metadata={"category": 'cover'})
        actions = ['MoveAhead', 'MoveLeft', 'MoveRight', 'MoveBack', 'RotateRight', 'RotateLeft']
        self.actions = {i:actions[i] for i in range(len(actions))}
        self.action_mapping = {"Pass":0, "MoveAhead":1, "MoveLeft":2, "MoveRight":3, "MoveBack":4, "RotateRight":5, "RotateLeft":6, "LookUp":9, "LookDown":10}
        self.keep_head_down = keep_head_down
        self.keep_head_straight = keep_head_straight
        self.search_pitch_explore = search_pitch_explore
        self.add_obs_when_navigating_if_explore_fail = add_obs_when_navigating_if_explore_fail
        self.add_obs_when_navigating = False
        self.explorer = None
        self.task = task

    def init_navigation(self,bounds):
        self.step_count = 0
        self.goal = Foo()
        self.parse_goal(self.obs.goal)
        logging.error(self.goal.description)
        
        self.rng = np.random.RandomState(0)

        self.cover = False
        if self.goal.category == 'cover' or self.goal.category == 'point_nav':
            print("Z is:", self.z)
            self.explorer = Explore(
                self.obs, 
                self.goal, 
                bounds, 
                z=self.z, 
                keep_head_down=self.keep_head_down, 
                keep_head_straight=self.keep_head_straight, 
                search_pitch_explore=self.search_pitch_explore
                )
            self.cover = True 

    def parse_goal(self, goal, fig=None):
        self.goal.category = goal.metadata['category']
        self.goal.description = self.goal.category
        if self.goal.category == 'point_nav':
            self.goal.targets = goal.metadata['targets']
        else:
            self.goal.targets = ['dummy']
 
    def act(self, add_obs=True, fig=None, point_goal=None):
        self.step_count += 1
        if self.step_count == 1:
            pass

        if self.cover:
            action, param = self.explorer.act(self.obs, fig, point_goal, add_obs=add_obs) #, object_masks=object_masks)
            return action, param #, is_valids, inds

    def add_observation(self, action, add_obs=True):
        self.step_count += 1
        self.explorer.add_observation(self.obs, action, add_obs=add_obs)

    def get_map_pos_from_aithor_pos(self, aithor_pos):
        return self.explorer.mapper.get_position_on_map_from_aithor_position(aithor_pos)

    def get_clostest_reachable_map_pos(self, map_pos):
        reachable = self.get_reachable_map_locations(sample=False)
        inds_i, inds_j = np.where(reachable)
        reachable_where = np.stack([inds_i, inds_j], axis=0)
        dist = distance.cdist(np.expand_dims(map_pos, axis=0), reachable_where.T)
        argmin = np.argmin(dist)
        ind_i, ind_j = inds_i[argmin], inds_j[argmin]
        return ind_i, ind_j

    def get_reachable_map_locations(self, sample=True):
        reachable = self.explorer._get_reachable_area()
        state_xy = self.explorer.mapper.get_position_on_map()
        if sample:
            inds_i, inds_j = np.where(reachable)
            dist = np.sqrt(np.sum(np.square(np.expand_dims(state_xy,axis=1) - np.stack([inds_i, inds_j], axis=0)),axis=0))
            dist_thresh = dist>20.0
            inds_i = inds_i[dist_thresh]
            inds_j = inds_j[dist_thresh]
            if inds_i.shape[0]==0:
                print("FOUND NO REACHABLE INDICES")
                return [], []
            ind = np.random.randint(inds_i.shape[0])
            ind_i, ind_j = inds_i[ind], inds_j[ind]

            return ind_i, ind_j
        else:
            return reachable


    def set_point_goal(self, ind_i, ind_j, dist_thresh=0.3, search_mode=False, search_mode_reduced=False):
        '''
        ind_i and ind_j are indices in the map
        we denote camX0 as the first camera angle + position (spawning position) and camX as current camera angle + position
        '''
        self.obs.goal = Namespace(metadata={"category": 'point_nav', "targets":[ind_i, ind_j]})
        self.parse_goal(self.obs.goal)
        self.explorer.set_point_goal(ind_i, ind_j, dist_thresh=dist_thresh, search_mode=search_mode, search_mode_reduced=search_mode_reduced)

    def search_random_locs_for_object(
        self, 
        search_object,
        max_steps=75, 
        vis=None, 
        text='', 
        object_tracker=None, 
        max_fail=30, 
        ):
        '''
        Search random goals for object
        search_object: object category string or list of category strings - stop navigation and return object info when found
        '''

        for n in range(args.num_search_locs_object):
            ind_i, ind_j = self.get_reachable_map_locations(sample=True)
            self.set_point_goal(ind_i, ind_j, dist_thresh=args.dist_thresh)

            out = self.navigate_to_point_goal(
                vis=vis, 
                text=f"Search for {search_object}", 
                object_tracker=object_tracker,
                search_object=search_object,
                )

            if len(out)>0:
                return out

        return out

    def search_local_region(
        self, 
        vis=None, 
        text='Search local region', 
        object_tracker=None, 
        search_object=None,
        ):
        '''
        Search local region around agent where it is standing
        search_object: object category string or list of category strings - stop navigation and return object info when found
        '''

        out = {}

        action_sequence = ['RotateLeft', 'LookUp', 'RotateRight', 'RotateRight', 'LookDown', 'LookDown', 'RotateLeft', 'RotateLeft', 'LookUp', 'RotateRight']
        
        self.bring_head_to_center(vis=vis)

        steps = 0
        found_obj = False
        num_failed = 0
        for action in action_sequence:

            if self.task.is_done():
                print("Task done! Skipping search.")
                break

            if args.verbose:
                print(f"search_local_region: {action}")

            self.task.step(action=action)

            action_successful = self.task.action_success()
            rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
            self.update_navigation_obs(rgb,depth, action_successful)
            # whenever not acting - add obs
            self.add_observation(action, add_obs=False)

            camX0_T_camX = self.explorer.get_camX0_T_camX()

            if vis is not None:
                vis.add_frame(rgb, text=text)

            if object_tracker is not None and action_successful:
                object_tracker.update(rgb, depth, camX0_T_camX, vis=vis, target_object=search_object)

                if search_object is not None:
                    centroids, labels = object_tracker.get_centroids_and_labels(
                        return_ids=False, object_cat=search_object
                        )
                    if len(centroids)>0:
                        out = {'centroids':centroids, 'labels':labels}
                        return out
        if search_object is not None:
            return {}           

    def navigate_to_point_goal(
        self, 
        max_steps=75, 
        vis=None, 
        text='', 
        object_tracker=None, 
        search_object=None,
        max_fail=30, 
        ):
        '''
        search_object: object category string or list of category strings - stop navigation and return object info when found
        '''


        steps = 0
        num_failed = 0
        while True:

            if self.task.is_done():
                print("Task done! Skipping navigation to point goal.")
                break

            try:
                action, param = self.act(add_obs=False)
            except:
                action = 'Pass'
            
            camX0_T_camX = self.explorer.get_camX0_T_camX()

            if steps>0:
                if vis is not None:
                    vis.add_frame(rgb, text=text, add_map=True)
                if object_tracker is not None and action_successful:
                    object_tracker.update(rgb, depth, camX0_T_camX, vis=vis, target_object=search_object)

                    # if search_object is not None:
                    #     centroids, labels = object_tracker.get_centroids_and_labels(
                    #         return_ids=False, object_cat=search_object
                    #         )
                    #     if len(centroids)>0:
                    #         out = {'centroids':centroids, 'labels':labels}
                    #         return out

            if args.verbose:
                print(f"navigate_to_point_goal: {action}") #, action_rearrange, action_ind)

            if action=='Pass':
                break
            else:
                self.task.step(action=action)

                action_successful = self.task.action_success()
                rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                self.update_navigation_obs(rgb, depth, action_successful)

            if not action_successful:
                num_failed += 1

            steps += 1

            if object_tracker is not None:
                # object_tracker.update(rgb, depth, camX0_T_camX, vis=vis, target_object=search_object)
                if search_object is not None:
                    centroids, labels = object_tracker.get_centroids_and_labels(
                        return_ids=False, object_cat=search_object
                        )
                    if len(centroids)>0:
                        out = {'centroids':centroids, 'labels':labels}
                        return out

            if steps > max_steps:
                break

            if max_fail is not None:
                if num_failed >= max_fail:
                    if args.verbose: 
                        print("Max fail reached.")
                    break

        if search_object is not None:
            return {}   
        

    def orient_camera_to_point(
        self, 
        target_position, 
        vis=None, 
        text='Orient to object', 
        object_tracker=None
        ):


        pos_s = self.explorer.get_agent_position_camX0()
        target_position = np.array(list(target_position.values()))

        # YAW calculation - rotate to object
        agent_to_obj = np.squeeze(target_position) - pos_s 
        agent_local_forward = np.array([0, 0, 1.0]) 
        flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
        flat_dist_to_obj = np.linalg.norm(flat_to_obj)
        flat_to_obj /= flat_dist_to_obj

        det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
        turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

        turn_yaw = np.degrees(turn_angle) #+ noise[0]
        turn_pitch = np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj))

        yaw_cur, pitch_cur, _ = self.explorer.get_agent_rotations_camX0()

        relative_yaw = yaw_cur - turn_yaw
        if relative_yaw<-180:
            relative_yaw = relative_yaw+360
        if relative_yaw>180:
            relative_yaw = relative_yaw-360
        if relative_yaw > 0:
            yaw_action = 'RotateLeft'
        elif relative_yaw < 0:
            yaw_action = 'RotateRight'
        num_yaw = int(np.abs(np.round(relative_yaw / args.DT)))

        if num_yaw > 0:
            for t in range(num_yaw):

                if self.task.is_done():
                    print("Task done! Skipping orient to point goal.")
                    break

                if args.verbose:
                    print(f"orient_camera_to_point: {yaw_action}")

                self.task.step(action=yaw_action)

                action_successful = self.task.action_success()
                rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                self.update_navigation_obs(rgb,depth, action_successful)
                # whenever not acting - add obs
                self.add_observation(yaw_action, add_obs=False)

                if vis is not None:
                    vis.add_frame(rgb, text=text)

        if turn_pitch > 30.:
            turn_pitch = 30.
        if turn_pitch < -60.:
            turn_pitch = -60.

        relative_pitch = turn_pitch - pitch_cur
        if relative_pitch < 0:
            pitch_action = 'LookDown'
        elif relative_pitch > 0:
            pitch_action = 'LookUp'
        else:
            pitch_action = None
        num_pitch = int(np.abs(np.round(relative_pitch / args.HORIZON_DT)))
        print("num_pitch",num_pitch)

        if num_pitch > 0:
            for t in range(num_pitch):

                if self.task.is_done():
                    print("Task done! Skipping orient to point goal.")
                    break

                if args.verbose:
                    print(f"orient_camera_to_point: {pitch_action}")

                self.task.step(action=pitch_action)

                action_successful = self.task.action_success()
                rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                self.update_navigation_obs(rgb,depth, action_successful)
                # whenever not acting - add obs
                self.add_observation(pitch_action, add_obs=False)

                if vis is not None:
                    vis.add_frame(rgb, text=text)


    def adjust_depth(self, depth_frame):
        mask_err_below = depth_frame < 0.5
        depth_frame[mask_err_below] = 100.0
        return depth_frame


    def explore_env(
        self, 
        vis=None, 
        object_tracker=None, 
        max_fail=None, 
        max_steps=200, 
        return_obs_dict=False,
        use_aithor_coord_frame=False
        ):
        '''
        This function explores the environment based on 
        '''
        
        fig = None 

        step = 0
        valid = 0
        num_failed = 0
        if return_obs_dict:
            obs_dict = {'rgb':[], 'xyz':[], 'camX0_T_camX':[], 'camX0_candidates':[], 'pitch':[], 'yaw':[]}
        change_list = []
        while True:

            if step==0:
                
                self.init_navigation(None)

                if object_tracker is not None:
                    object_tracker.navigation = self

                if use_aithor_coord_frame:
                    camX0_T_camX = utils.aithor.get_origin_T_camX(self.task.controller.last_event, True)
                else:
                    camX0_T_camX = self.explorer.get_camX0_T_camX()
                rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                action_successful = True

                self.update_navigation_obs(rgb,depth, action_successful)

                if use_aithor_coord_frame:
                    camX0_T_camX = utils.aithor.get_origin_T_camX(self.task.controller.last_event, True)
                else:
                    camX0_T_camX = self.explorer.get_camX0_T_camX()
                camX0_T_camX0 = utils.geom.safe_inverse_single(camX0_T_camX)
                if vis is not None:
                    vis.camX0_T_camX0 = camX0_T_camX0
                    if object_tracker is not None:
                        vis.object_tracker = object_tracker

                if return_obs_dict:
                    # need this for supervision for visual search network (not used during inference)
                    origin_T_camX0_invert = utils.aithor.get_origin_T_camX(self.task.controller.last_event, True)


            try:
                action, param = self.act()
                action_ind = self.action_mapping[action]
            except:
                action = 'Pass'

            if use_aithor_coord_frame:
                camX0_T_camX = utils.aithor.get_origin_T_camX(self.task.controller.last_event, True)
            else:
                camX0_T_camX = self.explorer.get_camX0_T_camX()

            if return_obs_dict:
                rgb_x = rgb/255.
                rgb_x = torch.from_numpy(rgb_x).permute(2,0,1).float()
                depth_ = torch.from_numpy(depth).cuda().unsqueeze(0).unsqueeze(0)
                xyz = utils.geom.depth2pointcloud(depth_, torch.from_numpy(self.pix_T_camX).cuda().unsqueeze(0).float())
                depth_threshold = 0.5
                percent_depth_thresh = 0.5 # only keep views with proportion of depth > threshold (dont want looking at wall)
                if (np.count_nonzero(depth<depth_threshold)/(args.H*args.W) < percent_depth_thresh):
                    obs_dict['rgb'].append(rgb_x.unsqueeze(0))
                    obs_dict['xyz'].append(xyz)
                    obs_dict['camX0_T_camX'].append(camX0_T_camX.unsqueeze(0))
                    obs_dict['pitch'].append(torch.tensor([self.explorer.head_tilt]))
                    obs_dict['yaw'].append(torch.tensor([self.explorer.rotation]))
                    is_camX0_candidate = np.round(self.explorer.head_tilt)==0 and np.round(self.explorer.rotation)%90==0.
                    if is_camX0_candidate:
                        obs_dict['camX0_candidates'].append(torch.tensor([valid]))
                    valid += 1

            if step==0:
                if vis is not None:
                    vis.add_frame(rgb, text="Explore", add_map=False)
            else:
                if vis is not None:
                    vis.add_frame(rgb, text="Explore", add_map=True)
            
            if object_tracker is not None and action_successful:
                object_tracker.update(rgb, depth, camX0_T_camX, vis=vis)

            # if step==10:
            #     objects = self.task.controller.last_event.metadata['objects']
            #     centroids, labels = object_tracker.get_centroids_and_labels()
            #     for obj in objects:
            #         if obj['objectType']=='Microwave':
            #             break
            #     obj_center = torch.from_numpy(np.array(list(obj['axisAlignedBoundingBox']['center'].values()))).unsqueeze(0)
            #     camX0_T_origin = utils.geom.safe_inverse_single(origin_T_camX0_invert)
            #     obj_center_camX0 = utils.geom.apply_4x4(camX0_T_origin, obj_center.unsqueeze(0))
            #     st()

            if args.verbose:
                print(f"explore_env: {action}")

            if action=='Pass':
                break
            else:
                self.task.step(action=action)

                action_successful = self.task.action_success()
                rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                self.update_navigation_obs(rgb,depth, action_successful)

            if not action_successful:
                num_failed += 1
            if max_fail is not None:
                if num_failed >= max_fail:
                    if args.verbose: 
                        print("Max fail reached.")
                    break

            step += 1

        print("num steps taken:", step)

        if return_obs_dict:
            for key in list(obs_dict.keys()):
                obs_dict[key] = torch.cat(obs_dict[key], dim=0).cpu().numpy()
            if object_tracker is not None:
                obs_dict["objects_track_dict"] = object_tracker.objects_track_dict
            obs_dict['origin_T_camX0'] = origin_T_camX0_invert.cpu().numpy()
            return obs_dict

    def interact_object_xy(
        self, 
        action, 
        point_2D, 
        vis=None, 
        num_tries_place=2,
        offsets =   [
                    [0, 0], 
                    [0, 20], 
                    [20, 0], 
                    [0, -20], 
                    [-20, 0], 
                    [20, 20], 
                    [20, -20], 
                    [-20, 20], 
                    [-20, -20]
                    ]
        ):
        '''
        Used for picking and placing objects by specifying 2D point
        action: either ["PickupObject", "PutObject"]
        point_2D: point in image specifying object to pickup or place held object onto. Should be in image coordinates (i.e. in [0, W/H-1])
        offsets: offsets from initial point to try placement if previous placement fails
        '''

        success = False

        assert(action in ["PickupObject", "PutObject"]) # action must be PickupObject or PutObject
        
        for mult in range(1, num_tries_place+1):
            for offset in offsets:

                if self.task.is_done():
                    print("Task done! Skipping object interaction.")
                    break

                if offsets==[0,0] and mult>1: # skip 0,0 on second to last try
                    continue

                point_2D_ = point_2D.copy()
                point_2D_[0] = (point_2D_[0] + offset[0]*mult) / self.W
                point_2D_[1] = (point_2D_[1] + offset[1]*mult) / self.H       
                obj_relative_coord = [max(min(point_2D_[1], 0.99), 0.01), max(min(point_2D_[0], 0.99), 0.01)]
                
                if args.verbose:
                    print(f"interact_object_xy: {action}@{obj_relative_coord}")

                self.task.step(action, obj_relative_coord)

                success = self.task.action_success()

                if vis is not None:
                    if success: 
                        text = f"success {action}@{obj_relative_coord}"
                    else:
                        text = f"fail {action}@{obj_relative_coord}"
                    for _ in range(5):
                        rgb = np.float32(self.get_obs()[0]) #np.float32(self.get_image(self.controller))
                        rgb = cv2.circle(rgb, (int(point_2D_[1]* self.W),int(point_2D_[0]* self.H)), radius=5, color=(0, 0, 255),thickness=2)
                        vis.add_frame(rgb, text=text, add_map=True)

                if success:
                    break
            if success:
                break

        if not self.task.is_done():
            if success: 
                print(f"(Success) interact_object_xy: {action}@{obj_relative_coord}")
            else:
                print(f"(Fail) interact_object_xy: {action}@{obj_relative_coord}")
        
        return success

    def get_obs(self, head_tilt=None):
        obs = self.task.get_observations()
        rgb = obs["rgb"]
        depth = obs["depth"]
        depth = self.adjust_depth(depth.copy())
        return rgb, depth

    def update_navigation_obs(self, rgb, depth, action_successful):
        '''
        updates navigation mapping inputs
        rgb: rgb of current frame
        depth: depth map current frame
        action_successful: whether previous action was successful
        navigation: navigation class 
        '''
        self.obs.image_list = [rgb]
        self.obs.depth_map_list = [depth]
        self.obs.return_status = "SUCCESSFUL" if action_successful else "OBSTRUCTED"

    def bring_head_to_center(
        self, 
        vis=None,
        text='head to center',
        object_tracker=None, 
        ):

        yaw_cur, pitch_cur, _ = self.explorer.get_agent_rotations_camX0()

        relative_pitch = 0 - pitch_cur
        if relative_pitch < 0:
            pitch_action = 'LookDown'
        elif relative_pitch > 0:
            pitch_action = 'LookUp'
        else:
            pitch_action = None
        num_pitch = int(np.abs(np.round(relative_pitch / args.HORIZON_DT)))

        if num_pitch > 0:
            for t in range(num_pitch):

                if args.verbose:
                    print(f"bring_head_to_center: {pitch_action}")

                camX0_T_camX = self.explorer.get_camX0_T_camX()

                if object_tracker is not None and t>0 and action_successful:
                    object_tracker.update(rgb, depth, camX0_T_camX, vis=vis)
                
                self.task.step(action=pitch_action)

                action_successful = self.task.action_success()
                rgb, depth = self.get_obs(head_tilt=self.explorer.head_tilt)
                self.update_navigation_obs(rgb,depth, action_successful)
                # whenever not acting - add obs
                self.add_observation(pitch_action, add_obs=False)

                if vis is not None:
                    vis.add_frame(rgb, text=text)

    def update_obs(
        self, 
        rgb, 
        depth, 
        action_successful, 
        update_success_checker=False
        ):
        self.obs.image_list = [rgb]
        self.obs.depth_map_list = [depth]
        self.obs.return_status = "SUCCESSFUL" if action_successful else "OBSTRUCTED"
        if update_success_checker:
            self.success_checker.update_image(rgb) # update action success checker with new image


class CheckSuccessfulAction():
    def __init__(self, rgb_init, H, W, perc_diff_thresh = 0.01):
        '''
        rgb_init: the rgb image from the spawn viewpoint W, H, 3
        This class does a simple check with the previous image to see if it completed the action 
        '''
        self.rgb_prev = rgb_init
        self.perc_diff_thresh = perc_diff_thresh
        self.H = H
        self.W = W

    def update_image(self, rgb):
        self.rgb_prev = rgb

    def check_successful_action(self, rgb):
        num_diff = np.sum(np.sum(self.rgb_prev.reshape(self.W*self.H, 3) - rgb.reshape(self.W*self.H, 3), 1)>0)
        if num_diff < self.perc_diff_thresh*self.W*self.H:
            success = False
        else:
            success = True
        return success

class Depth():
    def __init__(self, estimate_depth=True, controller=None, on_aws=False, DH=300, DW=300):

        from nets.alfred_perception_models import AlfredSegmentationAndDepthModel
        from collections import Counter, OrderedDict

        self.controller = controller

        self.W = args.W
        self.H = args.H

        self.DH = DH
        self.DW = DW

        self.estimate_depth = estimate_depth
        if self.estimate_depth:
            self.map_pred_threshold = 65
            self.no_pickup_update = True
            self.cat_pred_threshold = 10
            self.valts_depth = True
            self.valts_trustworthy = True
            self.valts_trustworthy_prop = 0.9

            # self.valts_trustworthy_obj_prop0 = 1.0
            # self.valts_trustworthy_obj_prop = 1.0

            self.valts_trustworthy_obj_prop0 = 1.0
            self.valts_trustworthy_obj_prop = 1.0

            self.learned_visibility = True
            self.learned_visibility_no_mask = True
            self.separate_depth_for_straight = False
            self.depth_model_45_only = False
            self.depth_model_old = False
            self.min_depth = 0.0
            # self.max_depth = 5.0
            self.max_depth = 5.0
            self.max_depth_filter = 15.0
            self.num_bins = int(self.max_depth/0.1)
            print(f"MAX DEPTH: {self.max_depth}; NUMBER OF DEPTH BINS: {self.num_bins}")
            self.use_sem_seg = False

            self.depth_pred_model = AlfredSegmentationAndDepthModel(self.num_bins, self.max_depth, vec_head=True) # vec_head=False, num_c=0)

            path = args.depth_checkpoint_45

            state_dict = torch.load(path)['model']

            new_checkpoint = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_checkpoint[name] = v
            
            state_dict = new_checkpoint
            del new_checkpoint

            self.depth_pred_model.load_state_dict(state_dict)
            self.depth_pred_model.eval()
            self.depth_pred_model.cuda()

            if self.separate_depth_for_straight:
                self.max_depth_0 = 5.0
                self.num_bins_0 = int(self.max_depth_0/0.1)

                self.depth_pred_model_0 = AlfredSegmentationAndDepthModel(self.num_bins_0, self.max_depth_0, vec_head=True)
                path = args.depth_checkpoint_0
                state_dict = torch.load(path)['model']

                new_checkpoint = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_checkpoint[name] = v
                
                state_dict = new_checkpoint
                del new_checkpoint

                self.depth_pred_model_0.load_state_dict(state_dict)
                self.depth_pred_model_0.eval()
                self.depth_pred_model_0.cuda()

            if self.use_sem_seg:
                self.seg = SemgnetationHelper(self)

    def get_depth_map(self, rgb, head_tilt, filter_depth_by_sem=False):
        
        if self.estimate_depth:
            with torch.no_grad():
                if not (rgb.shape[0]==self.DW and rgb.shape[1]==self.DH):
                    rgb = cv2.resize(rgb.copy(), (self.DW, self.DH), interpolation = cv2.INTER_AREA)
                include_mask = torch.tensor(np.ones((1,1,self.DW,self.DH)).astype(bool).astype(float)).cuda()
                depth = self.depth_pred_later(rgb, head_tilt, include_mask)
                depth = self._preprocess_depth(depth, self.min_depth, self.max_depth_filter)
        else:
            print("Using GT depth")
            depth = self.controller.last_event.depth_frame

        return depth

    def depth_pred_later(self, rgb, camera_horizon, sem_seg_pred=None):
        # model is trained on 300 300 
        rgb = rgb.copy()
        if not (self.W==self.DW and self.H==self.DH):
            rgb = cv2.resize(rgb, (self.DW, self.DH), interpolation = cv2.INTER_AREA)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        rgb_image = torch.div(torch.from_numpy(rgb).permute((2, 0, 1)).unsqueeze(0).float(), 255.) # Note: removed .half()

        if camera_horizon > 25:
            _, pred_depth = self.depth_pred_model.predict(rgb_image.cuda().float())
            include_mask_prop=self.valts_trustworthy_obj_prop
            depth_img = pred_depth.get_trustworthy_depth(max_conf_int_width_prop=self.valts_trustworthy_prop, include_mask=sem_seg_pred, include_mask_prop=include_mask_prop) #default is 1.0
            depth_img = depth_img.squeeze().detach().cpu().numpy()
            self.learned_depth_frame = pred_depth.depth_pred.detach().cpu().numpy()
            self.learned_depth_frame = self.learned_depth_frame.reshape((self.num_bins,self.DW,self.DH))
            self.learned_depth_frame = self.max_depth * 1/self.num_bins * np.argmax(self.learned_depth_frame, axis=0) #Now shape is (300,300)
            del pred_depth
            depth = depth_img
            if not (self.W==self.DW and self.H==self.DH):
                depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            depth = np.expand_dims(depth, 2)
        elif self.separate_depth_for_straight and camera_horizon==0:
            _, pred_depth = self.depth_pred_model_0.predict(rgb_image.cuda().float())
            include_mask_prop=self.valts_trustworthy_obj_prop0
            depth_img = pred_depth.get_trustworthy_depth(max_conf_int_width_prop=self.valts_trustworthy_prop, include_mask=sem_seg_pred, include_mask_prop=include_mask_prop) #default is 1.0
            depth_img = depth_img.squeeze().detach().cpu().numpy()
            self.learned_depth_frame = pred_depth.depth_pred.detach().cpu().numpy()
            self.learned_depth_frame = self.learned_depth_frame.reshape((self.num_bins,self.DW,self.DH))
            self.learned_depth_frame = self.max_depth * 1/self.num_bins * np.argmax(self.learned_depth_frame, axis=0) #Now shape is (300,300)
            del pred_depth
            depth = depth_img
            if not (self.W==self.DW and self.H==self.DH):
                depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            depth = np.expand_dims(depth, 2)
        else:
            depth = np.ones((self.W, self.H, 1))*100.0 # make all depths huge because depth estimation would be innaccurate

        return depth

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0]*1 #shape (h,w)

        if False: #self.picked_up:
            mask_err_below = depth <0.5
            if not(self.picked_up_mask is None):
                mask_picked_up = self.picked_up_mask == 1
                depth[mask_picked_up] = 100.0 #100.0
        else:
            mask_err_below = np.logical_or(depth < min_d, depth > max_d)
        depth[mask_err_below] = 100.0 #100.0
        
        return depth


def save_mapping_obs():

    create_movie = False

    from task_base.aithor_base import Base
    from tidee.object_tracker import ObjectTrack
    from task_base.animation_util import Animation
    from nets.solq import DDETR
    from task_base.tidy_task import TIDEE_TASK
    from backend import saverloader
    import os
    import pickle
    base = Base()
    include_classes = base.include_classes
    mapnames_train = base.mapnames_train
    mapnames_val = base.mapnames_val
    mapnames_test = base.mapnames_test
    num_objects = args.num_objects
    controller = base.controller
    tidee_task = TIDEE_TASK(controller, "train")
    tidee_task.next_ep_called = True # hack to start episode
    tidee_task.done_called = False

    if args.do_predict_oop:
        num_classes = len(base.include_classes) 
        base.label_to_id = {False:0, True:1} # not out of place = False
        base.id_to_label = {0:'IP', 1:'OOP'}
        num_classes2 = 2 # 0 oop + 1 not oop + 2 no object
        base.score_boxes_name = 'pred1'
        base.score_labels_name = 'pred1'
        base.score_labels_name2 = 'pred2' # which prediction head has the OOP label
    else:
        num_classes = len(base.include_classes) 
        num_classes2 = None
        base.score_boxes_name = 'pred1' # only one prediction head so same for both
        base.score_labels_name = 'pred1'
    ddetr = DDETR(num_classes, load_pretrained=False, num_classes2=num_classes2)
    ddetr.cuda()
    print("loading checkpoint for solq...")
    ddetr_path = args.SOLQ_oop_checkpoint 
    _ = saverloader.load_from_path(ddetr_path, ddetr, None, strict=(not args.load_strict_false), lr_scheduler=None)
    print("loaded!")
    ddetr.eval()

    keep_head_down = False
    keep_head_straight = False
    search_pitch_explore = True
    # min_depth = None
    # max_depth = None

    for mapname in mapnames_train:
        print(mapname)
        root = os.path.join(args.mapping_obs_dir, mapname)
        if not os.path.exists(root):
            os.mkdir(root)
        for n in range(args.n_train_mapping_obs):
            
            pickle_fname = f'{n}.p'
            fname_ = os.path.join(root, pickle_fname)
            
            # if os.path.exists(fname_):
            #     continue

            tidee_task.controller.reset(scene=mapname)

            tidee_task.mapname_current = mapname

            tidee_task.step_count = 0

            object_tracker = ObjectTrack(
                    base.name_to_id, 
                    base.id_to_name, 
                    base.include_classes, 
                    base.W, base.H, 
                    pix_T_camX=base.pix_T_camX, 
                    # origin_T_camX0=None, 
                    ddetr=ddetr, 
                    controller=None, 
                    use_gt_objecttrack=False,
                    do_masks=True,
                    use_solq=True,
                    id_to_mapped_id=base.id_to_mapped_id,
                    on_aws = False, 
                    # navigator=navigation,
                    )

            navigation = Navigation(
                # controller=controller, 
                keep_head_down=keep_head_down, 
                keep_head_straight=keep_head_straight, 
                search_pitch_explore=search_pitch_explore, 
                pix_T_camX=base.pix_T_camX,
                task=tidee_task,
                )
            object_tracker.navigation = navigation

            if create_movie:
                print("LOGGING THIS ITERATION")
                vis = Animation(base.W, base.H, navigation=navigation, name_to_id=base.name_to_id)
                # print('Height:', self.H, 'Width:', self.W)
            else:
                vis = None

            obs_dict = navigation.explore_env(object_tracker=object_tracker, vis=vis, return_obs_dict=True)

            if vis is not None:
                vis.render_movie(args.movie_dir,0, tag='test_obs')

            print("saving", fname_)
            with open(fname_, 'wb') as f:
                pickle.dump(obs_dict, f, protocol=4)
            print("done.")

    for mapname in mapnames_val:
        print(mapname)
        root = os.path.join(args.mapping_obs_dir, mapname)
        if not os.path.exists(root):
            os.mkdir(root)
        for n in range(args.n_val_mapping_obs):

            pickle_fname = f'{n}.p'
            fname_ = os.path.join(root, pickle_fname)
            
            if os.path.exists(fname_):
                continue

            tidee_task.controller.reset(scene=mapname)

            tidee_task.mapname_current = mapname

            tidee_task.step_count = 0

            object_tracker = ObjectTrack(
                    base.name_to_id, 
                    base.id_to_name, 
                    base.include_classes, 
                    base.W, base.H, 
                    pix_T_camX=base.pix_T_camX, 
                    # origin_T_camX0=None, 
                    ddetr=ddetr, 
                    controller=None, 
                    use_gt_objecttrack=False,
                    do_masks=True,
                    use_solq=True,
                    id_to_mapped_id=base.id_to_mapped_id,
                    on_aws = False, 
                    # navigator=navigation,
                    )

            navigation = Navigation(
                controller=controller, 
                keep_head_down=keep_head_down, 
                keep_head_straight=keep_head_straight, 
                search_pitch_explore=search_pitch_explore, 
                pix_T_camX=base.pix_T_camX,
                task=tidee_task,
                )
            object_tracker.navigation = navigation
            
            if create_movie:
                print("LOGGING THIS ITERATION")
                vis = Animation(base.W, base.H, navigation=navigation, name_to_id=base.name_to_id)
                # print('Height:', self.H, 'Width:', self.W)
            else:
                vis = None
            
            obs_dict = navigation.explore_env(object_tracker=object_tracker, vis=vis, return_obs_dict=True)

            if vis is not None:
                vis.render_movie(args.movie_dir,0, tag='test_obs')

            print("saving", fname_)
            with open(fname_, 'wb') as f:
                pickle.dump(obs_dict, f, protocol=4)
            print("done.")
    #####%%%%%%%% UNCOMMENT BELOW IF WANT TESTING OBS %%%%%########
    # for mapname in mapnames_test:
    #     print(mapname)
    #     root = os.path.join(args.mapping_obs_dir, mapname)
    #     if not os.path.exists(root):
    #         os.mkdir(root)
    #     for n in range(args.n_test_mapping_obs):

    #         pickle_fname = f'{n}.p'
    #         fname_ = os.path.join(root, pickle_fname)
            
    #         if os.path.exists(fname_):
    #             continue

    #         tidee_task.controller.reset(scene=mapname)

            # tidee_task.mapname_current = mapname

    #         object_tracker = ObjectTrack(
    #                 base.name_to_id, 
    #                 base.id_to_name, 
    #                 base.include_classes, 
    #                 base.W, base.H, 
    #                 pix_T_camX=base.pix_T_camX, 
    #                 # origin_T_camX0=None, 
    #                 ddetr=ddetr, 
    #                 controller=None, 
    #                 use_gt_objecttrack=False,
    #                 do_masks=True,
    #                 use_solq=True,
    #                 id_to_mapped_id=base.id_to_mapped_id,
    #                 on_aws = False, 
    #                 # navigator=navigation,
    #                 )

    #         navigation = Navigation(
    #             controller=controller, 
    #             keep_head_down=keep_head_down, 
    #             keep_head_straight=keep_head_straight, 
    #             search_pitch_explore=search_pitch_explore, 
    #             pix_T_camX=base.pix_T_camX
    #             )
    #         object_tracker.navigation = navigation
            
    #         if create_movie:
    #             print("LOGGING THIS ITERATION")
    #             vis = Animation(base.W, base.H, navigation=navigation, name_to_id=base.name_to_id)
    #             # print('Height:', self.H, 'Width:', self.W)
    #         else:
    #             vis = None
            
    #         obs_dict = navigation.explore_env(object_tracker=object_tracker, vis=vis, return_obs_dict=True)

    #         if vis is not None:
    #             vis.render_movie(args.movie_dir,0, tag='test_obs')

    #         print("saving", fname_)
    #         with open(fname_, 'wb') as f:
    #             pickle.dump(obs_dict, f, protocol=4)
    #         print("done.")

    
if __name__ == '__main__':
    pass


