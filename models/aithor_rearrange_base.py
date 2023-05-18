import os
import sys
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
from task_base.aithor_base import Base
import cv2
import math
"""Inference loop for the AI2-THOR object rearrangement task."""
from utils.wctb import Utils, Relations_CenterOnly
import re
import time
import json
import gzip
import os
from utils.noise_models.sim_kinect_noise import add_gaussian_shifts, filterDisp

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

class Ai2Thor_Base(Base):

    def __init__(self):   

        super(Ai2Thor_Base, self).__init__()

    def update_navigation_obs(self, rgb, depth, action_successful, navigation, update_success_checker=True):
        '''
        updates navigation mapping inputs
        rgb: rgb of current frame
        depth: depth map current frame
        action_successful: whether previous action was successful
        navigation: navigation class 
        '''
        # action_successful = controller.last_event.metadata['lastActionSuccess']
        navigation.obs.image_list = [rgb]
        navigation.obs.depth_map_list = [depth]
        navigation.obs.return_status = "SUCCESSFUL" if action_successful else "OBSTRUCTED"

        if update_success_checker:
            self.success_checker.update_image(rgb)



    def move_object_state(
        self, 
        obj_label, 
        current_state, 
        goal_state, 
        action,
        navigation, 
        controller, 
        task, 
        max_steps=50, 
        vis=None
        ):
        '''
        Moves object from current_state to goal_state
        current_state: 3D centroid of object in walkthrough phase
        goal_state: 3D centroid of object in unshuffle phase
        task: rearrange task class
        navigation: navigation class 
        controller: ai2thor controller (only used for visualization)
        '''

        print("Note: subtracting object y by agent height before point nav")
        obj_center_camX0_ = {'x':current_state[0], 'y':-current_state[1], 'z':current_state[2]}
        map_pos = navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)

        ind_i, ind_j  = navigation.get_clostest_reachable_map_pos(map_pos) # get closest navigable point to object in overhead map

        navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.dist_thresh) # set point goal in map
        self.navigate_to_point_goal(controller, task, navigation, vis=vis, max_steps=max_steps, text=f"Navigate to {obj_label}", object_tracker=None) # navigate to point goal

        # orient to object
        navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.dist_thresh)
        self.orient_camera_to_point(controller, obj_center_camX0_, navigation, task, vis=vis, text=f"Orient to {obj_label}", object_tracker=None)

        if task.is_done():
            return

        if action=='pickup':
            print("PICKING UP OBJECT")
            action_pickup = 'pickup_' + undo_format_a(obj_label)
            action_ind = self.action_to_ind[action_pickup]
            task.step(action=action_ind)
            if vis is not None:
                for _ in range(5):
                    vis.add_frame(controller.last_event.frame, text="place")
        elif action=='open':
            print("OPENING OBJECT")
            action_pickup = 'open_by_type_' + undo_format_a(obj_label)
            action_ind = self.action_to_ind[action_pickup]
            task.step(action=action_ind)
            if vis is not None:
                for _ in range(5):
                    vis.add_frame(controller.last_event.frame, text="open")
            return
        else:
            assert(False) # what action is this

        if goal_state is None:
            return

        print("Note: subtracting object y by agent height before point nav")
        obj_center_camX0_ = {'x':goal_state[0], 'y':-goal_state[1], 'z':goal_state[2]}
        map_pos = navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)

        ind_i, ind_j  = navigation.get_clostest_reachable_map_pos(map_pos)

        navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.dist_thresh) # set point goal in map
        self.navigate_to_point_goal(controller, task, navigation, vis=vis, max_steps=max_steps, text=f"Navigate to {obj_label}", object_tracker=None) # navigate to point goal

        # orient to 3D point
        navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.dist_thresh)
        self.orient_camera_to_point(controller, obj_center_camX0_, navigation, task, vis=vis, text=f"Orient to {obj_label}", object_tracker=None)
        
        if not task.is_done():
            print("PLACING OBJECT")
            action_ind = self.action_to_ind['drop_held_object_with_snap']
            task.step(action=action_ind)

        if vis is not None:
            for _ in range(5):
                vis.add_frame(controller.last_event.frame, text="place")

    def navigate_to_point_goal(self, controller, task, navigation, vis=None, max_steps=50, text='', object_tracker=None):
        '''
        Navigates to point after setting the point goal in the map with navigation.set_point_goal
        task: rearrange task class
        navigation: navigation class 
        controller: ai2thor controller (only used for visualization)
        '''

        steps = 0
        while True:

            if task.is_done():
                break

            try:
                action, param = navigation.act(add_obs=False)
            except:
                action = 'Pass'
            action_rearrange = self.nav_action_to_rearrange_action[action]
            action_ind = self.action_to_ind[action_rearrange]

            if vis is not None:
                vis.add_frame(controller.last_event.frame, text=text)

            if self.verbose:
                print(action, action_rearrange, action_ind)

            if task.num_steps_taken() % 10 == 0:
                print(
                    f"{self.current_mode}: point nav (step {task.num_steps_taken()}):"
                    f" taking action {task.action_names()[action_ind]}"
                )
            if action=='Pass':
                break   
            else:
                task.step(action=action_ind)

                rgb, depth = self.get_obs(task, head_tilt=navigation.explorer.head_tilt)
                action_successful = self.success_checker.check_successful_action(rgb)
                self.update_navigation_obs(rgb,depth, action_successful, navigation)

            steps += 1

            if steps > max_steps:
                break


    def orient_camera_to_point(self, controller, target_position, navigation, task, object_tracker=None, vis=None, text=''):
        '''
        Orients yaw and pitch of camera to 3D point
        target_position: target 3D point to orient to in 3D coordinates relative to agent initial position
        task: rearrange task class
        navigation: navigation class 
        controller: ai2thor controller (used for visualization)
        '''

        pos_s = navigation.explorer.get_agent_position_camX0()
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

        yaw_cur, pitch_cur, _ = navigation.explorer.get_agent_rotations_camX0()

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
            action_rearrange = self.nav_action_to_rearrange_action[yaw_action]
            action_ind = self.action_to_ind[action_rearrange]
            for t in range(num_yaw):

                if task.is_done():
                    break
                if self.verbose:
                    print('Orienting to object', yaw_action)
                task.step(action=action_ind)

                rgb, depth = self.get_obs(task, head_tilt=navigation.explorer.head_tilt)
                action_successful = self.success_checker.check_successful_action(rgb)
                self.update_navigation_obs(rgb,depth, action_successful, navigation)
                # whenever not acting - add obs
                try:
                    navigation.add_observation(yaw_action, add_obs=False)
                except:
                    break

                if vis is not None:
                    vis.add_frame(controller.last_event.frame, text=text)

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
            action_rearrange = self.nav_action_to_rearrange_action[pitch_action]
            action_ind = self.action_to_ind[action_rearrange]
            for t in range(num_pitch):

                if task.is_done():
                    break

                if self.verbose:
                    print('Orienting to object', pitch_action)
                # print('pitch')
                task.step(action=action_ind)

                if vis is not None:
                    vis.add_frame(controller.last_event.frame, text=text)

                rgb, depth = self.get_obs(task, head_tilt=navigation.explorer.head_tilt)
                action_successful = self.success_checker.check_successful_action(rgb)
                self.update_navigation_obs(rgb,depth, action_successful, navigation)
                # whenever not acting - add obs
                try:
                    navigation.add_observation(pitch_action, add_obs=False)
                except:
                    break

    def get_obs(self, task, head_tilt=None, controller=None):
        observations = task.get_observations()
        rgb = observations['rgb'] * 255. 
        rgb = rgb.astype(np.uint8)
        if self.depth_estimator is not None:
            depth = self.depth_estimator.get_depth_map(rgb, head_tilt)
        else:
            depth = np.squeeze(observations['depth'])
            if self.noisy_depth:
                depth = self.get_noisy_depth_map(depth)

        return rgb, depth 

    def get_noisy_depth_map(self, depth):
        '''
        Simulates lidar noise
        taken from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/icra2014.pdf
        '''
        # various variables to handle the noise modelling
        scale_factor  = 100     # converting depth from m to cm 
        focal_length  = self.W #480.0   # focal length of the camera used 
        baseline_m    = 0.075   # baseline in m 
        invalid_disp_ = 99999999.9

        h, w = depth.shape 

        # depth_interp = add_gaussian_shifts(depth)

        # disp_= focal_length * baseline_m / (depth_interp + 1e-10)
        # depth_f = np.round(disp_ * 8.0)/8.0

        # out_disp = filterDisp(depth_f, self.dot_pattern_, invalid_disp_)

        # depth = focal_length * baseline_m / out_disp
        # depth[out_disp == invalid_disp_] = 0 
        # factor = 6.0
        factor = 5.0
        
        # The depth here needs to converted to cms so scale factor is introduced 
        # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
        depth = (35130/np.round((35130/np.round(depth*scale_factor)) + np.random.normal(size=(h, w))*(1.0/factor) + 0.5))/scale_factor 

        # plt.figure()
        # plt.imshow(depth)
        # plt.savefig('images/test2.png')
        depth[depth<=0.05] = 100.0 # filter out na depth
        depth = np.float32(depth)

        return depth

    def explore_env(
        self, 
        task, 
        navigation, 
        controller, 
        steps_max=None, 
        vis=None, 
        phase='', 
        inds_explore=[], 
        ):
        '''
        Explores the environment and maps it out
        task: rearrange task class
        navigation: navigation class 
        controller: ai2thor controller (used for visualization)
        steps_max: maximum number of steps to take during exploration
        vis: animation util class for generating movies
        '''
        
        step = 0
        num_sampled = 0
        exploring = True
        while not task.is_done():

            if step==0:
                
                navigation.init_navigation(None)

                object_tracker = ObjectTrack(
                    self.name_to_id, 
                    self.id_to_name, 
                    self.include_classes, 
                    self.W, self.H, 
                    pix_T_camX=self.pix_T_camX, 
                    # origin_T_camX0=None, 
                    ddetr=self.ddetr, 
                    controller=controller, 
                    use_gt_objecttrack=False,
                    do_masks=True,
                    use_solq=True,
                    id_to_mapped_id=self.id_to_mapped_id,
                    on_aws = False, 
                    navigator=navigation,
                    )

                camX0_T_camX = navigation.explorer.get_camX0_T_camX()
                rgb, depth = self.get_obs(task, head_tilt=navigation.explorer.head_tilt)
                action_successful = True

                self.update_navigation_obs(rgb,depth, action_successful, navigation)

                camX0_T_camX = navigation.explorer.get_camX0_T_camX()
                camX0_T_camX0 = utils.geom.safe_inverse_single(camX0_T_camX)
                if vis is not None:
                    vis.camX0_T_camX0 = camX0_T_camX0
                    if object_tracker is not None:
                        vis.object_tracker = object_tracker

                # this is only used when GT is turned on
                # origin_T_camX0 = utils.aithor.get_origin_T_camX(controller.last_event, True)
                origin_T_camX0 = utils.aithor.get_origin_T_camX(controller.last_event, True)
                camX0_T_origin = utils.geom.safe_inverse_single(origin_T_camX0)
                object_tracker.camX0_T_origin = camX0_T_origin


            # get next action to goal from navigation policy     
            try:
                action, param = navigation.act()
            except:
                action = 'Pass'
            action_rearrange = self.nav_action_to_rearrange_action[action]
            action_ind = self.action_to_ind[action_rearrange]

            camX0_T_camX = navigation.explorer.get_camX0_T_camX()

            if step==0:
                if vis is not None:
                    vis.add_frame(controller.last_event.frame, text="Explore", add_map=False)
            else:
                if vis is not None:
                    vis.add_frame(controller.last_event.frame, text="Explore", add_map=True)
            
            if object_tracker is not None and action_successful and not args.get_pose_change_from_GT and not args.use_GT_centroids_from_meta:
                object_tracker.update(rgb, depth, camX0_T_camX, vis=vis)

            if self.verbose:
                print(action_rearrange, action_ind)

            if task.num_steps_taken() % 10 == 0:
                print(
                    f"{self.current_mode}: {phase} phase EXPLORATION (step {task.num_steps_taken()}):"
                    f" taking action {task.action_names()[action_ind]}"
                )
            if action=='Pass':
                exploring = False
                num_sampled += 1
                try:
                    if not inds_explore: # or phase=="WALKTHROUGH":
                        ind_i, ind_j = navigation.get_reachable_map_locations(sample=True)
                        if not ind_i:
                            break
                        inds_explore.append([ind_i, ind_j])
                    else:
                        ind_i, ind_j = inds_explore.pop(0)
                except:
                    break

                print(f"{self.current_mode}: going to {ind_i} {ind_j}")

                navigation.set_point_goal(ind_i, ind_j, search_mode=True)  
            else:
                task.step(action=action_ind)

                rgb, depth = self.get_obs(task, head_tilt=navigation.explorer.head_tilt)
                action_successful = self.success_checker.check_successful_action(rgb) 
                self.update_navigation_obs(rgb,depth, action_successful, navigation)
                # print("ACTUAL:", controller.last_event.metadata['agent']['cameraHorizon'])                

            step += 1

            if steps_max is not None: 
                if step >= steps_max:
                    break

            if num_sampled>50:
                break
                
        return object_tracker, inds_explore, camX0_T_origin

    def get_relations_pickupable(self, object_tracker):
        
        if args.use_GT_centroids_from_meta:
            tracker_centroids, tracker_labels = object_tracker.get_objects_gt_from_meta()
        else:
            tracker_centroids, tracker_labels = object_tracker.get_centroids_and_labels()

        obj_rels = {}
        for obj_i in range(len(tracker_centroids)):

            centroid = tracker_centroids[obj_i]
            obj_category_name = tracker_labels[obj_i]

            if args.do_open and obj_category_name in self.OPENABLE_OBJECTS:
                if obj_category_name not in obj_rels:
                    obj_rels[obj_category_name] = {}
                    obj_rels[obj_category_name]['relations'] = []
                    obj_rels[obj_category_name]['centroids'] = []
                obj_rels[obj_category_name]['relations'].append(None)
                obj_rels[obj_category_name]['centroids'].append(centroid)

            if obj_category_name not in self.rearrange_pickupable:
                continue

            dists = np.sqrt(np.sum((tracker_centroids - np.expand_dims(centroid, axis=0))**2, axis=1))

            # remove centroids directly overlapping
            dist_thresh = dists>0.05 #self.OT_dist_thresh
            tracker_centroids_ = tracker_centroids[dist_thresh]
            tracker_labels_ = list(np.array(tracker_labels)[dist_thresh])

            # keep only centroids of different labels to compare against
            keep = np.array(tracker_labels_)!=obj_category_name
            tracker_centroids_ = tracker_centroids_[keep]
            tracker_labels_ = list(np.array(tracker_labels_)[keep])

            keep = []
            for l in tracker_labels_:
                if l not in self.rearrange_pickupable:
                    keep.append(True)
                else:
                    keep.append(False)
            keep = np.array(keep)
            tracker_centroids_ = tracker_centroids_[keep]
            tracker_labels_ = list(np.array(tracker_labels_)[keep])

            # ignore floor for now
            relations = self.extract_relations_centroids(centroid, obj_category_name, tracker_centroids_, tracker_labels_, floor_height=-100)

            if obj_category_name not in obj_rels:
                obj_rels[obj_category_name] = {}
                obj_rels[obj_category_name]['relations'] = []
                obj_rels[obj_category_name]['centroids'] = []
            obj_rels[obj_category_name]['relations'].append(relations)
            obj_rels[obj_category_name]['centroids'].append(centroid)

            # print(relations)

        return obj_rels
        

    def get_out_of_place(self, walkthrough_obj_rels, unshuffle_obj_rels):
        
        out_of_place = {}
        object_dict = {}
        id_ = 0
        id2_ = 0

        # rearranged objects
        for key in list(walkthrough_obj_rels.keys()):
            if key in unshuffle_obj_rels:
                count_similar = 0
                count_dissimilar = 0

                # if args.do_open and (key in self.OPENABLE_OBJECTS):
                #     for i_l in range(len(unshuffle_obj_rels[key]['centroids'])):
                #         out_of_place[id_] = {}
                #         out_of_place[id_]['label'] = key
                #         out_of_place[id_]['walkthrough_state'] = None
                #         out_of_place[id_]['unshuffle_state'] = unshuffle_obj_rels[key]['centroids'][i_l]
                #         out_of_place[id_]['action'] = 'open'
                #         id_ += 1
                if key in self.OPENABLE_OBJECTS:
                    continue

                if args.loop_through_cat:
                    inds_loop = list(np.arange(min(
                        [len(unshuffle_obj_rels[key]['relations']), 
                        len(walkthrough_obj_rels[key]['relations']), 
                        2]
                        )))
                else:
                    inds_loop = [0]
                for i_l in inds_loop:
                    rels_u = unshuffle_obj_rels[key]['relations'][i_l]
                    if args.match_relations_walk:
                        rels_w = walkthrough_obj_rels[key]['relations'][i_l]
                    else:
                        rels_w = sum(walkthrough_obj_rels[key]['relations'], []) #walkthrough_obj_rels[wth][0]
                    
                    similar = []
                    different = []
                    for rel1 in rels_u:
                        found_one = False
                        for rel2 in rels_w:
                            if rel1==rel2:
                                count_similar += 1
                                found_one = True
                                similar.append(rel1)
                                break
                        if not found_one:
                            count_dissimilar += 1
                            different.append(rel1)
                    print("OBJECT", key)
                    print("Num similar=", count_similar, "Num dissimilar=", count_dissimilar)

                    # if count_similar == 0 and count_dissimilar > 2:
                    #     out_of_place[key] = {}
                    #     out_of_place[key]['walkthrough_state'] = walkthrough_obj_rels[key]['centroids'][0]
                    #     out_of_place[key]['unshuffle_state'] = unshuffle_obj_rels[key]['centroids'][0]
                    count_dissimilar += 1e-6
                    oop = count_similar/count_dissimilar < args.dissimilar_threshold and count_dissimilar > args.thresh_num_dissimilar

                    object_dict[id2_] = {}
                    object_dict[id2_]["similar"] = similar
                    object_dict[id2_]["different"] = different
                    object_dict[id2_]["label"] = key
                    object_dict[id2_]["out_of_place"] = oop
                    id2_ += 1

                    if oop:
                        out_of_place[id_] = {}
                        out_of_place[id_]['label'] = key
                        out_of_place[id_]['walkthrough_state'] = walkthrough_obj_rels[key]['centroids'][i_l]
                        out_of_place[id_]['unshuffle_state'] = unshuffle_obj_rels[key]['centroids'][i_l]
                        out_of_place[id_]['action'] = 'pickup'
                        id_ += 1

        # open and close
        for key in list(unshuffle_obj_rels.keys()):
            if args.do_open and (key in self.OPENABLE_OBJECTS):
                for i_l in range(len(unshuffle_obj_rels[key]['centroids'])):
                    out_of_place[id_] = {}
                    out_of_place[id_]['label'] = key
                    out_of_place[id_]['walkthrough_state'] = None
                    out_of_place[id_]['unshuffle_state'] = unshuffle_obj_rels[key]['centroids'][i_l]
                    out_of_place[id_]['action'] = 'open'
                    id_ += 1

        return out_of_place, object_dict

    # def eval_gt_and_relation(self, unshuffle_obj_gt, walkthrough_obj_gt, unshuffle_obj_rels, walkthrough_obj_rels):
    #     '''
    #     This is used to give a qualitative check on how well the relations are for inferring out of place
    #     '''
    #     print("OUT OF PLACE BEGIN")

    #     oop_objs = []
    #     for key in list(unshuffle_obj_gt.keys()):
    #         obj1 = unshuffle_obj_gt[key][0]
    #         obj2 = walkthrough_obj_gt[key][0]
    #         if obj1 is not None and obj2 is not None:
    #             for rec_i in range(len(obj1)):
    #                 if not (obj1[rec_i]==obj2[rec_i]):
    #                     oop_objs.append(unshuffle_obj_gt[key][1])
    #                     print('-------------------')
    #                     print(key, obj2, obj1)
    #                     print('walk_pos GT:', walkthrough_obj_gt[key][2], 'unsh_pos GT:', unshuffle_obj_gt[key][2])
    #                     # print("WALKTHROUGH:")
    #                     wth = walkthrough_obj_gt[key][1]
    #                     # if wth in walkthrough_obj_rels:
    #                     #     print(walkthrough_obj_rels[wth])
    #                     # else:
    #                     #     print(f"Nothing for {wth}")
    #                     # print("UNSHUFFLE:")
    #                     uth = unshuffle_obj_gt[key][1]
    #                     # if uth in unshuffle_obj_rels:
    #                     #     print(unshuffle_obj_rels[uth])
    #                     # else:
    #                     #     print(f"Nothing for {uth}")
    #                     print("OUT OF PLACE")
    #                     if wth in walkthrough_obj_rels and uth in unshuffle_obj_rels:
    #                         count_similar = 0
    #                         count_dissimilar = 0
    #                         rels_u = unshuffle_obj_rels[uth]['relations'][0]
    #                         rels_w = sum(walkthrough_obj_rels[wth]['relations'], []) #walkthrough_obj_rels[wth][0]
    #                         for rel1 in rels_u:
    #                             found_one = False
    #                             for rel2 in rels_w:
    #                                 if rel1==rel2:
    #                                     count_similar += 1
    #                                     found_one = True
    #                                     break
    #                             if not found_one:
    #                                 count_dissimilar += 1
    #                         print("OUT OF PLACE:","Num similar=", count_similar, "Num dissimilar=", count_dissimilar)
    #                         print('walk_pos:', walkthrough_obj_rels[wth]['centroids'], 'unsh_pos:', unshuffle_obj_rels[uth]['centroids'])
    #                     print('-------------------')
    #                     break
                                

    #     print("IN PLACE BEGIN")

    #     oop_objs = []
    #     for key in list(unshuffle_obj_gt.keys()):
    #         obj1 = unshuffle_obj_gt[key][0]
    #         obj2 = walkthrough_obj_gt[key][0]
    #         if obj1 is not None and obj2 is not None:
    #             same = True
    #             for rec_i in range(len(obj1)):
    #                 if not (obj1[rec_i]==obj2[rec_i]):
    #                     # oop_objs.append(unshuffle_obj_gt[key][1])
    #                     same = False
    #                     break
    #             if same:
    #                 print('-------------------')
    #                 print(key, obj2, obj1)
    #                 print('walk_pos GT:', walkthrough_obj_gt[key][2], 'unsh_pos GT:', unshuffle_obj_gt[key][2])
    #                 # print("WALKTHROUGH:")
    #                 wth = walkthrough_obj_gt[key][1]
    #                 # if wth in walkthrough_obj_rels:
    #                 #     print(walkthrough_obj_rels[wth])
    #                 # else:
    #                 #     print(f"Nothing for {wth}")
    #                 # print("UNSHUFFLE:")
    #                 uth = unshuffle_obj_gt[key][1]
    #                 # if uth in unshuffle_obj_rels:
    #                 #     print(unshuffle_obj_rels[uth])
    #                 # else:
    #                 #     print(f"Nothing for {uth}")
    #                 print("IN PLACE")
    #                 if wth in walkthrough_obj_rels and uth in unshuffle_obj_rels:
    #                     count_similar = 0
    #                     count_dissimilar = 0
    #                     rels_u = unshuffle_obj_rels[uth]['relations'][0]
    #                     rels_w = sum(walkthrough_obj_rels[wth]['relations'], []) #walkthrough_obj_rels[wth][0]
    #                     for rel1 in rels_u:
    #                         found_one = False
    #                         for rel2 in rels_w:
    #                             if rel1==rel2:
    #                                 count_similar += 1
    #                                 found_one = True
    #                                 break
    #                         if not found_one:
    #                             count_dissimilar += 1
    #                     print("IN PLACE:", "Num similar=", count_similar, "Num dissimilar=", count_dissimilar)
    #                     print('walk_pos:', walkthrough_obj_rels[wth]['centroids'], 'unsh_pos:', unshuffle_obj_rels[uth]['centroids'])
    #                 print('-------------------')
        
    def extract_relations_centroids(self, centroid_target, label_target, obj_centroids, obj_labels, floor_height, pos_translator=None, overhead_map=None, visualize_relations=False): 

        '''Extract relationships of interest from a list of objects'''

        obj_labels_np = np.array(obj_labels.copy())

        ################# Check Relationships #################
        # check pairwise relationships. this loop is order agnostic, since pairwise relationships are mostly invertible
        if visualize_relations:
            relations_dict = {}
            for relation in self.relations_executors_pairs:
                relations_dict[relation] = []
        relations = []
        for relation in self.relations_executors_pairs:
            relation_fun = self.relations_executors_pairs[relation]
            if relation=='closest-to' or relation=='farthest-to' or relation=='supported-by':
                if relation=='supported-by':
                    if label_target in self.receptacles:
                        continue
                    yes_recept = []
                    for obj_label_i in obj_labels:
                        if obj_label_i in self.receptacles:
                            yes_recept.append(True)
                        else:
                            yes_recept.append(False)
                    yes_recept = np.asarray(yes_recept).astype(bool)
                    obj_centroids_ = obj_centroids[yes_recept]
                    obj_labels_ = list(obj_labels_np[yes_recept])
                    relation_ind = relation_fun(centroid_target, obj_centroids_, ground_plane_h=floor_height)
                    if relation_ind==-2:
                        pass
                    elif relation_ind==-1:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name('Floor')))
                        if visualize_relations:
                            relations_dict[relation].append(centroid_target)
                    else:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name(obj_labels_[relation_ind])))
                        if visualize_relations:
                            relations_dict[relation].append(obj_centroids_[relation_ind])
    
                else:
                    relation_ind = relation_fun(centroid_target, obj_centroids)
                    if relation_ind==-2:
                        pass
                    elif relation_ind==-1:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name('Floor')))
                        if visualize_relations:
                            relations_dict[relation].append(centroid_target)
                    else:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name(obj_labels[relation_ind])))
                        if visualize_relations:
                            relations_dict[relation].append(obj_centroids[relation_ind])
            else:
                for i in range(len(obj_centroids)):

                    is_relation = relation_fun(centroid_target, obj_centroids[i])
                
                    if is_relation:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name(obj_labels[i])))
                        if visualize_relations:
                            relations_dict[relation].append(obj_centroids[i])

        if visualize_relations:
            colors_rels = {
            'next-to': (0, 255, 0),
            'supported-by': (0, 255, 0),
            'closest-to': (0, 255, 255)
            }
            img = overhead_map.copy()

            c_target = pos_translator(centroid_target)
            color = (255, 0, 0)
            thickness = 1
            cv2.circle(img, c_target[[1,0]], 7, color, thickness)
            radius = 5
            for relation in list(relations_dict.keys()):
                centers_relation = relations_dict[relation]
                color = colors_rels[relation]
                for c_i in range(len(centers_relation)):
                    center_r = centers_relation[c_i]
                    c_rel_im = pos_translator(center_r)
                    cv2.circle(img, c_rel_im[[1,0]], radius, color, thickness)

            plt.figure(figsize=(8,8))
            plt.imshow(img)
            plt.savefig('images/test.png')
            st()

        return relations


# class Pose_Tracker():

#     def __init__(self, position, rotation, horizon):
#         # initial pose in aithor coords
#         self.init_position = position
#         self.init_rotation = rotation
#         self.init_horizon = horizon

#     def update_pose(self, mapper_pos, mapper_rot, mapper_hor):
#         # alter aithor coords by mapper estimnated pose
#         # print(self.init_rotation)
#         if np.abs(self.init_rotation['y'])<5:
#             self.position = {'x':self.init_position['x']+mapper_pos['x'], 'y':self.init_position['y'], 'z':self.init_position['z']+mapper_pos['z']}
#         elif np.abs(self.init_rotation['y'])<95.0:
#             self.position = {'x':self.init_position['x']+mapper_pos['z'], 'y':self.init_position['y'], 'z':self.init_position['z']-mapper_pos['x']}
#         elif np.abs(self.init_rotation['y'])<185.0:
#             self.position = {'x':self.init_position['x']-mapper_pos['x'], 'y':self.init_position['y'], 'z':self.init_position['z']-mapper_pos['z']}
#         elif np.abs(self.init_rotation['y'])<275.0:
#             self.position = {'x':self.init_position['x']-mapper_pos['z'], 'y':self.init_position['y'], 'z':self.init_position['z']+mapper_pos['x']}
#         else:
#             self.position = {'x':self.init_position['x']-mapper_pos['z'], 'y':self.init_position['y'], 'z':self.init_position['z']+mapper_pos['x']}
#             print("THIS INITIAL DOESNT BELONG?")
#             # assert(False)
#         self.rotation = {'x':self.init_rotation['x'], 'y':self.init_rotation['y']+mapper_rot, 'z':self.init_rotation['z']}
#         self.horizon = self.init_horizon + mapper_hor
#         self.rotation['y'] %= 360

#     def get_pose(self):
#         pose = {}
#         pose["rotation"] = self.rotation
#         pose["position"] = self.position
#         pose["horizon"] = self.horizon
#         return pose

            


            

