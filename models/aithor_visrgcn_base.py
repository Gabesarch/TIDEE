

from arguments import args
from argparse import Namespace
import numpy as np
import random
import os
import math
from tensorboardX import SummaryWriter
# from ai2thor_docker.ai2thor_docker.x_server import startx
import ipdb
import pickle
import torch
import cv2
import utils.aithor
import utils.wctb
from PIL import Image
st = ipdb.set_trace
from task_base.aithor_base import Base
from utils.wctb import Utils, Relations
from utils.wctb import ThorPositionTo2DFrameTranslator
import utils.box
import glob
import sys
from utils.aithor import get_amodal2d, get_3dbox_in_geom_format, get_origin_T_camX
from SOLQ.util.misc import nested_tensor_from_tensor_list
from SOLQ.util import box_ops
import torch.nn.functional as F
import torchvision.ops

class Ai2Thor_Base(Base):

    def __init__(self):   

        super(Ai2Thor_Base, self).__init__()

        # self.relations = Memory(None)
        # self.relations_executors_pairs = self.relations.relations_executors_pairs
        self.include_classes_floor = self.include_classes + ['Floor']
        self.name_to_id_floor = self.name_to_id
        self.name_to_id_floor['Floor'] = len(self.include_classes_floor) - 1
        self.id_to_name_floor = self.id_to_name
        self.id_to_name_floor[len(self.include_classes_floor) - 1] = 'Floor' 

        self.superstructures = [['Chair', 'DiningTable'], ['Television', 'ArmChair', 'TVStand', 'Sofa', 'CoffeeTable', 'Ottoman']]

        self.facing_classes = [
            'Toilet', 'Laptop', 'Chair', 'Desk', 'Television',
            'ArmChair', 'Sofa', 
            'Microwave', 'CoffeeMachine', 'Fridge', 
            'Toaster', 
            ]

        ############## Set up relation utils ##############
        self.utils = Utils(args.H, args.W)
        self.relations_util = Relations(args.H, args.W)
        self.relations_executors_pairs = {
            'above': self.relations_util._is_above,
            # 'below': self.relations_util._is_below,
            'next-to': self.relations_util._is_next_to,
            'supported-by': self.relations_util._is_supported_by,
            'aligned-with': self.relations_util._is_aligned,
            'facing': self.relations_util._is_facing,
            # 'equal-height': self.relations_util._is_equal_height,
        }

        self.rel_to_id = {list(self.relations_executors_pairs.keys())[i]:i for i in range(len(self.relations_executors_pairs))}

        '''
        self.relations_executors_pairs_reciprocal = {
            'above': 'below',
            'equal-height': 'equal-height'
        }
        '''
        self.relations_executors_triples = {
            'between': self.relations_util._is_between,
            'similar-dist-to': self.relations_util._is_similar_dist_to,
        }

    def get_memex_relations(self):

        objects = self.controller.last_event.metadata['objects']
        objects_valid = []
        objects_mapping = {}
        for i in range(len(objects)):
            if objects[i]['objectType'] in self.name_to_id:
                objects_valid.append(objects[i])
                objects_mapping[objects[i]['objectId']] = {'name':objects[i]['name'], 'type':objects[i]['objectType']}
        objects_mapping['Floor'] = {'name':'Floor', 'type':'Floor'}

        self.controller.step({"action": "ToggleMapView"})
        cam_position = self.controller.last_event.metadata["cameraPosition"]
        cam_orth_size = self.controller.last_event.metadata["cameraOrthSize"]
        pos_translator = ThorPositionTo2DFrameTranslator(
            self.controller.last_event.frame.shape, self.utils.position_to_tuple(cam_position), cam_orth_size
        )
        overhead_map = self.controller.last_event.frame
        mask_frame_overhead = self.controller.last_event.instance_masks
        seg_frame_overhead = self.controller.last_event.instance_segmentation_frame
        depth_overhead = self.controller.last_event.depth_frame
        self.controller.step({"action": "ToggleMapView"})
        _, relations_text, _, object_list_names, object_list_cats  = self.extract_relations(self.controller, objects_valid, depth_overhead, mask_frame_overhead, pos_translator=pos_translator, overhead_map=overhead_map)
        # print(object_list_names)

        # add receptacle relations
        relation = 'supported-by'
        for obj in objects_valid:
            receptacle_id = obj['parentReceptacles']
            if receptacle_id is None:
                continue
            # print(receptacle_id)
            receptacle_id = receptacle_id[0]
            if 'Floor' in receptacle_id:
                receptacle_id = 'Floor'
            receptacle_name = objects_mapping[receptacle_id]['name']
            receptacle_catname = objects_mapping[receptacle_id]['type']
            target_name = obj['name']
            target_catname = obj['objectType']
            rel_text = "The {0} is {1} the {2} \n".format(target_name, relation, receptacle_name)
            if rel_text not in relations_text:
                relations_text.append(rel_text)
                if target_name not in object_list_names:
                    object_list_names.append(target_name)
                    object_list_cats.append(target_catname)
                if receptacle_name not in object_list_names:
                    object_list_names.append(receptacle_name)
                    object_list_cats.append(receptacle_catname)

        nav_pts = self.controller.last_event.metadata["actionReturn"]
        nav_pts_orig = np.array([list(d.values()) for d in nav_pts])
        object_name_to_feat = {}
        for obj in objects_valid:
            if obj['name'] not in object_list_names:
                continue
            obj_center = np.array(list(obj['axisAlignedBoundingBox']['center'].values()))

            self.controller.step(action="GetReachablePositions")
            reachable = self.controller.last_event.metadata["actionReturn"]
            reachable = np.array([list(d.values()) for d in reachable])
            distances = np.sqrt(np.sum((reachable - np.expand_dims(obj_center, axis=0))**2, axis=1))
            map_pos = obj_center

            within_radius = np.logical_and(distances > args.radius_min, distances < args.radius_max)
            valid_pts = reachable[within_radius]
            
            obj_bbox = None
            for _ in range(args.num_tries_memory):
                found_one_ = self.find_starting_viewpoint(valid_pts, obj_center)

                if not found_one_:
                    continue

                instance_detections2d = self.controller.last_event.instance_detections2D

                if obj['objectId'] not in instance_detections2d:
                    continue

                obj_bbox = np.array(instance_detections2d[obj['objectId']])
                break

            with torch.no_grad():
                rgb = self.controller.last_event.frame # (H, W, 3)
                rgbs = (torch.from_numpy(rgb.copy()).float() / 255.0).cuda().permute(2,0,1).unsqueeze(0) # (1, 3, H, W)
                feature_map = self.ddetr.model.backbone(nested_tensor_from_tensor_list(rgbs))[0][args.backbone_layer_ddetr].decompose()[0]
            
            if obj_bbox is None:
                # print("Could not find good viewpoint for memory object.. skipping feature extraction..")
                pooled_obj_feat = torch.ones([1, self.visual_feat_size]).cuda()
            else:
                feature_crop = torchvision.ops.roi_align(feature_map, [torch.from_numpy(obj_bbox / 8).float().unsqueeze(0).cuda()], output_size=(32,32))
                pooled_obj_feat = F.adaptive_avg_pool2d(feature_crop, (1,1)).squeeze(-1).squeeze(-1)

            object_name_to_feat[obj['name']] = pooled_obj_feat

        object_ids = [self.name_to_id[name] for name in object_list_cats]
        object_ids = torch.tensor(object_ids, dtype=torch.long)

        relations_text_new = []
        for triplet in relations_text:
            if triplet=='' or triplet in [str(i) for i in list(np.arange(20))]:
                continue
            split_ = triplet.split()
            if split_[1] not in object_list_names:
                continue
            if split_[5] not in object_list_names:
                continue
            relations_text_new.append(triplet)
                
        subj_pred_obj_adj_scene, preds_super  = self.get_adj_mats_from_rels(relations_text_new, object_list_names, object_list_cats)

        all_zero_feat = torch.zeros_like(object_name_to_feat[next(iter(object_name_to_feat))]).cuda()
        object_feats = [object_name_to_feat[name] if name in object_name_to_feat else all_zero_feat for name in object_list_names ]

        object_feats = torch.cat(object_feats, dim=0)

        return subj_pred_obj_adj_scene, object_ids, object_feats

    def find_starting_viewpoint(self, valid_pts, obj_center, num_tries=10):
        '''
        Finds a "good" starting viewpoint based on object center in Aithor 3D coordinates and valid spawn points
        '''
        found_one = False
        if valid_pts.shape[0]==0:
            return found_one
        for i in range(num_tries): # try 10 times to find a good location
            s_ind = np.random.randint(valid_pts.shape[0])
            pos_s = valid_pts[s_ind]

            # add height from center of agent to camera
            pos_s[1] = pos_s[1] + 0.675

            # YAW calculation - rotate to object
            agent_to_obj = np.squeeze(obj_center) - pos_s 
            agent_local_forward = np.array([0, 0, 1.0]) 
            flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
            flat_dist_to_obj = np.linalg.norm(flat_to_obj) + 1e-6
            flat_to_obj /= flat_dist_to_obj

            det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
            turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

            # add noise so not right in the center
            noise = np.array([0.0, 0.0]) #np.random.normal(0, 2, size=2)

            turn_yaw = np.degrees(turn_angle) + noise[0]
            turn_pitch = -np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj)) + noise[1]

            turn_yaw = np.round(turn_yaw/args.DT) * args.DT

            # just add a little noise to the pitch
            rand_pick = np.random.uniform(0,1)
            if rand_pick < 0.1:
                turn_pitch = np.floor(turn_pitch/args.HORIZON_DT) * args.HORIZON_DT
            elif rand_pick > 0.9:
                turn_pitch = np.ceil(turn_pitch/args.HORIZON_DT) * args.HORIZON_DT
            else:
                turn_pitch = np.round(turn_pitch/args.HORIZON_DT) * args.HORIZON_DT

            # move agent and rotate to object
            event = self.controller.step('TeleportFull', x=pos_s[0], y=pos_s[1] - 0.675, z=pos_s[2], rotation=dict(x=0.0, y=int(turn_yaw), z=0.0), horizon=int(turn_pitch), standing=True)

            # print("TURN PITCH:", turn_pitch)

            found_one = True
            break

        return found_one

    def run_agent(self, mapname=None):

        if args.do_class_labels:
            outputs = self.run_agent_class(mapname)
        elif args.do_features:
            outputs = self.run_agent_features()
        else:
            assert(False)
        return outputs

    def get_oop_nodes(self, mapname=None):
        
        out_dict = {}

        obj_ids_scene_batch, subj_pred_obj_adj_scene_batch, obj_feats_scene_batch = [], [], []            
        
        if args.do_load_oop_nodes_and_supervision:
            base_path = os.path.join(args.vmemex_supervision_dir, mapname)
            objects_valid = glob.glob(base_path+"/*.p")
        else:
            objects = self.controller.last_event.metadata['objects']
            objects_valid = []
            for i in range(len(objects)):
                if objects[i]['objectType'] in self.name_to_id:
                    objects_valid.append(objects[i])


        # loop through objects
        successes = 0
        while True:

            if args.do_load_oop_nodes_and_supervision:
                rand_choice = np.random.randint(len(objects_valid))
                load_path = objects_valid[rand_choice]   
            else:
                rand_choice = np.random.randint(len(objects_valid))
                obj_meta_select = objects_valid[rand_choice]         
                
                obj_category_name = obj_meta_select['objectType']
                pickupable = obj_meta_select["pickupable"]

                if (obj_category_name not in self.name_to_id) or not pickupable:
                    continue

                print("object selected is:", obj_category_name)

            if args.do_load_oop_nodes_and_supervision:                
                print(load_path.split('/')[-1])
                with open(load_path, 'rb') as handle:
                    load_dict = pickle.load(handle)
                subj_pred_obj_adj_scene = load_dict['subj_pred_obj_adj_scene']
                object_ids = load_dict['object_ids']
                object_feats = torch.cat(load_dict['object_feats'], dim=0)
            else:
                assert(False) # save out supervision first with save_oop_nodes_and_supervision()

            # Add to batch
            subj_pred_obj_adj_scene_batch.append(subj_pred_obj_adj_scene)
            obj_ids_scene_batch.append(object_ids)
            obj_feats_scene_batch.append(object_feats)

            successes += 1

            if successes==args.objects_per_scene:
                break

        return subj_pred_obj_adj_scene_batch, obj_ids_scene_batch, obj_feats_scene_batch

    def save_oop_nodes_and_supervision(self, mapname):
        '''
        This saves out supervision for training houses for each object. Saves visual features, ID, and relation adjacency matrix

        Here we take the default location of each object to be the in place location to supervise the RGCN
        '''

        out_dict = {}

        obj_ids_scene_batch, subj_pred_obj_adj_scene_batch = [], []

        objects = self.controller.last_event.metadata['objects']
        objects_valid = []
        objects_mapping = {}
        for i in range(len(objects)):
            if objects[i]['objectType'] in self.name_to_id:
                objects_valid.append(objects[i])
                objects_mapping[objects[i]['objectId']] = {'name':objects[i]['name'], 'type':objects[i]['objectType']}
        objects_mapping['Floor'] = {'name':'Floor', 'type':'Floor'}

        self.controller.step(action="GetReachablePositions")
        nav_pts = self.controller.last_event.metadata["actionReturn"]
        nav_pts_orig = np.array([list(d.values()) for d in nav_pts])
        object_name_to_feat = {}
        obj_index = 0
        objects_valid2 = []
        for obj in objects_valid:

            obj_index += 1
            print(obj_index,'/',len(objects_valid))
            obj_center = np.array(list(obj['axisAlignedBoundingBox']['center'].values()))

            self.controller.step(action="GetReachablePositions")
            reachable = self.controller.last_event.metadata["actionReturn"]
            reachable = np.array([list(d.values()) for d in reachable])
            distances = np.sqrt(np.sum((reachable - np.expand_dims(obj_center, axis=0))**2, axis=1))
            map_pos = obj_center

            within_radius = np.logical_and(distances > args.radius_min, distances < args.radius_max)
            valid_pts = reachable[within_radius]

            obj_bbox = None
            for _ in range(args.num_tries_memory):
                found_one_ = self.find_starting_viewpoint(valid_pts, obj_center)

                if not found_one_:
                    continue

                instance_detections2d = self.controller.last_event.instance_detections2D

                if obj['objectId'] not in instance_detections2d:
                    continue
            
                obj_bbox = np.array(instance_detections2d[obj['objectId']])
                break

            with torch.no_grad():
                rgb = self.controller.last_event.frame # (H, W, 3)
                rgbs = (torch.from_numpy(rgb.copy()).float() / 255.0).cuda().permute(2,0,1).unsqueeze(0) # (1, 3, H, W)
                feature_map = self.ddetr.model.backbone(nested_tensor_from_tensor_list(rgbs))[0][args.backbone_layer_ddetr].decompose()[0]
            
            if obj_bbox is None:
                # print("Could not find good viewpoint for memory object.. skipping feature extraction..")
                pooled_obj_feat = torch.ones([1, self.visual_feat_size]).cuda()
            else:
                feature_crop = torchvision.ops.roi_align(feature_map, [torch.from_numpy(obj_bbox / 8).float().unsqueeze(0).cuda()], output_size=(32,32))
                pooled_obj_feat = F.adaptive_avg_pool2d(feature_crop, (1,1)).squeeze(-1).squeeze(-1)

            object_name_to_feat[obj['name']] = pooled_obj_feat
            objects_valid2.append(obj)

        # loop through objects
        successes = 0
        ind = 0
        for obj_meta_select in objects_valid2:
            
            obj_category_name = obj_meta_select['objectType']
            pickupable = obj_meta_select["pickupable"]

            if (obj_category_name not in self.name_to_id) or not pickupable:
                continue

            print("object selected is:", obj_category_name)

            base_path = os.path.join(args.vmemex_supervision_dir, mapname)
            if not os.path.exists(base_path):
                os.mkdir(base_path)
            filename = obj_meta_select['name'] + '.p'
            save_path = os.path.join(base_path, filename)

            if os.path.exists(save_path):
                continue

            if args.only_include_receptacle:

                relation = 'supported-by'
                receptacle_id = obj_meta_select['parentReceptacles']
                if receptacle_id is None:
                    continue
                receptacle_id = receptacle_id[0]
                if 'Floor' in receptacle_id:
                    receptacle_id = 'Floor'
                receptacle_name = objects_mapping[receptacle_id]['name']
                receptacle_catname = objects_mapping[receptacle_id]['type']
                target_name = obj_meta_select['name']
                target_catname = obj_meta_select['objectType']

                relations_text = []
                relations_text.append("The {0} is {1} the {2} \n".format(target_name, relation, receptacle_name))
                object_list_names = []
                object_list_names.extend([target_name, receptacle_name])
                object_list_cats = []
                object_list_cats.extend([target_catname, receptacle_catname])

                # print(relations_text)
                
                subj_pred_obj_adj_scene, preds_super  = self.get_adj_mats_from_rels(relations_text, object_list_names, object_list_cats)

                object_ids = [self.name_to_id[name] for name in object_list_cats]
                object_ids = torch.tensor(object_ids, dtype=torch.long)

                all_zero_feat = torch.zeros_like(object_name_to_feat[next(iter(object_name_to_feat))]).cuda()
                object_feats = [object_name_to_feat[name] if name in object_name_to_feat else all_zero_feat for name in object_list_names ]

                print("SAVING TO", save_path)

                out_dict = {'subj_pred_obj_adj_scene':subj_pred_obj_adj_scene, 'object_ids':object_ids, 'object_feats': object_feats}
                # yield out_dict
                with open(save_path, 'wb') as handle:
                    pickle.dump(out_dict, handle, protocol=4)

            else: # include all relations and classes
                
                # get_gt_relations(self.controller, obj_meta=obj_meta_select)
                self.controller.step({"action": "ToggleMapView"})
                cam_position = self.controller.last_event.metadata["cameraPosition"]
                cam_orth_size = self.controller.last_event.metadata["cameraOrthSize"]
                pos_translator = ThorPositionTo2DFrameTranslator(
                    self.controller.last_event.frame.shape, self.utils.position_to_tuple(cam_position), cam_orth_size
                )
                overhead_map = self.controller.last_event.frame
                mask_frame_overhead = self.controller.last_event.instance_masks
                seg_frame_overhead = self.controller.last_event.instance_segmentation_frame
                depth_overhead = self.controller.last_event.depth_frame
                self.controller.step({"action": "ToggleMapView"})
                _, relations_text, _, object_list_names, object_list_cats = self.extract_relations(self.controller, objects_valid2, depth_overhead, mask_frame_overhead, target_object=obj_meta_select, pos_translator=pos_translator, overhead_map=overhead_map)
                object_ids = [self.name_to_id[name] for name in object_list_cats]
                object_ids = torch.tensor(object_ids, dtype=torch.long)

                if object_ids.shape[0]==0:
                    continue

                # num_walls = [1 if name=='Wall' else 0 for name in object_list_cats]
                # num_walls = sum(num_walls)
                # # for now just have whole image as wall image
                # if num_walls>0:
                #     rgb_to_append = rgb_torch.unsqueeze(0).repeat(num_walls,1,1,1)
                #     rgb_crops = torch.cat([rgb_crops, rgb_to_append], dim=0)
                
                subj_pred_obj_adj_scene, preds_super  = self.get_adj_mats_from_rels(relations_text, object_list_names, object_list_cats)
                # pred_ids = [self.rel_to_id[pred] for pred in preds_super]
                # pred_ids = torch.tensor(pred_ids, dtype=torch.long)

                all_zero_feat = torch.zeros_like(object_name_to_feat[next(iter(object_name_to_feat))]).cuda()
                object_feats = [object_name_to_feat[name] if name in object_name_to_feat else all_zero_feat for name in object_list_names ]

                out_dict = {'subj_pred_obj_adj_scene':subj_pred_obj_adj_scene, 'object_ids':object_ids, 'object_feats': object_feats}
                # yield out_dict
                with open(save_path, 'wb') as handle:
                    pickle.dump(out_dict, handle, protocol=4)

    def extract_relations(self, controller, objects, depth_overhead, mask_frame_overhead, target_object=None, pos_translator=None, overhead_map=None, memory=False): 


        '''Extract relationships of interest from a list of objects'''
        do_visualize=False
        ################# Obtain Object List #################
        # preprocess object list
        all_obj_xmin = 1000
        all_obj_xmax = -1000
        all_obj_zmin = 1000
        all_obj_zmax = -1000
        processed_objects = []
        superstructures = self.superstructures.copy()
        object_list = []
        object_list_cats = []
        for obj in objects:
            if obj['objectType'] not in self.name_to_id:
                continue
            processed_obj = {}
            processed_obj['name'] = obj['name']
            processed_obj['objectId'] = obj['objectId']
            processed_obj['class_name'] = obj['objectType']
            processed_obj['meta'] = obj
            bbox = np.array(obj['axisAlignedBoundingBox']['cornerPoints'])
            xmin, xmax = np.min(bbox[:,0]), np.max(bbox[:,0])
            ymin, ymax = np.min(bbox[:,1]), np.max(bbox[:,1])
            zmin, zmax = np.min(bbox[:,2]), np.max(bbox[:,2])
            processed_obj['bbox'] = np.array([xmin, zmin, ymin, xmax, zmax, ymax])

            all_obj_xmin = min(all_obj_xmin, xmin)
            all_obj_xmax = max(all_obj_xmax, xmax)
            all_obj_zmin = min(all_obj_zmin, zmin)
            all_obj_zmax = max(all_obj_zmax, zmax)

            if obj['objectId'] in mask_frame_overhead:
                mask_binary = mask_frame_overhead[obj['objectId']]

                depth_masked = depth_overhead[mask_binary]

                # for aligned
                mask_example = np.float32(mask_binary)
                mask_example[mask_example==1] = 255.
                mask_example = np.uint8(mask_example)
                gray_BGR = cv2.cvtColor(mask_example, cv2.COLOR_GRAY2BGR)
                contours,hierarchy = cv2.findContours(mask_example, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                cnt = contours[0]
                rows,cols = gray_BGR.shape[:2]
                [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
                # slope = vy / vx
                processed_obj['orientation'] = np.array([vx[0], vy[0], x[0], y[0]])

                # for facing
                wmin, hmin = pos_translator(processed_obj['bbox'][:2])
                wmax, hmax = pos_translator(processed_obj['bbox'][3:5])
                processed_obj['overhead_box'] = np.array([hmin, wmin, hmax, wmax])
                if obj['objectType'] in self.facing_classes:
                    facing_orientation = self.relations_util.get_facing_dir(depth_masked, mask_binary, processed_obj['orientation'])
                    processed_obj['facing_dir'] = facing_orientation
                else:
                    processed_obj['facing_dir'] = None
            else:
                processed_obj['orientation'] = None
                processed_obj['overhead_box'] = None
                processed_obj['facing_dir'] = None

            # if obj['objectType'] == "Floor":
            #     wmin, hmin = pos_translator(processed_obj['bbox'][:2])
            #     wmax, hmax = pos_translator(processed_obj['bbox'][3:5])
            #     cur_map = utils.improc.draw_box_single(overhead_map, [wmin, hmin, wmax, hmax], np.array([1,0,0]) * 255.0)
            #     plt.figure(figsize=(8,8))
            #     plt.imshow(cur_map)
            #     plt.show()
            #     return 

            # if obj['pickupable']:
            #     continue
            
            super_check = []
            for i in range(len(superstructures)):
                super_check.extend(superstructures[i])
            if obj['objectType'] not in super_check:
                superstructures.append([obj['objectType']])

            # object_list.append(obj['name'])
            # object_list_cats.append(obj['objectType'])

            processed_objects.append(processed_obj)

        # add walls to object lists
        # construct wall bounding boxes
        reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
        reachable_pos = np.array([[pos['x'], pos['z']] for pos in reachable_positions])
        xmin, zmin = np.min(reachable_pos, axis=0)
        xmax, zmax = np.max(reachable_pos, axis=0)
        xmin = min(xmin - 0.3, all_obj_xmin)
        xmax = max(xmax + 0.3, all_obj_xmax)
        zmin = min(zmin - 0.3, all_obj_zmin)
        zmax = max(zmax + 0.3, all_obj_zmax)
        xs = [xmin, xmax]
        zs = [zmin, zmax]
        processed_obj = {'name': 'Wall_1', 'class_name': 'Wall', 'bbox': np.array([xmin-0.05, zmin, 0, xmin+0.05, zmax, 1])}
        processed_obj['orientation'] = np.array([0., 1., 0., 0.])
        wmin, hmin = pos_translator(processed_obj['bbox'][:2])
        wmax, hmax = pos_translator(processed_obj['bbox'][3:5])
        processed_obj['overhead_box'] = np.array([hmin, wmin, hmax, wmax])
        processed_objects.append(processed_obj)

        processed_obj = {'name': 'Wall_2', 'class_name': 'Wall', 'bbox': np.array([xmax-0.05, zmin, 0, xmax+0.05, zmax, 1])}
        processed_obj['orientation'] = np.array([0., 1., 0., 0.])
        wmin, hmin = pos_translator(processed_obj['bbox'][:2])
        wmax, hmax = pos_translator(processed_obj['bbox'][3:5])
        processed_obj['overhead_box'] = np.array([hmin, wmin, hmax, wmax])
        processed_objects.append(processed_obj)

        processed_obj = {'name': 'Wall_3', 'class_name': 'Wall', 'bbox': np.array([xmin, zmin-0.05, 0, xmax, zmin+0.05, 1])}
        processed_obj['orientation'] = np.array([1., 0., 0., 0.])
        wmin, hmin = pos_translator(processed_obj['bbox'][:2])
        wmax, hmax = pos_translator(processed_obj['bbox'][3:5])
        processed_obj['overhead_box'] = np.array([hmin, wmin, hmax, wmax])
        processed_objects.append(processed_obj)

        processed_obj = {'name': 'Wall_4', 'class_name': 'Wall', 'bbox': np.array([xmin, zmax-0.05, 0, xmax, zmax+0.05, 1])}
        processed_obj['orientation'] = np.array([1., 0., 0., 0.])
        processed_objects.append(processed_obj)
        wmin, hmin = pos_translator(processed_obj['bbox'][:2])
        wmax, hmax = pos_translator(processed_obj['bbox'][3:5])
        processed_obj['overhead_box'] = np.array([hmin, wmin, hmax, wmax])

        ################# Check Relationships #################
        # check pairwise relationships. this loop is order agnostic, since pairwise relationships are mostly invertible
        relation_pairs = []
        for i in range(len(processed_objects)):
            if processed_objects[i]['class_name'] not in self.include_classes or processed_objects[i]['class_name'] == "Wall":
                continue
            for j in range(len(processed_objects)):
                if i == j:
                    continue
                if processed_objects[i]['class_name'] == processed_objects[j]['class_name'] and processed_objects[i]['class_name'] == 'Wall':
                    continue
                if processed_objects[j]['class_name'] not in self.include_classes:
                    continue
                # if processed_objects[i]['class_name']=="Laptop" or processed_objects[j]['class_name']=="Laptop":
                #     st()
                if target_object is not None:
                    if processed_objects[i]['objectId'] is not target_object['objectId']: # and processed_objects[j]['objectId'] is not target_object['objectId']:
                        continue
                for relation in self.relations_executors_pairs:
                    relation_fun = self.relations_executors_pairs[relation]
                    if relation == "next-to":
                        is_relation = relation_fun(processed_objects[i]['bbox'], processed_objects[j]['bbox'], ref_is_wall = processed_objects[j]['class_name']=='Wall')
                    elif relation == "aligned-with":
                        if processed_objects[j]['class_name'] == "Floor" or processed_objects[i]['class_name'] == "Floor" or processed_objects[j]['class_name']=='Wall':
                            continue
                        is_relation = relation_fun(processed_objects[i]['orientation'], processed_objects[j]['orientation'], processed_objects[i]['bbox'], processed_objects[j]['bbox'])
                    elif relation == "facing":
                        if processed_objects[j]['class_name'] == "Floor" or processed_objects[i]['class_name'] == "Floor" or processed_objects[j]['class_name']=='Wall':
                            continue
                        # if processed_objects[i]['class_name'] == "HousePlant":
                        #     st()
                        is_relation = relation_fun(processed_objects[j]['overhead_box'], processed_objects[i]['facing_dir'], processed_objects[i]['overhead_box'], processed_objects[i]['bbox'], processed_objects[j]['bbox'])
                    else:
                        is_relation = relation_fun(processed_objects[i]['bbox'], processed_objects[j]['bbox'])
                        
                    if is_relation:

                        if processed_objects[i]['name'] not in object_list:
                            object_list.append(processed_objects[i]['name'])
                            object_list_cats.append(processed_objects[i]['class_name'])

                        if processed_objects[j]['name'] not in object_list:
                            object_list.append(processed_objects[j]['name'])
                            object_list_cats.append(processed_objects[j]['class_name'])

                        # print("The {0} is {1} the {2}".format(processed_objects[i]['class_name'], relation, processed_objects[j]['class_name']))
                        relation_pairs.append([i, j, relation])
                        # relation_pairs.append([j, i, self.relations_executors_pairs_reciprocal[relation]])
                        # if overhead_map is not None and do_visualize:
                        #     wmin, hmin = pos_translator(processed_objects[i]['bbox'][:2])
                        #     wmax, hmax = pos_translator(processed_objects[i]['bbox'][3:5])
                        #     cur_map = utils.improc.draw_box_single(overhead_map, [wmin, hmin, wmax, hmax], np.array([1,0,0]) * 255.0)

                        #     wmin, hmin = pos_translator(processed_objects[j]['bbox'][:2])
                        #     wmax, hmax = pos_translator(processed_objects[j]['bbox'][3:5])
                        #     cur_map = utils.improc.draw_box_single(cur_map, [wmin, hmin, wmax, hmax], np.array([0,0,1]) * 255.0)

                        #     plt.figure(figsize=(8,8))
                        #     plt.imshow(cur_map)
                        #     plt.show()

        # 
        plotted = []
        cur_map = np.copy(overhead_map)
        relations_text = []
        # obj_names = []
        relations_groups_text = {i:[] for i in range(len(superstructures))}
        superstruct_mapping = {}
        for i in range(len(superstructures)):
            super_names = superstructures[i]
            for n in super_names:
                superstruct_mapping[n] = i
        # new_ind = 0
        for relation_pair in relation_pairs:
            i, j, relation = relation_pair
        #     if relation == "next-to":
        #         wmin, hmin = pos_translator(processed_objects[i]['bbox'][:2])
        #         wmax, hmax = pos_translator(processed_objects[i]['bbox'][3:5])
        #         w_i = (wmin + wmax) // 2
        #         h_i = (hmin + hmax) // 2
        #         if i not in plotted:
        #             plotted.append(i)
        #             cur_map = utils.improc.draw_box_single(cur_map, [wmin, hmin, wmax, hmax], np.array([0,0,1]) * 255.0)
        #         wmin, hmin = pos_translator(processed_objects[j]['bbox'][:2])
        #         wmax, hmax = pos_translator(processed_objects[j]['bbox'][3:5])
        #         w_j = (wmin + wmax) // 2
        #         h_j = (hmin + hmax) // 2
        #         if j not in plotted:
        #             plotted.append(j)
        #             cur_map = utils.improc.draw_box_single(cur_map, [wmin, hmin, wmax, hmax], np.array([0,0,1]) * 255.0)
        #         cur_map = utils.improc.draw_line_single(cur_map, [w_i, h_i, w_j, h_j], np.array([1,0,0])*255.0)

            # prune some relations
            if memory:
                remove = False
                for super_s in self.superstructures:
                    if processed_objects[i]['class_name'] in super_s and processed_objects[j]['class_name'] not in super_s:
                        remove = True
                if remove:
                    continue

            #print("The {0} is {1} the {2}".format(processed_objects[i]['name'], relation, processed_objects[j]['name']))
            # print("The {0} is {1} the {2}".format(processed_objects[i]['class_name'], relation, processed_objects[j]['class_name']))
            relations_text.append("The {0} is {1} the {2} \n".format(processed_objects[i]['name'], relation, processed_objects[j]['name']))
            # obj_names.append(processed_objects[i]['name'])
            # obj_names.append(processed_objects[j]['name'])

            ind_superstruct = superstruct_mapping[processed_objects[i]['class_name']]
            relations_groups_text[ind_superstruct].append("The {0} is {1} the {2} \n".format(processed_objects[i]['name'], relation, processed_objects[j]['name']))

        return cur_map, relations_text, relations_groups_text, object_list, object_list_cats

    def get_adj_mats_from_rels(self, relations_text, obj_names, obj_types_super):

        subjs_super = []
        preds_super = []
        objs_super = []
        obj_names_super = []
        obj_types_super = []
        num_cats = len(list(self.name_to_id.values()))
        num_relations = len(self.relations_executors_pairs)
        for triplet in relations_text:
            if triplet=='' or triplet in [str(i) for i in list(np.arange(20))]:
                continue
            split_ = triplet.split()
            subjs_super.append(split_[1])
            preds_super.append(split_[3])
            objs_super.append(split_[5])
        # obj_names = object_name_list
        # obj_names_super = obj_names #list(set(subjs_super))
        # obj_types_super = object_cats_list #[n_.split('_')[0] for n_ in obj_names]
        num_objs = len(obj_names)

        obj_types_scene_ = np.array([self.name_to_id[obj_types_super[i]] for i in range(len(obj_types_super))])
        pad_len = num_objs - obj_types_scene_.shape[0]
        obj_types_scene_ = np.concatenate((obj_types_scene_, -1*np.ones(pad_len)), 0)
        obj_types_scene = obj_types_scene_

        # get adjacenecy matrices
        name_to_id = {obj_names[i]:i for i in range(len(obj_names))}
        num_relations = len(list(self.rel_to_id.keys()))
        subj_pred_obj_adj_scene = torch.zeros((num_objs, num_relations, num_objs))
        for tiplet_idx in range(len(subjs_super)):
            split_ = triplet.split()
            subj_ind = name_to_id[subjs_super[tiplet_idx]]
            pred_ind = self.rel_to_id[preds_super[tiplet_idx]]
            obj_ind = name_to_id[objs_super[tiplet_idx]]
            subj_pred_obj_adj_scene[subj_ind, pred_ind, obj_ind] = 1.

        return subj_pred_obj_adj_scene, preds_super 

    def group(self, seq, sep):
        g = []
        for el in seq:
            if el in sep:
                yield g[1:]
                g = []
            g.append(el)
        yield g

