import ipdb
st = ipdb.set_trace
from arguments import args
import numpy as np
import utils.geom
import torch
from backend import saverloader
from PIL import Image
import sys
from SOLQ.util import box_ops
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from utils.wctb import Utils, Relations_CenterOnly
from SOLQ.util.misc import nested_tensor_from_tensor_list
import torchvision
import torch.nn.functional as F
import torchvision.ops
from nets.relationsnet_visual import OOPNet
from utils.ddetr_utils import check_for_detections_two_head
import utils.aithor

class IDOOP():

    def __init__(
        self,
        pix_T_camX,
        W,H,
        include_classes,
        id_to_name, 
        name_to_id, 
        ddetr=None,
        task=None,
        id_to_mapped_id=None, 
        ): 
        '''
        This class uses all GT
        '''

        # self.aithor_base = aithor_base
        # controller = aithor_base.controller
        self.W = W
        self.H = H
        self.pix_T_camX = pix_T_camX
        self.include_classes = include_classes
        self.id_to_name = id_to_name
        self.name_to_id = name_to_id
        self.id_to_mapped_id = id_to_mapped_id

        self.task = task

        self.score_labels_name1 = "pred1"
        self.score_labels_name2 = "pred2"

        self.dist_thresh = args.dist_thresh

        # out of place detector
        if args.do_visual_oop:

            self.ddetr = ddetr

            self.label_to_id = {False:0, True:1} # not out of place = False
            self.id_to_label = {0:False, 1:True}

            self.score_threshold_oop = args.score_threshold_oop #0.6
            self.score_threshold = args.confidence_threshold

    def exhuastive_oop_search(
        self, 
        navigation, 
        max_steps=500, 
        max_fail=30, 
        vis=None, 
        object_tracker=None
        ):
        '''
        navigation: navigation class 
        controller: agent controller
        oop_objects_gt: gt out of place meta data
        max_steps: maximum number of steps for search for choosing a new search area
        vis: visualization class if want to visualize robot actions
        '''
        # error = None

        # action_successful = True
        oop_dict = {}
        found_oop = False
        for s in range(args.num_search_locs_oop):
            print(f"Starting search #{s}")
            ind_i, ind_j = navigation.get_reachable_map_locations(sample=True)

            if not ind_i:
                oop_dict['found_oop'] = False
                return oop_dict

            search_mode = True
            navigation.set_point_goal(ind_i, ind_j, search_mode=search_mode)            

            steps = 0
            num_failed = 0
            while True:

                if self.task.is_done():
                    print("Task done! Skipping OOP search.")
                    break

                action, param = navigation.act(add_obs=False)

                if args.verbose:
                    print(f"exhuastive_oop_search: {action}")

                camX0_T_camX = navigation.explorer.get_camX0_T_camX()

                if steps>0 and object_tracker is not None and action_successful:
                    object_tracker.update(rgb, depth, camX0_T_camX, vis=vis)

                    if vis is not None:
                        vis.add_frame(rgb, text=f"Search #{s}")

                    if args.do_visual_oop:
                        oop_dict = self.check_for_oop_ddetr(rgb, depth, camX0_T_camX)
                        if len(oop_dict)>0:
                            if vis is not None:
                                for _ in range(5):
                                    vis.add_frame(rgb, text="Detection", box=oop_dict['box'])
                    else:
                        assert(False)

                if 'Pass' in action:
                    if steps==0:
                        pass
                    else:
                        break
                else:
                    self.task.step(action=action)

                action_successful = self.task.action_success() # controller.last_event.metadata["lastActionSuccess"]
                rgb, depth = navigation.get_obs(head_tilt=navigation.explorer.head_tilt)
                navigation.update_navigation_obs(rgb,depth, action_successful)

                if len(oop_dict)>0:
                    found_oop = True
                    oop_dict['found_oop'] = found_oop
                    if vis is not None:
                        for _ in range(5):
                            vis.add_frame(rgb, text="Detection", box=oop_dict['box'])
                    return oop_dict

                if not action_successful:
                    num_failed += 1

                steps += 1

                if steps >= max_steps:
                    break

                if max_fail is not None:
                    if num_failed >= max_fail:
                        if args.verbose: 
                            print("Max fail reached.")
                        break

        oop_dict['found_oop'] = False
        return oop_dict


    def check_for_oop_ddetr(self, rgb, depth, camX0_T_camX):
        oop_dict = {}
        out = check_for_detections_two_head(
            rgb, self.ddetr, self.W, self.H, 
            self.score_labels_name1, self.score_labels_name2, 
            score_threshold_head1=self.score_threshold, score_threshold_head2=self.score_threshold_oop, do_nms=True, target_object=None, target_object_score_threshold=None,
            solq=True, return_masks=True, nms_threshold=args.nms_threshold, id_to_mapped_id=self.id_to_mapped_id, return_features=False,
            )
        if len(out['pred_labels2'])>0:
            check_oop = [self.id_to_label[l] for l in out['pred_labels2']]
            if np.any(check_oop):
                print("Detected an out of place object!")
                idx_oop = np.where(check_oop)[0][0]
                oop_dict['label'] = out['pred_labels'][idx_oop]
                oop_dict['object_name'] = self.id_to_name[oop_dict['label']]
                oop_dict['box'] = out['pred_boxes'][idx_oop]
                oop_dict['score'] = out['pred_scores'][idx_oop]
                oop_dict['mask'] = out['pred_masks'][idx_oop]
                oop_dict['label_oop'] = out['pred_labels2'][idx_oop]
                oop_dict['score_oop'] = out['pred_scores2'][idx_oop]
                centroid = utils.aithor.get_centroid_from_detection_no_controller(
                    oop_dict['mask'], depth, 
                    self.W, self.H, 
                    centroid_mode='median', 
                    pix_T_camX=self.pix_T_camX, 
                    origin_T_camX=camX0_T_camX
                    )
                if centroid is None:
                    return {}
                oop_dict['centroid'] = centroid
                # get object backbone features
                rgb_x = (torch.from_numpy(rgb.copy()).float() / 255.0).cuda().permute(2,0,1).unsqueeze(0)
                # obj_bbox = pred_boxes[o_i]
                obj_bbox = (int(oop_dict['box'][0]), int(oop_dict['box'][1]), int(oop_dict['box'][2]), int(oop_dict['box'][3])) #obj_bbox.astype(np.int32)
                obj_bbox = np.array(obj_bbox)
                with torch.no_grad():
                    feature_map = self.ddetr.model.backbone(nested_tensor_from_tensor_list(rgb_x))[0][args.backbone_layer_ddetr].decompose()[0]
                    feature_crop = torchvision.ops.roi_align(feature_map, [torch.from_numpy(obj_bbox / 8).float().unsqueeze(0).cuda()], output_size=(32,32))
                    pooled_obj_feat = F.adaptive_avg_pool2d(feature_crop, (1,1)).squeeze(-1).squeeze(-1)
                oop_dict['features'] = pooled_obj_feat
        return oop_dict

        

    def pick_up_oop_obj(self, navigation, found_oop_dict, vis=None, object_tracker=None):
        
        obj_center = found_oop_dict['centroid']
        obj_name = found_oop_dict['object_name']

        obj_center_camX0_ = {'x':obj_center[0], 'y':-obj_center[1], 'z':obj_center[2]}
        map_pos = navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)

        ind_i, ind_j  = navigation.get_clostest_reachable_map_pos(map_pos) # get closest navigable point to object in overhead map

        navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.dist_thresh) # set point goal in map
        navigation.navigate_to_point_goal(vis=vis, text=f"Navigate to {obj_name}", object_tracker=object_tracker)
        
        navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.dist_thresh)
        navigation.orient_camera_to_point(obj_center_camX0_, vis=vis, text=f"Orient to {obj_name}", object_tracker=object_tracker)

        # get 2D reprojection
        camX0_T_camX = navigation.explorer.get_camX0_T_camX()
        obj_center_camX0_reproj = {'x':obj_center[0], 'y':obj_center[1], 'z':obj_center[2]}
        camX_T_camX0 = utils.geom.safe_inverse_single(camX0_T_camX)
        rgb = self.task.get_observations()["rgb"]
        point_2D, method_text = object_tracker.get_2D_point(
            camX_T_origin=camX_T_camX0, 
            obj_center_camX0_=obj_center_camX0_reproj, 
            object_category=obj_name, 
            rgb=rgb, 
            score_threshold=args.score_threshold_interaction
            )

        print(f"Picking up {obj_name}")

        # self.task.pickup_object(x=, y: float)

        offsets =   [
                    [0, 0], 
                    [0, 10], 
                    [10, 0], 
                    [0, -10], 
                    [-10, 0], 
                    [10, 10], 
                    [10, -10], 
                    [-10, 10], 
                    [-10, -10]
                    ]

        success = navigation.interact_object_xy(
            "PickupObject", 
            point_2D, 
            vis=vis, 
            offsets=offsets
            )

        return success

    def check_if_oop_in_view_gt(self, objects, oop_objects_gt_Ids, instance_detections2D):

        in_view = {}
        for obj in objects:
            if obj['name'] not in oop_objects_gt_Ids:
                print("FIXME")
                print(obj['name'], oop_objects_gt_Ids)
                continue
            if obj["visible"] and obj['objectId'] in instance_detections2D:
                obj_center = obj['axisAlignedBoundingBox']['center']
                obj_center = np.array(list(obj_center.values()))
                obj_id = obj['objectId']
                # in_view['obj_center'] = obj_center
                # in_view['objectId'] = obj_id
                # in_view['name'] = obj['name']
                # in_view['box'] = instance_detections2D[obj_id]
                in_view['box'] = instance_detections2D[obj_id]
                in_view['obj_center'] = obj_center
                in_view['objectId'] = obj_id
                in_view['objectType'] = obj['objectType']
                in_view['name'] = obj['name']
                break
                # in_view.append(obj)
        return in_view