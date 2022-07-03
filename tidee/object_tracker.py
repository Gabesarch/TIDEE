import numpy as np
import utils.aithor
import torch
from PIL import Image
import ipdb
st = ipdb.set_trace
from utils.ddetr_utils import check_for_detections
import numpy as np
import utils.aithor
import utils.geom
import torch
from PIL import Image
from arguments import args
import sys
from backend import saverloader
import os
import cv2
import matplotlib.pyplot as plt
import ipdb
st = ipdb.set_trace
from utils.ddetr_utils import check_for_detections
from scipy.spatial import distance

class ObjectTrack():

    def __init__(
        self, 
        name_to_id, 
        id_to_name, 
        include_classes, 
        W, H, 
        pix_T_camX=None, 
        ddetr=None, 
        use_gt_objecttrack=False, 
        controller=None, 
        navigator=None, 
        check_if_centroid_falls_within_map=False, 
        do_masks=False, 
        use_solq=False, 
        id_to_mapped_id=None, 
        on_aws=False,
        origin_T_camX0=None,
        ): 
        '''
        check_if_centroid_falls_within_map: make sure centroid is within map bounds as given by navigator
        '''

        print("Initializing object tracker")

        self.origin_T_camX0 = origin_T_camX0

        self.controller = controller

        self.use_gt_objecttrack = use_gt_objecttrack

        self.include_classes = include_classes

        self.do_masks = do_masks
        self.use_solq = use_solq
        self.nms_threshold = args.nms_threshold
        self.id_to_mapped_id = id_to_mapped_id

        # self.ddetr = ddetr
        # self.ddetr.cuda()
        assert(not (do_masks and not use_solq))
        
        if not use_gt_objecttrack:
            if ddetr is None:
                if use_solq:
                    sys.path.append('project_cleanup/SOLQ')
                    args.data_mode = "solq"
                    from nets.solq import DDETR
                    # exec(compile(open('exp_aithor_solq.py').read(), 'exp_aithor_solq.py', 'exec'))
                    load_pretrained = False
                    self.ddetr = DDETR(len(self.include_classes), load_pretrained).cuda()
                    if on_aws:
                        checkpoint_root = "./models/checkpoints/"
                        model_name = 'solq-00013500.pth'
                    else:
                        # checkpoint_root = "/projects/katefgroup/viewpredseg/checkpoints/TEACH_solq_aithor04"
                        # model_name = 'model-00013500.pth'
                        checkpoint_root = "/projects/katefgroup/viewpredseg/checkpoints/TEACH_solq_aithor05"
                        model_name = 'model-00023000.pth'
                    path = os.path.join(checkpoint_root, model_name)
                    print("...found checkpoint %s"%(path))
                    checkpoint = torch.load(path)
                    pretrained_dict = checkpoint['model_state_dict']
                    # pretrained_dict = {'model.'+k: v for k, v in pretrained_dict.items()}
                    self.ddetr.load_state_dict(pretrained_dict, strict=True)
                    # load_pretrained = False
                    # _ = saverloader.load(model_name, checkpoint_root, self.ddetr, None, strict=True)
                    self.ddetr.eval().cuda()
                else:
                    sys.path.append('project_cleanup/Deformable-DETR')
                    from nets.ddetr import DDETR
                    load_pretrained = False
                    # exec(compile(open('exp_aithor_ddetr.py').read(), 'exp_aithor_ddetr.py', 'exec'))
                    self.ddetr = DDETR(len(self.include_classes), load_pretrained).cuda()
                    # checkpoint_root = "/projects/katefgroup/viewpredseg/checkpoints/"
                    if on_aws:
                        checkpoint_root = "./models/checkpoints/"
                        model_name = "model-00010000.pth"
                    else:
                        checkpoint_root = "/projects/katefgroup/viewpredseg/checkpoints/ddetr_aithor_amodal01"
                        model_name = "model-00010000.pth"
                    path = os.path.join(checkpoint_root, model_name)
                    print("...found checkpoint %s"%(path))
                    checkpoint = torch.load(path)
                    pretrained_dict = checkpoint['model_state_dict']
                    # _ = saverloader.load(model_name, checkpoint_root, self.ddetr, None, strict=True)
                    self.ddetr.load_state_dict(pretrained_dict, strict=True)
                    self.ddetr.eval().cuda()
            else:
                self.ddetr = ddetr
                self.ddetr.cuda()

        self.objects_track_dict = {}
        self.attributes = {"label", "locs", "holding", "scores", "can_use"}
    
        self.score_threshold = args.confidence_threshold #0.4 # threshold for adding objects to memory 
        self.target_object_threshold = args.confidence_threshold_searching # if searching for specific object, lower threshold
        
        # print(f"Score threshold={self.score_threshold}, score threshold searching={self.target_object_threshold}")

        self.W = W
        self.H = H

        self.name_to_id = name_to_id
        self.id_to_name = id_to_name
        
        self.dist_threshold = args.OT_dist_thresh #1.0

        self.pix_T_camX = pix_T_camX

        self.only_one_obj_per_cat = args.only_one_obj_per_cat

        self.id_index = 0

        self.navigator = navigator
        self.check_if_centroid_falls_within_map = check_if_centroid_falls_within_map
        self.centroid_map_threshold = 1.5 # allowable distance to nearest navigable point
        self.centroid_map_threshold_in_bound = 0.75 # allowable distance to nearest non-navigable point

        self.score_boxes_name = 'pred1' # only one prediction head so same for both
        self.score_labels_name = 'pred1'

    def update(
        self, 
        rgb, 
        depth, 
        camX0_T_camX, 
        return_det_dict=False, 
        use_gt=False, 
        target_object=None, 
        vis=None, 
        return_features=False
        ):

        
        if self.use_gt_objecttrack:
            print("Using GT Object Tracker")
            pred_scores, pred_labels, pred_boxes_or_masks, centroids_gt = self.get_objects_gt(self.controller)
        else:
            if target_object is not None:
                if type(target_object)==str:
                    target_object_id = self.name_to_id[target_object]
                elif type(target_object)==list:
                    target_object_id = [self.name_to_id[target_object_] for target_object_ in target_object]
                else:
                    assert(False)
            else:
                target_object_id = None
            out = check_for_detections(
                rgb, self.ddetr, self.W, self.H, 
                self.score_labels_name, self.score_boxes_name, 
                score_threshold_ddetr=self.score_threshold, do_nms=True, target_object=target_object_id, target_object_score_threshold=self.target_object_threshold,
                solq=self.use_solq, return_masks=self.do_masks, nms_threshold=self.nms_threshold, id_to_mapped_id=self.id_to_mapped_id, return_features=return_features,
                )
            pred_labels = out["pred_labels"]
            pred_scores = out["pred_scores"]
            if self.do_masks:
                pred_boxes_or_masks = out["pred_masks"] 
            else:
                pred_boxes_or_masks = out["pred_boxes"] 

        if return_det_dict:
            det_dict = out
            det_dict["centroid"] = []

        if self.check_if_centroid_falls_within_map:
            reachable = self.navigator.get_reachable_map_locations(sample=False)
            inds_i, inds_j = np.where(reachable)
            reachable_where = np.stack([inds_i, inds_j], axis=0)
        
        diffs_ = []
        if len(pred_scores)>0:
            if vis is not None:
                rgb_ = np.float32(rgb.copy())
            for d in range(len(pred_scores)):
                if not self.use_gt_objecttrack:
                    label = self.id_to_name[pred_labels[d]]
                    if label in self.id_to_mapped_id.keys():
                        label = self.id_to_mapped_id[label]
                else:
                    label = pred_labels[d]
                score = pred_scores[d]
                box = pred_boxes_or_masks[d]
                if len(box)==0:
                    continue
                if self.use_gt_objecttrack and args.use_GT_masks:
                    centroid = utils.aithor.get_centroid_from_detection_no_controller(
                        box, depth, 
                        self.W, self.H, 
                        centroid_mode='median', 
                        pix_T_camX=self.pix_T_camX, 
                        origin_T_camX=camX0_T_camX
                        )
                elif self.use_gt_objecttrack and args.use_GT_centroids:
                    centroids_gt[d]
                else:
                    centroid = utils.aithor.get_centroid_from_detection_no_controller(
                        box, depth, 
                        self.W, self.H, 
                        centroid_mode='median', 
                        pix_T_camX=self.pix_T_camX, 
                        origin_T_camX=camX0_T_camX
                        )
                if centroid is None:
                    continue

                if self.check_if_centroid_falls_within_map:
                    obj_center_camX0_ = {'x':centroid[0], 'y':centroid[1], 'z':centroid[2]}
                    map_pos_centroid = self.navigator.get_map_pos_from_aithor_pos(obj_center_camX0_)
                    dist_to_reachable = distance.cdist(np.expand_dims(map_pos_centroid, axis=0), reachable_where.T)
                    argmin = np.argmin(dist_to_reachable)
                    map_pos_reachable_closest = [inds_i[argmin], inds_j[argmin]]
                    # map_pos_reachable_closest = self.navigator.get_clostest_reachable_map_pos(map_pos_centroid)
                    if not map_pos_centroid[0] or not map_pos_reachable_closest[1]:
                        pass
                    else:
                        dist = np.linalg.norm(map_pos_centroid - map_pos_reachable_closest) * self.navigator.explorer.resolution 
                        if dist>self.centroid_map_threshold:
                            continue
                
                if return_det_dict:
                    det_dict['centroid'].append(centroid)

                if not self.use_gt_objecttrack:
                    if label not in self.name_to_id:
                        continue
                else:
                    if label=="Floor":
                        continue

                if vis is not None and args.visualize_masks:
                    rect_th = 1
                    if label==target_object:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                    if len(box)==4:
                        cv2.rectangle(rgb_, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),color, rect_th)
                    else:
                        box = np.squeeze(box)
                        masked_img = np.where(box[...,None], color, rgb_)
                        rgb_ = cv2.addWeighted(rgb_, 0.8, np.float32(masked_img), 0.2,0)

                # check if detected object already exists. if it does, add only one with highest score
                locs = []
                IDs_same = []
                for id_ in self.objects_track_dict.keys():
                    if self.objects_track_dict[id_]["label"]==label and self.objects_track_dict[id_]["locs"] is not None:
                        locs.append(self.objects_track_dict[id_]["locs"])
                        IDs_same.append(id_)
                if self.only_one_obj_per_cat:
                    if len(locs)>0:
                        locs = np.array(locs)
                        dists = np.sqrt(np.sum((locs - np.expand_dims(centroid, axis=0))**2, axis=1))
                        dists_thresh = dists<self.dist_threshold
                        if np.sum(dists_thresh)>0:
                            same_ind = np.where(dists_thresh)[0][0]
                            same_id = IDs_same[same_ind]
                            loc_cur = self.objects_track_dict[same_id]['locs']
                            score_cur = self.objects_track_dict[same_id]['scores']
                            holding_cur = self.objects_track_dict[same_id]['holding']
                            # add one with highest score if they are the same object
                            if not holding_cur:
                                if score>=score_cur:
                                    self.objects_track_dict[same_id]['scores'] = score
                                    self.objects_track_dict[same_id]['locs'] = centroid
                                else:
                                    self.objects_track_dict[same_id]['scores'] = score_cur
                                    self.objects_track_dict[same_id]['locs'] = loc_cur
                    else:
                        self.objects_track_dict[self.id_index] = {}
                        for attr in self.attributes:
                            if attr=="locs":
                                self.objects_track_dict[self.id_index][attr] = centroid
                            elif attr=="label":
                                self.objects_track_dict[self.id_index][attr] = label
                            elif attr=="holding":
                                self.objects_track_dict[self.id_index][attr] = False
                            elif attr=="scores":
                                self.objects_track_dict[self.id_index][attr] = score
                            elif attr=="can_use":
                                self.objects_track_dict[self.id_index][attr] = True
                            else:
                                print(attr)
                                assert(False) # didnt add this attribute
                        self.id_index += 1
                else:
                    if len(locs)>0:
                        locs = np.array(locs)
                        dists = np.sqrt(np.sum((locs - np.expand_dims(centroid, axis=0))**2, axis=1))
                        dists_thresh = dists<self.dist_threshold
                    else:
                        dists_thresh = 0 # no objects of this class in memory
                    if np.sum(dists_thresh)>0:
                        same_ind = np.where(dists_thresh)[0][0]
                        same_id = IDs_same[same_ind]
                        loc_cur = self.objects_track_dict[same_id]['locs']
                        score_cur = self.objects_track_dict[same_id]['scores']
                        holding_cur = self.objects_track_dict[same_id]['holding']
                        # add one with highest score if they are the same object
                        if not holding_cur:
                            if score>=score_cur:

                                self.objects_track_dict[same_id]['scores'] = score
                                self.objects_track_dict[same_id]['locs'] = centroid
                                # self.objects_track_dict[label]['holding'].append(False)
                            else:
                                self.objects_track_dict[same_id]['scores'] = score_cur
                                self.objects_track_dict[same_id]['locs'] = loc_cur
                                # self.objects_track_dict[label]['holding'].append(False)
                    else:
                        self.objects_track_dict[self.id_index] = {}
                        for attr in self.attributes:
                            if attr=="locs":
                                self.objects_track_dict[self.id_index][attr] = centroid
                            elif attr=="label":
                                self.objects_track_dict[self.id_index][attr] = label
                            elif attr=="holding":
                                self.objects_track_dict[self.id_index][attr] = False
                            elif attr=="scores":
                                self.objects_track_dict[self.id_index][attr] = score
                            elif attr=="can_use":
                                self.objects_track_dict[self.id_index][attr] = True
                            else:
                                print(attr)
                                assert(False) # didnt add this attribute
                        self.id_index += 1

            if vis is not None:
                vis.add_frame(rgb_, text="Update object tracker")
        
        if return_det_dict:
            return det_dict

    def add_centroid_from_point(self, point_2D_, depth, camX0_T_camX, object_cat, override_existing=True, override_existing_id=None, pad=20, just_return_centroid=False):

        box = np.clip(np.array([int(point_2D_[1]*self.W)-pad, int(point_2D_[0]*self.H)-pad, int(point_2D_[1]*self.W)+pad, int(point_2D_[0]*self.H)+pad]), a_min=0, a_max=self.W-1)
        c_depth = utils.aithor.get_centroid_from_detection_no_controller(box, depth, self.W, self.H, centroid_mode='median', pix_T_camX=self.pix_T_camX, origin_T_camX=camX0_T_camX, num_valid_thresh=1)
        if just_return_centroid or c_depth is None:
            return c_depth

        if override_existing:
            if override_existing_id is not None:
                self.objects_track_dict[override_existing_id]["locs"] = c_depth
                self.objects_track_dict[override_existing_id]["holding"] = False
                self.objects_track_dict[override_existing_id]["score"] = 1.01
            else:
                same_id = self.get_overlapping_id(c_depth, object_cat)
                if same_id is not None:
                    self.objects_track_dict[same_id] = {}
                    self.objects_track_dict[same_id]['scores'] = 1.01
                    self.objects_track_dict[same_id]['label'] = object_cat
                    self.objects_track_dict[same_id]['locs'] = c_depth
                    self.objects_track_dict[same_id]['holding'] = False
                    self.objects_track_dict[same_id]['can_use'] = True
                    self.id_index += 1
                else:
                    self.objects_track_dict[self.id_index] = {}
                    self.objects_track_dict[self.id_index]['scores'] = 1.01
                    self.objects_track_dict[self.id_index]['label'] = object_cat
                    self.objects_track_dict[self.id_index]['locs'] = c_depth
                    self.objects_track_dict[self.id_index]['holding'] = False
                    self.objects_track_dict[self.id_index]['can_use'] = True
                    self.id_index += 1
        else:
            self.objects_track_dict[self.id_index] = {}
            self.objects_track_dict[self.id_index]['scores'] = 1.01
            self.objects_track_dict[self.id_index]['label'] = object_cat
            self.objects_track_dict[self.id_index]['locs'] = c_depth
            self.objects_track_dict[self.id_index]['holding'] = False
            self.objects_track_dict[self.id_index]['can_use'] = True
            self.id_index += 1

    def get_overlapping_id(self, c_depth, object_cat):
        locs = []
        IDs_same = []
        for id_ in self.objects_track_dict.keys():
            if self.objects_track_dict[id_]["label"]==object_cat and self.objects_track_dict[id_]["locs"] is not None:
                locs.append(self.objects_track_dict[id_]["locs"])
                IDs_same.append(id_)
        if len(locs)>0:
            locs = np.array(locs)
            dists = np.sqrt(np.sum((locs - np.expand_dims(c_depth, axis=0))**2, axis=1))
            dists_thresh = dists<self.dist_threshold
        else:
            dists_thresh = 0
        if np.sum(dists_thresh)>0:
            same_ind = np.where(dists_thresh)[0][0]
            same_id = IDs_same[same_ind]
        else:
            same_id = None
        return same_id


    def filter_centroids_out_of_bounds(self):
        reachable = self.navigator.get_reachable_map_locations(sample=False)
        inds_i, inds_j = np.where(reachable)
        reachable_where = np.stack([inds_i, inds_j], axis=0)

        centroids, labels, IDs = self.get_centroids_and_labels(return_ids=True)

        for idx in range(len(IDs)):
            centroid = centroids[idx]
            obj_center_camX0_ = {'x':centroid[0], 'y':centroid[1], 'z':centroid[2]}
            map_pos_centroid = self.navigator.get_map_pos_from_aithor_pos(obj_center_camX0_)
            dist_to_reachable = distance.cdist(np.expand_dims(map_pos_centroid, axis=0), reachable_where.T)
            argmin = np.argmin(dist_to_reachable)
            map_pos_reachable_closest = [inds_i[argmin], inds_j[argmin]]
            if not map_pos_centroid[0] or not map_pos_reachable_closest[1]:
                pass
            else:
                dist = np.linalg.norm(map_pos_centroid - map_pos_reachable_closest) * self.navigator.explorer.resolution 
                if dist>self.centroid_map_threshold:
                    # print("centroid outside map bounds. continuing..")
                    del self.objects_track_dict[IDs[idx]]

    def filter_centroids_in_navigable_bounds(self):

        # if self.check_if_centroid_falls_within_map:
        reachable = self.navigator.get_reachable_map_locations(sample=False)

        reachable = 1 - reachable

        inds_i, inds_j = np.where(reachable)
        reachable_where = np.stack([inds_i, inds_j], axis=0)

        centroids, labels, IDs = self.get_centroids_and_labels(return_ids=True)

        for idx in range(len(IDs)):
            centroid = centroids[idx]
            obj_center_camX0_ = {'x':centroid[0], 'y':centroid[1], 'z':centroid[2]}
            map_pos_centroid = self.navigator.get_map_pos_from_aithor_pos(obj_center_camX0_)
            dist_to_reachable = distance.cdist(np.expand_dims(map_pos_centroid, axis=0), reachable_where.T)
            argmin = np.argmin(dist_to_reachable)
            map_pos_reachable_closest = [inds_i[argmin], inds_j[argmin]]
            if not map_pos_centroid[0] or not map_pos_reachable_closest[1]:
                pass
            else:
                dist = np.linalg.norm(map_pos_centroid - map_pos_reachable_closest) * self.navigator.explorer.resolution 
                if dist>self.centroid_map_threshold_in_bound:
                    # print("centroid outside map bounds. continuing..")
                    del self.objects_track_dict[IDs[idx]]


    def get_ID_of_holding(self):
        for key in list(self.objects_track_dict.keys()):
            cur_dict = self.objects_track_dict[key]
            if cur_dict['holding']:
                return key
        return None


    def get_label_of_holding(self):
        key = self.get_ID_of_holding()
        if key is not None:
            return self.objects_track_dict[key]['label']
        return None

    def get_centroids_and_labels(self, return_ids=False, object_cat=None):
        '''
        get centroids and labels in memory
        object_cat: object category string or list of category strings
        '''
        # order by score 
        scores = []
        IDs = []
        for key in list(self.objects_track_dict.keys()):
            cur_dict = self.objects_track_dict[key]
            if not cur_dict['holding']:
                scores.append(cur_dict['scores'])
                IDs.append(key)

        scores_argsort = np.argsort(-np.array(scores))
        IDs = np.array(IDs)[scores_argsort]

        # iterate through with highest score first
        centroids = []
        labels = []
        IDs_ = []
        for key in list(IDs):
            cur_dict = self.objects_track_dict[key]
            if object_cat is not None:
                # check for category if input
                if type(object_cat)==str:
                    cat_check = cur_dict['label']==object_cat
                elif type(object_cat)==list:
                    cat_check = cur_dict['label'] in object_cat
                else:
                    assert(False) # wrong object_cat input
            else:
                cat_check = True
            if not cur_dict['holding'] and cat_check and cur_dict['can_use']:
                centroids.append(cur_dict['locs'])
                labels.append(cur_dict['label'])
                IDs_.append(key)
                # IDs.append(key)

        if return_ids:
            return np.array(centroids), labels, IDs_

        return np.array(centroids), labels

    def get_2D_point(
        self, 
        camX_T_origin=None, 
        obj_center_camX0_=None, 
        object_category=None, 
        rgb=None, 
        score_threshold=0.0
        ):
        '''
        modes: reproject_centroid, 
        '''

        if not self.use_gt_objecttrack: # first use detector? 
            # first see if detector has it
            out = check_for_detections(
                    rgb, self.ddetr, self.W, self.H, 
                    self.score_labels_name, self.score_boxes_name, 
                    score_threshold_ddetr=score_threshold, do_nms=False, return_features=False,
                    solq=self.use_solq, return_masks=self.do_masks, nms_threshold=self.nms_threshold, id_to_mapped_id=self.id_to_mapped_id,
                    )
            pred_labels = out["pred_labels"]
            pred_scores = out["pred_scores"]
            if self.do_masks:
                pred_boxes_or_masks = out["pred_masks"] 
            else:
                pred_boxes_or_masks = out["pred_boxes"] 
            pred_scores_sorted = np.argsort(-pred_scores)
            for d in range(len(pred_scores)):
                idx = pred_scores_sorted[d]
                label = self.id_to_name[pred_labels[idx]]
                if label==object_category:
                    if self.do_masks:
                        mask = pred_boxes_or_masks[idx]
                        where_target_i, where_target_j = np.where(mask)
                        idx_mask = int(len(where_target_i)/2)
                        where_target_i = where_target_i[idx_mask]
                        where_target_j = where_target_j[idx_mask]
                        center2D = [where_target_i, where_target_j]
                    else:
                        box = pred_boxes_or_masks[idx]
                        x_min, y_min, x_max, y_max = list(np.round(box).astype(np.int))
                        center2D = [(y_min+y_max)/2, (x_min+x_max)/2]
                        print("RETURNING DETECTED 2D POINT")
                    return center2D, "DETECTED"

        print("RETURNING REPROJECTED 2D POINT")

        # Camera2Pixels
        obj_center_camX0_ = torch.from_numpy(np.array(list(obj_center_camX0_.values())))
        obj_center_camX0_ = torch.reshape(obj_center_camX0_, [1, 1, 3])
        object_center_camX = utils.geom.apply_4x4(camX_T_origin.float(), obj_center_camX0_.float())
        pix_T_cam = torch.from_numpy(self.pix_T_camX).unsqueeze(0).float()
        center2D = np.squeeze(utils.geom.apply_pix_T_cam(pix_T_cam, object_center_camX).numpy())
        center2D = center2D[[1,0]] # need to swap these
        center2D = list(center2D)
        return center2D, "REPROJECTED"


    def get_objects_gt(self, controller):
        '''
        Gets object info from segmentation mask
        '''

        origin_T_camX = utils.aithor.get_origin_T_camX(controller.last_event, False).cuda()

        semantic = controller.last_event.instance_segmentation_frame
        object_id_to_color = controller.last_event.object_id_to_color
        color_to_object_id = controller.last_event.color_to_object_id

        obj_ids = np.unique(semantic.reshape(-1, semantic.shape[2]), axis=0)
        
        obj_metadata_IDs = []
        for obj_m in controller.last_event.metadata['objects']: #objects:
            obj_metadata_IDs.append(obj_m['objectId'])

        instance_masks = controller.last_event.instance_masks
        instance_detections2d = controller.last_event.instance_detections2D

        bboxes = []
        labels = []
        scores = []
        centroids = []

        loop_inds = np.arange(obj_ids.shape[0])

        for obj_idx_loop in list(loop_inds): # skip target object

            # sometimes this fails?
            try:
                obj_color = tuple(obj_ids[obj_idx_loop])
                object_id = color_to_object_id[obj_color]
            except:
                continue

            if object_id not in obj_metadata_IDs:
                continue

            obj_meta_index = obj_metadata_IDs.index(object_id)
            obj_meta = controller.last_event.metadata['objects'][obj_meta_index]

            obj_category_name = obj_meta['objectType']

            i_mask = instance_masks[object_id]

            # obj_bbox = instance_detections2d[object_id] #[[0,2,1,3]]

            # obj_3dbox_origin = utils.aithor.get_3dbox_in_geom_format(obj_meta)
            # # get amodal box
            # boxlist2d_amodal, obj_3dbox_camX = utils.aithor.get_amodal2d(origin_T_camX.cuda(), obj_3dbox_origin.cuda(), torch.from_numpy(self.pix_T_camX).unsqueeze(0).cuda(), self.H, self.W)
            # boxlist2d_amodal = boxlist2d_amodal.cpu().numpy()
            
            # boxlist2d_amodal_clip = np.zeros(4)
            # boxlist2d_amodal_clip[[0,2]] = np.clip(boxlist2d_amodal[[0,2]], 0, self.W)
            # boxlist2d_amodal_clip[[1,3]] = np.clip(boxlist2d_amodal[[1,3]], 0, self.H)
            # iou_inview = utils.box.boxlist_2d_iou(boxlist2d_amodal.reshape(1,4), boxlist2d_amodal_clip.reshape(1,4))

            # obj_bbox = obj_bbox # [start_x, start_y, end_x, end_y]   

            # centroid = np.array(list(obj_meta['axisAlignedBoundingBox']['center'].values())).unsqueeze(0).cuda()
            # centroid = utils.geom.apply_4x4(self.origin_T_camX0.unsqueeze(0).cuda().float(), centroid.unsqueeze(1).cuda().float()).squeeze(1)
            # # centroid[:,1] = -centroid[:,1]   

            # bring to camX0 reference frame
            centroid = torch.from_numpy(np.array(list(obj_meta['axisAlignedBoundingBox']['center'].values()))).unsqueeze(0).cuda()
            centroid = utils.geom.apply_4x4(self.camX0_T_origin.unsqueeze(0).cuda().float(), centroid.unsqueeze(1).cuda().float()).squeeze(1)
            centroid[:,1] = -centroid[:,1]      
            centroid = centroid.squeeze().cpu().numpy()       

            bboxes.append(i_mask) #obj_bbox)
            labels.append(obj_category_name)
            scores.append(1.)
            centroids.append(centroid)

        return scores, labels, bboxes, centroids

    def get_objects_gt_from_meta(self):

        # origin_T_camX = utils.aithor.get_origin_T_camX(controller.last_event, False).cuda()

        # semantic = controller.last_event.instance_segmentation_frame
        # object_id_to_color = controller.last_event.object_id_to_color
        # color_to_object_id = controller.last_event.color_to_object_id

        # obj_ids = np.unique(semantic.reshape(-1, semantic.shape[2]), axis=0)
        
        # obj_metadata_IDs = []
        # for obj_m in controller.last_event.metadata['objects']: #objects:
        #     obj_metadata_IDs.append(obj_m['objectId'])

        # instance_masks = controller.last_event.instance_masks
        # instance_detections2d = controller.last_event.instance_detections2D

        bboxes = []
        labels = []
        scores = []
        centroids = []

        # loop_inds = np.arange(obj_ids.shape[0])

        objects = self.controller.last_event.metadata['objects']

        for obj_meta in objects: # skip target object

            # obj_meta = controller.last_event.metadata['objects']

            obj_category_name = obj_meta['objectType']

            # i_mask = instance_masks[object_id]

            # obj_bbox = instance_detections2d[object_id] #[[0,2,1,3]]

            # obj_3dbox_origin = utils.aithor.get_3dbox_in_geom_format(obj_meta)
            # # get amodal box
            # boxlist2d_amodal, obj_3dbox_camX = utils.aithor.get_amodal2d(origin_T_camX.cuda(), obj_3dbox_origin.cuda(), torch.from_numpy(self.pix_T_camX).unsqueeze(0).cuda(), self.H, self.W)
            # boxlist2d_amodal = boxlist2d_amodal.cpu().numpy()
            
            # boxlist2d_amodal_clip = np.zeros(4)
            # boxlist2d_amodal_clip[[0,2]] = np.clip(boxlist2d_amodal[[0,2]], 0, self.W)
            # boxlist2d_amodal_clip[[1,3]] = np.clip(boxlist2d_amodal[[1,3]], 0, self.H)
            # iou_inview = utils.box.boxlist_2d_iou(boxlist2d_amodal.reshape(1,4), boxlist2d_amodal_clip.reshape(1,4))

            # obj_bbox = obj_bbox # [start_x, start_y, end_x, end_y]   

            # centroid = np.array(list(obj_meta['axisAlignedBoundingBox']['center'].values()))   

            # transform to camX0 reference frame
            centroid = torch.from_numpy(np.array(list(obj_meta['axisAlignedBoundingBox']['center'].values()))).unsqueeze(0).cuda()
            centroid = utils.geom.apply_4x4(self.camX0_T_origin.unsqueeze(0).cuda().float(), centroid.unsqueeze(1).cuda().float()).squeeze(1)
            centroid[:,1] = -centroid[:,1]    
            centroid = centroid.squeeze().cpu().numpy()             

            # bboxes.append(i_mask) #obj_bbox)

            labels.append(obj_category_name)
            scores.append(1.)
            centroids.append(centroid)

        return np.array(centroids), labels

    