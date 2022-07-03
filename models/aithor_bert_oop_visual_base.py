import numpy as np
import os
from arguments import args
import random
import math
from tensorboardX import SummaryWriter
from ai2thor_docker.ai2thor_docker.x_server import startx
from utils.wctb import Utils, Relations_CenterOnly, ThorPositionTo2DFrameTranslator
import torch
import utils.aithor
import utils.geom
from PIL import Image
import ipdb
st = ipdb.set_trace
import matplotlib.pyplot as plt
from task_base.aithor_base import Base
from itertools import cycle
from task_base.messup import mess_up, mess_up_from_loaded
from tidee.object_tracker import ObjectTrack
from task_base.tidy_task import TIDEE_TASK
from tidee.navigation import Navigation
from task_base.animation_util import Animation
from transformers import BertTokenizer
from utils.ddetr_utils import check_for_detections_two_head

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

class Ai2Thor_Base(Base):

    def __init__(self):   

        super(Ai2Thor_Base, self).__init__()

        self.utils = Utils(args.H, args.W)
        self.relations_util = Relations_CenterOnly(args.H, args.W)
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

        self.dist_threshold = args.dist_thresh

        self.max_relations = args.max_relations

    def run_agent(self, load_dict=None, obs_dict=None):

        outputs = self.run_agent_solq(load_dict=load_dict, obs_dict=obs_dict)
        return outputs

    def run_agent_solq(self, load_dict=None, obs_dict=None, z=[0.05, 2.0]):
        '''
        max_openable_per_cat: how many max of each category to get observations for
        '''
        
        if args.randomize_object_placements:
            print("Randomizing object placement")
            # randomize scene
            random_seed = np.random.randint(1000)
            self.controller.step(action="InitialRandomSpawn",
                randomSeed=random_seed,
                forceVisible=True,
                numPlacementAttempts=5,
                placeStationary=False,
            )

        # if args.closed_prob is not None:
        if args.randomize_object_state:
            print("Randomizing object states")
            # print("opening some objects...")
            for obj_meta in self.controller.last_event.metadata['objects']: #objects:

                # obj_category_name = obj_meta['objectType']
                # print(obj_category_name)

                # if obj_category_name not in self.name_to_id:
                #     continue

                if obj_meta['openable']:
                    openness_increments = 0.1
                    openness_degrees = [-1] + list(np.round(np.arange(0,1+openness_increments,openness_increments),decimals=1))
                    increments = openness_degrees[1:]
                    closed_prob = 0.5
                    p_not_closed = (1-closed_prob)/len(increments[1:])
                    p = list(np.ones(len(increments[1:]))*p_not_closed) # keep probability of closed fixed (a bit higher than the rest)
                    p = [closed_prob] + p 
                    openness = np.random.choice(increments, p=p)
                    openness = np.round(openness, decimals=1)
                    # print("opening", obj_meta['objectId'], "by", openness)
                    self.controller.step(
                        action="OpenObject",
                        objectId=obj_meta['objectId'],
                        openness=openness,
                        forceAction=True # set to True otherwise agent needs to be close to it
                    )

                if obj_meta['cookable']:
                    prob = np.random.uniform(0, 1)
                    if prob>0.8:
                        self.controller.step(
                            action="CookObject",
                            objectId=obj_meta['objectId'],
                            forceAction=True
                        )

                if obj_meta['canFillWithLiquid']:
                    prob = np.random.uniform(0, 1)
                    if prob>0.8:
                        liquid = np.random.choice(np.array(["coffee", "wine", "water"]))
                        self.controller.step(
                            action="FillObjectWithLiquid",
                            objectId=obj_meta['objectId'],
                            fillLiquid=liquid,
                            forceAction=True
                        )

                if obj_meta['sliceable']:
                    prob = np.random.uniform(0, 1)
                    if prob>0.7:
                        self.controller.step(
                            action="SliceObject",
                            objectId=obj_meta['objectId'],
                            forceAction=True
                        )

                if obj_meta['toggleable']:
                    prob = np.random.uniform(0, 1)
                    if prob>0.8:
                        self.controller.step(
                            action="ToggleObjectOn",
                            objectId=obj_meta['objectId'],
                            forceAction=True
                        )

                if obj_meta['dirtyable']:
                    prob = np.random.uniform(0, 1)
                    if prob>0.8:
                        self.controller.step(
                            action="DirtyObject",
                            objectId=obj_meta['objectId'],
                            forceAction=True
                        )


        if args.do_predict_oop:
            if args.mess_up_from_loaded:
                object_dict = load_dict['object_dict'] 
                oop_IDs = load_dict['oop_IDs'] 
                object_messup_meta = load_dict['objects_messup'] 
                mess_up_from_loaded(self.controller, object_messup_meta)
                print("MESSED UP FROM LOADED")
            else:
                object_dict, oop_IDs = mess_up(self.controller, self.include_classes, args.num_objects)
        else:
            object_dict = None     

        # object_tracker = ObjectTrack(self.name_to_id, self.id_to_name, self.include_classes, self.W, self.H, self.pix_T_camX, ddetr=self.ddetr)

        object_tracker = ObjectTrack(
                    self.name_to_id, 
                    self.id_to_name, 
                    self.include_classes, 
                    self.W, self.H, 
                    pix_T_camX=self.pix_T_camX, 
                    # origin_T_camX0=None, 
                    ddetr=self.ddetr, 
                    controller=None, 
                    use_gt_objecttrack=False,
                    do_masks=True,
                    use_solq=True,
                    id_to_mapped_id=self.id_to_mapped_id,
                    on_aws = False, 
                    navigator=None,
                    )

        if False:
            print("LOGGING THIS ITERATION")
            vis = Animation(self.W, self.H, navigation=navigation, name_to_id=self.name_to_id)
            print('Height:', self.H, 'Width:', self.W)
        else:
            vis = None
        
        if args.explore_env:
            self.tidee_task = TIDEE_TASK(
                self.controller, 
                'test', 
                max_episode_steps=args.max_steps_object_goal_nav
                )
            self.tidee_task.next_ep_called = True # hack to start episode
            self.tidee_task.done_called = False
            self.tidee_task.mapname_current = self.controller.scene

            navigation = Navigation(
                # controller=controller, 
                keep_head_down=args.keep_head_down, 
                keep_head_straight=args.keep_head_straight, 
                search_pitch_explore=args.search_pitch_explore, 
                pix_T_camX=self.pix_T_camX,
                task=self.tidee_task,
                )

            obs_dict = navigation.explore_env(
                object_tracker=object_tracker, 
                vis=vis, 
                return_obs_dict=True,
                max_fail=30, 
                use_aithor_coord_frame=True,
                )

            # if vis is not None:
            #     vis.render_movie(args.movie_dir, 0, tag=f'bert_oop')
            #     st()

        if args.load_object_tracker:
            assert(obs_dict is not None)

                        
        rgb_scene_batch, targets, multiview_batch = [], [], []          

        objects_meta_all = self.controller.last_event.metadata['objects']

        random.shuffle(objects_meta_all)

        # st()
        objects_meta_all_cycle = cycle(objects_meta_all)
        
        # valid_pts_selected = []
        # loop through objects
        successes = 0
        attempts = 0
        object_names = []
        view_ids = []
        while True:

            if args.randomize_scene_lighting_and_material:
                print("Randomizing lighting and materials")
                
                prob = np.random.uniform(0, 1)
                if prob>0.4:
                    # some augmentations
                    self.controller.step(
                        action="RandomizeMaterials",
                        useTrainMaterials=None,
                        useValMaterials=None,
                        useTestMaterials=None,
                        inRoomTypes=None
                    )
                
                prob = np.random.uniform(0, 1)
                if prob>0.5:
                    self.controller.step(
                        action="RandomizeLighting",
                        brightness=(0.5, 1.5),
                        randomizeColor=False,
                        hue=(0, 1),
                        saturation=(0.5, 1),
                        synchronized=False
                    )

            observations = {}

            if args.do_predict_oop:
                # select random object to target in view
                oop_ind = np.random.randint(len(oop_IDs))
                oop_ID_select = oop_IDs[oop_ind]
                obj = object_dict[oop_ID_select]['meta']
            else:
                obj = next(objects_meta_all_cycle)

            print("BATCH SIZE", args.data_batch_size)
            
            break_crit = successes >= args.data_batch_size
            if break_crit:
                break

            # here we want spawn agent near an object
            obj_category_name = obj['objectType']
            # print(obj_category_name)

            if obj_category_name not in self.name_to_id:
                continue

            print("obtaining batch ", successes) 

            # Calculate distance to object center
            obj_center = np.array(list(obj['axisAlignedBoundingBox']['center'].values()))
            # aithor_pos_camX = obj_center

            self.controller.step(action="GetReachablePositions")
            reachable = self.controller.last_event.metadata["actionReturn"]
            reachable = np.array([list(d.values()) for d in reachable])
            distances = np.sqrt(np.sum((reachable - np.expand_dims(obj_center, axis=0))**2, axis=1))
            map_pos = obj_center

            within_radius = np.logical_and(distances > args.radius_min, distances < args.radius_max)
            valid_pts = reachable[within_radius]

            found_one = self.find_starting_viewpoint(valid_pts, obj_center)
            if not found_one:
                continue

            movement_util = Movement_Util()

            targets_ = []
            rgb_scene_batch_ = []
            origin_T_camX_batch_ = []
            depth_batch_ = []
            # multiview_ = []
            fail = True
            views_obtained = 0
            for s in range(args.S+args.views_to_attempt):
                '''
                generate trajectory by taking random actions
                '''
                
                movement_util.move_agent(self.controller, mode=args.movement_mode)
                                
                obs = self.get_observations(
                    object_dict=object_dict, 
                    fail_if_no_objects=args.fail_if_no_objects, 
                    object_tracker=object_tracker,
                    trajectory_index=views_obtained,
                    batch_index=successes,
                    vis=vis
                    )

                if obs is None:
                    movement_util.movement_failed = True
                    movement_util.undo_previous_movement(self.controller)
                    continue # try again

                targets_.append(obs["target_frame"])
                rgb_scene_batch_.append(obs["rgb_torch"])

                views_obtained += 1

                if views_obtained==args.S:
                    fail = False
                    break

            # if vis is not None:
            #     vis.render_movie(args.movie_dir, 0, tag=f'bert_oop')
            #     st()

            if fail:
                print("failed. let's try again..")
                continue   
            if len(targets_)==0:
                print("failed. let's try again..")
                continue

            targets.extend(targets_)
            rgb_scene_batch.extend(rgb_scene_batch_)

            successes += 1
            attempts = 0

        if len(rgb_scene_batch)==0:
            return None

        rgb_scene_batch = torch.stack(rgb_scene_batch).float()
        
        outputs = [rgb_scene_batch, targets]

        return outputs 

    def find_starting_viewpoint(self, valid_pts, obj_center):
        '''
        Finds a "good" starting viewpoint based on object center in Aithor 3D coordinates and valid spawn points
        '''
        found_one = False
        if valid_pts.shape[0]==0:
            return found_one
        for i in range(10): # try 10 times to find a good location
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

    def get_observations(
        self, 
        object_dict=None, 
        verbose=False, 
        fail_if_no_objects=True, 
        object_tracker=None, 
        shuffle_inds=True,
        trajectory_index=None,
        batch_index=None,
        vis=None
        ):
        rgb = self.controller.last_event.frame
        depth = self.controller.last_event.depth_frame

        # plt.figure(1); plt.clf()
        # plt.imshow(rgb)
        # plt.savefig('images/test.png')
        # st()

        origin_T_camX = utils.aithor.get_origin_T_camX(self.controller.last_event, True)
        det_dict = object_tracker.update(
            rgb, 
            depth, 
            origin_T_camX, 
            return_features=True, 
            return_det_dict=True,
            vis=vis
            )
        

        out = check_for_detections_two_head(
            rgb, self.ddetr, self.W, self.H, 
            self.score_labels_name1, self.score_labels_name2, 
            score_threshold_head1=self.score_threshold, score_threshold_head2=self.score_threshold_oop, do_nms=True, target_object=None, target_object_score_threshold=None,
            solq=True, return_masks=True, nms_threshold=args.nms_threshold, id_to_mapped_id=self.id_to_mapped_id, return_features=True,
            )
        pred_scores, labels_det, boxes_det, masks_det, features = out["pred_scores2"], out["pred_labels"], out["pred_boxes"], out["pred_masks"], out["features"]
        # else:
        #     pred_scores, labels_det, boxes_det, features = det_dict["pred_scores"], det_dict["pred_labels"], det_dict["pred_boxes"], det_dict["features"] # not 'box' could be boxes or masks depending on args

        rgb_PIL = Image.fromarray(rgb)
        rgb_norm = rgb.astype(np.float32) * 1./255
        rgb_torch = torch.from_numpy(rgb_norm.copy()).permute(2, 0, 1)#.cuda() #to(self.device0) #.cuda()

        origin_T_camX = utils.aithor.get_origin_T_camX(self.controller.last_event, False)#.cuda()

        semantic = self.controller.last_event.instance_segmentation_frame
        object_id_to_color = self.controller.last_event.object_id_to_color
        color_to_object_id = self.controller.last_event.color_to_object_id
        
        obj_metadata_IDs = []
        for obj_m in self.controller.last_event.metadata['objects']: #objects:
            obj_metadata_IDs.append(obj_m['objectId'])

        instance_masks = self.controller.last_event.instance_masks
        instance_detections2d = self.controller.last_event.instance_detections2D
        obj_meta_all = self.controller.last_event.metadata['objects']
        masks = []
        bboxes = []
        labels = []
        centroids = []
        filter_inds = []
        visibilities = []

        obj_boxes = []
        obj_catnames = []
        obj_pred_boxes = []
        obj_pred_scores = []
        object_ids = []
        obj_segs = []
        # bert_token_ids_batch = []
        # bert_attention_masks_batch = []
        relations_batch = []
        relations_GT_batch = []
        ddetr_features = []
        obj_names = []
        object_ids2 = []

        oops = 0
        not_oops = 0

        # pred_scores, labels_det, boxes_det, features = check_for_detections(
        #     rgb, self.ddetr, self.W, self.H, 
        #     self.score_labels_name, self.score_boxes_name, 
        #     score_threshold_ddetr=self.score_threshold_ddetr, do_nms=True, return_features=True,
        #     )

        idxs = []
        for object_id in instance_masks.keys(): #range(obj_ids.shape[0]): 

            if object_id not in obj_metadata_IDs:
                continue

            idxs.append(object_id)  

        if shuffle_inds:
            random.shuffle(idxs)

        relations_to_add = []

        for object_id in idxs: 

            obj_meta_index = obj_metadata_IDs.index(object_id)
            obj_meta = obj_meta_all[obj_meta_index]

            obj_category_name = obj_meta['objectType']

            if obj_category_name not in self.name_to_id:
                continue

            obj_name = obj_meta['name']

            oop_label = self.label_to_id[object_dict[obj_name]['out_of_place']]
                    
            if not args.eval_test_set:
                if oop_label==1 and oops==args.num_each_oop:
                    continue

                if oop_label==0 and not_oops==args.num_each_oop:
                    continue

            i_mask = instance_masks[object_id]
            num_points = np.sum(i_mask)

            obj_bbox = instance_detections2d[object_id]

            # remove objects that are very small
            if num_points < (args.min_percent_points*self.W*self.H):
                continue
            
            obj_3dbox_origin = utils.aithor.get_3dbox_in_geom_format(obj_meta)
            # get amodal box
            boxlist2d_amodal, obj_3dbox_camX = utils.aithor.get_amodal2d(origin_T_camX.cuda(), obj_3dbox_origin.cuda(), torch.from_numpy(self.pix_T_camX).unsqueeze(0).cuda(), self.H, self.W)
            boxlist2d_amodal = boxlist2d_amodal.cpu().numpy()
            
            boxlist2d_amodal_clip = np.zeros(4)
            boxlist2d_amodal_clip[[0,2]] = np.clip(boxlist2d_amodal[[0,2]], 0, self.W)
            boxlist2d_amodal_clip[[1,3]] = np.clip(boxlist2d_amodal[[1,3]], 0, self.H)
            iou_inview = float(np.squeeze(utils.box.boxlist_2d_iou(boxlist2d_amodal.reshape(1,4), boxlist2d_amodal_clip.reshape(1,4))))

            if False: # plot boxes
                # clist_orgin = torch.mean(obj_3dbox_origin, axis=2)
                # clist_camX = utils.geom.apply_4x4(camX_T_origin, clist_orgin)
                # center_pix = list(utils.geom.apply_pix_T_cam(torch.from_numpy(self.pix_T_camX).unsqueeze(0), clist_camX).squeeze().cpu().numpy().astype(np.int32))
                # center_pix[1] = 256-center_pix[1]
                # center_pix = tuple(center_pix)

                # img = rgb.copy()
                # cv2.circle(img,center_pix, 2, (0,0,255), -1)
                # plt.imshow(img)
                # plt.savefig('images/test3.png')

                img = rgb.copy()
                cv2.rectangle(img, (int(boxlist2d_amodal[0]), int(boxlist2d_amodal[1])), (int(boxlist2d_amodal[2]), int(boxlist2d_amodal[3])),color=(0, 255, 0), thickness=1)
                cv2.rectangle(img, (int(obj_bbox[0]), int(obj_bbox[1])), (int(obj_bbox[2]), int(obj_bbox[3])),color=(255, 0, 0), thickness=1)
                plt.imshow(img)
                plt.savefig('images/test.png')

                rgb_box3d = self.summ_writer.summ_box_by_corners_aithor('test', rgb_torch.cpu().unsqueeze(0) - 0.5, obj_3dbox_camX.cpu(), torch.ones((1,1)), torch.ones((1,1)), torch.from_numpy(self.pix_T_camX).unsqueeze(0), only_return=True)
                rgb_box3d = (rgb_box3d + 0.5) *255. 
                rgb_box3d = rgb_box3d.cpu().squeeze(0).permute(1,2,0).numpy().astype(np.uint)
                plt.imshow(rgb_box3d)
                plt.savefig('images/test2.png')
                print("HERE")
                st()

            area_amodal = (boxlist2d_amodal_clip[2] - boxlist2d_amodal_clip[0]) * (boxlist2d_amodal_clip[3] - boxlist2d_amodal_clip[1])
            area_modal = (obj_bbox[2] - obj_bbox[0]) * (obj_bbox[3] - obj_bbox[1])
            percent_occluded = area_modal/area_amodal

            visibility = min(iou_inview, percent_occluded) # percent of object in frame, percent occluded

            # remove anything not visible enough
            if args.visibility_threshold is not None:
                if visibility < args.visibility_threshold:
                    continue

            if args.amodal:
                obj_bbox = boxlist2d_amodal_clip
            else:
                obj_bbox = obj_bbox # [start_x, start_y, end_x, end_y]    

            center_x = ((obj_bbox[0] + obj_bbox[2]) / 2) / self.W
            center_y = ((obj_bbox[1] + obj_bbox[3]) / 2) / self.H
            width = (obj_bbox[2] - obj_bbox[0]) / self.W
            height = (obj_bbox[3] - obj_bbox[1]) / self.H
            obj_bbox_coco_format = torch.from_numpy(np.array([center_x, center_y, width, height]))

            # if not args.use_gt_centroids_and_labels:
            # if True: 
            iou_det = utils.box.boxlist_2d_iou(boxlist2d_amodal.reshape(1,4), boxes_det)
            # iou_det_thresh = 0.3
            if not np.any(iou_det>args.iou_det_thresh):
                if args.eval_test_set:
                    if args.use_masks:
                        obj_segs.append(torch.from_numpy(i_mask))
                    obj_boxes.append(obj_bbox_coco_format)
                    obj_catnames.append(obj_category_name)
                    obj_names.append(obj_name)
                    visibilities.append(torch.tensor([visibility]))
                    if args.do_predict_oop:
                        object_ids.append(torch.tensor([self.name_to_id[obj_category_name]]))
                        object_ids2.append(torch.tensor([self.label_to_id[object_dict[obj_meta['name']]['out_of_place']]]))
                    else:
                        object_ids.append(torch.tensor([self.name_to_id[obj_category_name]]))
                continue
            matched_det_ind = np.argmax(iou_det)
            obj_cat_matched = self.id_to_name[labels_det[matched_det_ind]]
            bbox_det = boxes_det[matched_det_ind]
            mask_det = masks_det[matched_det_ind]
            pred_score = pred_scores[matched_det_ind]
            features_matched = features[matched_det_ind]


            encoded_dict, paragraph, relations = self.get_bert_input(
                                                    mask_det, 
                                                    obj_cat_matched,
                                                    depth, 
                                                    origin_T_camX, 
                                                    object_tracker
                                                    )
            
            centroid = np.array(list(obj_meta['axisAlignedBoundingBox']['center'].values()))

            encoded_dict_GT, paragraph_GT, relations_GT = self.get_bert_input_GT(
                                                    centroid,
                                                    obj_category_name,
                                                    depth, 
                                                    origin_T_camX, 
                                                    object_tracker
                                                )
                
            if relations is None or relations_GT is None:
                if args.eval_test_set:
                    # for eval we save everything
                    if args.use_masks:
                        obj_segs.append(torch.from_numpy(i_mask))
                    obj_boxes.append(obj_bbox_coco_format)
                    obj_catnames.append(obj_category_name)
                    obj_names.append(obj_name)
                    visibilities.append(torch.tensor([visibility]))
                    if args.do_predict_oop:
                        object_ids.append(torch.tensor([self.name_to_id[obj_category_name]]))
                        object_ids2.append(torch.tensor([self.label_to_id[object_dict[obj_meta['name']]['out_of_place']]]))
                    else:
                        object_ids.append(torch.tensor([self.name_to_id[obj_category_name]]))
                continue
            
            if args.finetune_on_one_object and oop_label==1:
                relations_to_add = relations

            if args.finetune_on_one_object:

                one_object_type_list = args.one_object_type.copy()
                one_supporting_object_type_list = args.one_supporting_object_type.copy()

                for obj_type_add_i in range(len(one_object_type_list)):
                    obj_add_rel = f'The {self.utils.format_class_name(one_object_type_list[obj_type_add_i])} is supported by the {self.utils.format_class_name(one_supporting_object_type_list[obj_type_add_i])}'
                    
                    if obj_add_rel in relations:
                        print("RELATION DETECTED.. CHANGING TO OOP")
                        oop_label = 1
                        break

            if args.use_masks:
                obj_segs.append(torch.from_numpy(i_mask))
            obj_boxes.append(obj_bbox_coco_format)
            obj_catnames.append(obj_category_name)
            obj_names.append(obj_name)
            visibilities.append(torch.tensor([visibility]))
            if args.do_predict_oop:
                object_ids.append(torch.tensor([self.name_to_id[obj_category_name]]))
                object_ids2.append(torch.tensor([self.label_to_id[object_dict[obj_meta['name']]['out_of_place']]]))
            else:
                object_ids.append(torch.tensor([self.name_to_id[obj_category_name]]))

            # bert_token_ids_batch.append(encoded_dict['input_ids'].squeeze())
            # # obj_labels_batch.append(label)
            # # object_ids.append(torch.tensor([oop_label]))
            # bert_attention_masks_batch.append(encoded_dict['attention_mask'].squeeze())
            # paragraphs_batch.append(paragraph)
            relations_batch.append(relations)
            relations_GT_batch.append(relations_GT)
            obj_pred_boxes.append(torch.from_numpy(bbox_det).float())
            obj_pred_scores.append(torch.tensor([pred_score]))
            # obj_boxes.append(torch.from_numpy(obj_bbox).float())
            # obj_catnames.append(obj_category_name)
            ddetr_features.append(features_matched)
            if oop_label==1:
                oops += 1
            elif oop_label==0:
                not_oops += 1
            else:
                assert(False)

        if args.eval_test_set:
            relations_batch = []
            obj_pred_boxes = []
            ddetr_features = []
            for matched_det_ind in range(len(boxes_det)):
                obj_cat_matched = self.id_to_name[labels_det[matched_det_ind]]
                bbox_det = boxes_det[matched_det_ind]
                mask_det = masks_det[matched_det_ind]
                pred_score = pred_scores[matched_det_ind]
                features_matched = features[matched_det_ind]


                encoded_dict, paragraph, relations = self.get_bert_input(
                                                        mask_det, 
                                                        obj_cat_matched,
                                                        depth, 
                                                        origin_T_camX, 
                                                        object_tracker
                                                        )
                if relations is None:
                    continue
                                                    
                relations_batch.append(relations)
                # relations_GT_batch.append(relations_GT)
                obj_pred_boxes.append(torch.from_numpy(bbox_det).float())
                obj_pred_scores.append(torch.tensor([pred_score]))
                # obj_boxes.append(torch.from_numpy(obj_bbox).float())
                # obj_catnames.append(obj_category_name)
                ddetr_features.append(features_matched)

        
        print("Num oop:", oops, "num not oop:", not_oops)
        
        print("Number of objects found:", len(object_ids))
        if fail_if_no_objects:
            if len(object_ids)==0 or len(relations_batch)==0: # want atleast one object
                if verbose:
                    print("No objects found")
                # fail = True
                return None
                # print("not enough objects")
                # continue
        if len(object_ids)>0:
            obj_boxes = torch.stack(obj_boxes).float() #.cuda()
            # object_ids = torch.cat(object_ids, dim=0).long() #.cuda()
            visibilities = torch.cat(visibilities, dim=0).float() #.cuda()
            obj_segs = torch.stack(obj_segs)
        else:
            return None

        if args.finetune_on_one_object:
            assert(args.do_language_only_oop) # must be langauge only
            one_object_type_list = args.one_object_type.copy() # list of objs
            one_supporting_object_type_list = args.one_supporting_object_type.copy() # list of objs
            for obj_type_add_i in range(args.num_add_per_iter):
                rand_int_choice = np.random.randint(len(one_object_type_list))
                one_object_type = one_object_type_list.pop(rand_int_choice)
                one_supporting_object_type = one_supporting_object_type_list.pop(rand_int_choice)
                inds_to_remove = None
                print(relations_to_add)
                for r_i in range(len(relations_to_add)):
                    relations_to_add[r_i] = f'The {self.utils.format_class_name(one_object_type)} is' + relations_to_add[r_i].split('is')[-1]
                    if 'supported' in relations_to_add[r_i]:
                        inds_to_remove = r_i
                if inds_to_remove is not None:
                    relations_to_add.pop(inds_to_remove)
                obj_add_rel = f'The {self.utils.format_class_name(one_object_type)} is supported by the {self.utils.format_class_name(one_supporting_object_type)}'
                relations_to_add.append(obj_add_rel)

                relations_to_add = np.array(relations_to_add)
                np.random.shuffle(relations_to_add)
                relations_to_add = list(relations_to_add)

                # sentences_formatted = []
                # for relation_sentence in relations_to_add:
                #     # sentence_full = ' '.join(word for word in relation_sentence)
                #     sentence_full = relation_sentence.replace('-', ' ')
                #     sentences_formatted.append(sentence_full)
                # paragraph = '. '.join(sentence for sentence in sentences_formatted)

                # encoded_dict = self.bert_tokenizer.encode_plus(
                #         paragraph,                      # paragraph to encode.
                #         add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                #         max_length = 512,           # Pad & truncate all sentences.
                #         pad_to_max_length = True,
                #         return_attention_mask = True,   # Construct attn. masks.
                #         return_tensors = 'pt',     # Return pytorch tensors.
                #     )
                # bert_token_ids_batch.append(encoded_dict['input_ids'].squeeze())
                object_ids2.append(torch.tensor([1])) # out of place label
                # bert_attention_masks_batch.append(encoded_dict['attention_mask'].squeeze())
                # paragraphs_batch.append(paragraph)
                relations_batch.append(relations_to_add)
                ddetr_features.append(torch.zeros_like(features_matched))
                obj_pred_boxes.append(torch.zeros_like(torch.from_numpy(bbox_det).float()))
        
        target_frame = {}
        target_frame['boxes'] = obj_boxes
        target_frame['visibility'] = visibilities
        if args.use_masks:
            obj_masks = obj_segs #.cuda()
            target_frame['masks'] = obj_masks
        
        obs = {}
        object_ids = torch.cat(object_ids, dim=0)
        object_ids2 = torch.cat(object_ids2, dim=0)
        # bert_token_ids_batch = torch.stack(bert_token_ids_batch)
        # bert_attention_masks_batch = torch.stack(bert_attention_masks_batch)
        ddetr_features_batch = torch.stack(ddetr_features)
        obj_pred_boxes = torch.stack(obj_pred_boxes)
        obj_pred_scores = torch.cat(obj_pred_scores)
        # target_frame = {}
        # target_frame['boxes'] = obj_boxes.float()
        target_frame['pred_boxes'] = obj_pred_boxes.float()
        target_frame['pred_scores'] = obj_pred_scores.float()
        agent_metadata = self.controller.last_event.metadata['agent']
        agent_metadata['scene'] = self.controller.scene
        agent_metadata['trajectory_index'] = trajectory_index
        agent_metadata['batch_index'] = batch_index
        target_frame['aithor_metadata'] = agent_metadata
        target_frame['labels'] = object_ids.long()
        target_frame['labels2'] = object_ids2.long()
        # target_frame['bert_token_ids'] = bert_token_ids_batch
        # target_frame['bert_attention_masks'] = bert_attention_masks_batch
        # target_frame['paragraph'] = paragraphs_batch
        # target_frame['relations'] = relations_batch
        # target_frame['relations_GT'] = relations_GT_batch
        target_frame['relations'] = {}
        target_frame['relations']['pred'] = relations_batch
        target_frame['relations']['GT'] = relations_GT_batch
        # target_frame['rgb'] = torch.from_numpy(rgb.copy()).unsqueeze(0)
        target_frame['ddetr_features'] = ddetr_features_batch

        obs["rgb_torch"] = rgb_torch
        obs["target_frame"] = target_frame

        return obs

    def get_bert_input(self, mask, obj_cat_target, depth, origin_T_camX, object_tracker):

        centroid = utils.aithor.get_centroid_from_detection_no_controller(
            mask, depth, 
            self.W, self.H, 
            centroid_mode='median', 
            pix_T_camX=self.pix_T_camX, 
            origin_T_camX=origin_T_camX
            )
        if centroid is None:
            return None, None, None

        tracker_centroids, tracker_labels = object_tracker.get_centroids_and_labels()
        if len(tracker_centroids)==0 or len(centroid)==0:
            return None, None, None
        dists = np.sqrt(np.sum((tracker_centroids - np.expand_dims(centroid, axis=0))**2, axis=1))
    
        dist_thresh = dists>self.dist_threshold
        tracker_centroids = tracker_centroids[dist_thresh]
        tracker_labels = list(np.array(tracker_labels)[dist_thresh])

        keep = np.array(tracker_labels)!=obj_cat_target #obj_cat_matched
        tracker_centroids = tracker_centroids[keep]
        tracker_labels = list(np.array(tracker_labels)[keep])

        floor_height = utils.aithor.get_floor_height(self.controller, self.W, self.H)

        # plt.figure(1); plt.clf()
        # plt.imshow(self.controller.last_event.frame)
        # plt.savefig('images/test2.png')
        try:
            relations = self.extract_relations_centroids(
                centroid, obj_cat_target, 
                tracker_centroids, tracker_labels, 
                floor_height=floor_height, pos_translator=None, overhead_map=None,
                )
        except:
            return None, None, None

        # get bert embeddings for each relation
        if len(relations) > self.max_relations:
            relations = random.sample(relations, self.max_relations)

        # # print(f"This episode has {len(relations)} relations.")
        # sentences_formatted = []
        # for relation_sentence in relations:
        #     # sentence_full = ' '.join(word for word in relation_sentence)
        #     sentence_full = relation_sentence.replace('-', ' ')
        #     sentences_formatted.append(sentence_full)
        # paragraph = '. '.join(sentence for sentence in sentences_formatted)

        # # print(paragraph)

        # # tokenize for bert
        # encoded_dict = self.bert_tokenizer.encode_plus(
        #         paragraph,                      # paragraph to encode.
        #         add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        #         max_length = 512,           # Pad & truncate all sentences.
        #         pad_to_max_length = True,
        #         return_attention_mask = True,   # Construct attn. masks.
        #         return_tensors = 'pt',     # Return pytorch tensors.
        #     )

        return None, None, relations

    def get_bert_input_GT(self, centroid, obj_cat_target, depth, origin_T_camX, object_tracker):

        # centroid = utils.aithor.get_centroid_from_detection_no_controller(
        #     mask, depth, 
        #     self.W, self.H, 
        #     centroid_mode='median', 
        #     pix_T_camX=self.pix_T_camX, 
        #     origin_T_camX=origin_T_camX
        #     )
        # if centroid is None:
        #     return None, None, None

        

        tracker_centroids, tracker_labels = object_tracker.get_objects_gt_from_meta(self.controller)
        dists = np.sqrt(np.sum((tracker_centroids - np.expand_dims(centroid, axis=0))**2, axis=1))
    
        dist_thresh = dists>self.dist_threshold
        tracker_centroids = tracker_centroids[dist_thresh]
        tracker_labels = list(np.array(tracker_labels)[dist_thresh])

        keep = np.array(tracker_labels)!=obj_cat_target #obj_cat_matched
        tracker_centroids = tracker_centroids[keep]
        tracker_labels = list(np.array(tracker_labels)[keep])

        floor_height = utils.aithor.get_floor_height(self.controller, self.W, self.H)

        # plt.figure(1); plt.clf()
        # plt.imshow(self.controller.last_event.frame)
        # plt.savefig('images/test2.png')
        try:
            relations = self.extract_relations_centroids(
                centroid, obj_cat_target, 
                tracker_centroids, tracker_labels, 
                floor_height=floor_height, pos_translator=None, overhead_map=None,
                )
        except:
            return None, None, None

        # get bert embeddings for each relation
        if len(relations) > self.max_relations:
            relations = random.sample(relations, self.max_relations)

        # # print(f"This episode has {len(relations)} relations.")
        # sentences_formatted = []
        # for relation_sentence in relations:
        #     # sentence_full = ' '.join(word for word in relation_sentence)
        #     sentence_full = relation_sentence.replace('-', ' ')
        #     sentences_formatted.append(sentence_full)
        # paragraph = '. '.join(sentence for sentence in sentences_formatted)

        # # print(paragraph)

        # # tokenize for bert
        # encoded_dict = self.bert_tokenizer.encode_plus(
        #         paragraph,                      # paragraph to encode.
        #         add_special_tokens = True, # Add '[CLS]' and '[SEP]'
        #         max_length = 512,           # Pad & truncate all sentences.
        #         pad_to_max_length = True,
        #         return_attention_mask = True,   # Construct attn. masks.
        #         return_tensors = 'pt',     # Return pytorch tensors.
        #     )

        return None, None, relations

    def extract_relations_centroids(self, centroid_target, label_target, obj_centroids, obj_labels, floor_height, pos_translator=None, overhead_map=None): 

        '''Extract relationships of interest from a list of objects'''

        obj_labels_np = np.array(obj_labels.copy())

        ################# Check Relationships #################
        # check pairwise relationships. this loop is order agnostic, since pairwise relationships are mostly invertible
        if args.visualize_relations:
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
                    if not obj_labels:
                        continue
                    yes_recept = []
                    for obj_label_i in obj_labels:
                        if obj_label_i in self.receptacles:
                            yes_recept.append(True)
                        else:
                            yes_recept.append(False)
                    yes_recept = np.array(yes_recept)
                    obj_centroids_ = obj_centroids[yes_recept]
                    obj_labels_ = list(obj_labels_np[yes_recept])
                    if len(obj_centroids_)==0 or len(centroid_target)==0:
                        continue
                    relation_ind = relation_fun(centroid_target, obj_centroids_, ground_plane_h=floor_height)
                    if relation_ind==-2:
                        pass
                    elif relation_ind==-1:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name('Floor')))
                        if args.visualize_relations:
                            relations_dict[relation].append(centroid_target)
                    else:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name(obj_labels_[relation_ind])))
                        if args.visualize_relations:
                            relations_dict[relation].append(obj_centroids_[relation_ind])
    
                else:
                    if len(obj_centroids)==0:
                        continue
                    relation_ind = relation_fun(centroid_target, obj_centroids)
                    if relation_ind==-2:
                        pass
                    elif relation_ind==-1:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name('Floor')))
                        if args.visualize_relations:
                            relations_dict[relation].append(centroid_target)
                    else:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name(obj_labels[relation_ind])))
                        if args.visualize_relations:
                            relations_dict[relation].append(obj_centroids[relation_ind])
            else:
                for i in range(len(obj_centroids)):

                    if len(obj_labels[i])==0:
                        continue

                    is_relation = relation_fun(centroid_target, obj_centroids[i])
                
                    if is_relation:
                        relations.append("The {0} is {1} the {2}".format(self.utils.format_class_name(label_target), relation.replace('-', ' '), self.utils.format_class_name(obj_labels[i])))
                        if args.visualize_relations:
                            relations_dict[relation].append(obj_centroids[i])

        if args.visualize_relations:
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


class Movement_Util():

    def __init__(self):   

        self.previous_movement = None
        self.movement_failed = False
        self.rotate_inc = 0.0
        self.move_inc = 0.0

    def undo_previous_movement(self, controller):
        if self.previous_movement is not None:
            print("Undoing movement")
            movements_undo = {"MoveAhead":"MoveBack", "RotateRight":"RotateLeft", "RotateLeft":"RotateRight", "LookUp":"LookDown", "LookDown":"LookUp"}
            movement = movements_undo[self.previous_movement]
            if "Rotate" in self.previous_movement:
                controller.step(
                    action=movement,
                    degrees=args.DT + self.rotate_inc
                )
            elif "Look" in self.previous_movement:
                controller.step(
                    action=movement,
                    degrees=args.HORIZON_DT + self.rotate_inc
                )
            else:
                controller.step(
                    action=movement,
                    moveMagnitude=args.STEP_SIZE + self.move_inc
                )

    def move_agent(self, controller, mode="forward_first"):

        if mode=="forward_first":
            movement = "MoveAhead"
            turn_right_or_left = np.random.randint(4)
            if turn_right_or_left==0:
                turn_dir = "RotateRight"
            elif turn_right_or_left==1:
                turn_dir = "RotateLeft"
            elif turn_right_or_left==2:
                turn_dir = "LookDown"
            else:
                turn_dir = "LookUp"
                
            position = np.array(list(controller.last_event.metadata["agent"]["position"].values()))
            rotation = np.array(list(controller.last_event.metadata["agent"]["rotation"].values()))

            # always try moving forward
            controller.step(
                action=movement,
                moveMagnitude=args.STEP_SIZE
            )
            
            position_ = np.array(list(controller.last_event.metadata["agent"]["position"].values()))
            rotation_ = np.array(list(controller.last_event.metadata["agent"]["rotation"].values()))

            # didn't move forward, then rotate
            if np.sum(position - position_)==0:
                if "Look" in turn_dir:
                    controller.step(
                        action=turn_dir,
                        degrees=args.HORIZON_DT
                    )
                else:
                    controller.step(
                        action=turn_dir,
                        degrees=args.DT
                    )
                self.previous_movement = turn_dir
            else:
                self.previous_movement = movement
            
            position_ = np.array(list(controller.last_event.metadata["agent"]["position"].values()))
            rotation_ = np.array(list(controller.last_event.metadata["agent"]["rotation"].values()))

            pos_diff = position_ - position 
            rot_diff = rotation_ - rotation

            # if sum(pos_diff) == 0 and sum(rot_diff) == 0:
            #     continue
            # print("pos_diff", pos_diff)
            # print("rot_diff", rot_diff)
        elif mode=="random":
            movements = ["MoveAhead", "RotateRight", "RotateLeft"]
            movement_idxs = np.arange(3)
            np.random.shuffle(movement_idxs)
            for movement_idx in list(movement_idxs):
                movement = movements[movement_idx]
                if "Rotate" in movement:
                    controller.step(
                        action=movement,
                        degrees=args.DT
                    )
                else:
                    controller.step(
                        action=movement,
                        moveMagnitude=args.STEP_SIZE
                    )
                if controller.last_event.metadata['lastActionSuccess']:
                    break
        elif mode=="random_with_momentum":
            
            p = np.random.uniform(0,1)
            if p > 0.5 and (self.previous_movement is not None) and (not self.movement_failed):
                movement = self.previous_movement
                if "Rotate" in movement:
                    rotate_inc_ = np.abs(np.random.normal(0, args.DT/4))
                    self.rotate_inc += rotate_inc_
                    controller.step(
                        action=movement,
                        degrees=args.DT + self.rotate_inc,
                    )
                elif "Look" in movement:
                    rotate_inc_ = np.abs(np.random.normal(0, args.HORIZON_DT/4))
                    self.rotate_inc += rotate_inc_
                    controller.step(
                        action=movement,
                        degrees=args.HORIZON_DT + self.rotate_inc,
                    )
                else:
                    move_inc_ = np.abs(np.random.normal(0, args.STEP_SIZE/4))
                    self.move_inc += move_inc_
                    controller.step(
                        action=movement,
                        moveMagnitude=args.STEP_SIZE + self.move_inc,
                    )

            else:
                movements = ["MoveAhead", "RotateRight", "RotateLeft", "LookUp", "LookDown"]
                movement_idxs = np.arange(len(movements))
                np.random.shuffle(movement_idxs)
                for movement_idx in list(movement_idxs):
                    movement = movements[movement_idx]
                    if "Rotate" in movement:
                        self.rotate_inc = np.abs(np.random.normal(0, args.DT/4))
                        controller.step(
                            action=movement,
                            degrees=args.DT + self.rotate_inc
                        )
                    elif "Look" in movement:
                        self.rotate_inc = np.abs(np.random.normal(0, args.HORIZON_DT/4))
                        controller.step(
                            action=movement,
                            degrees=args.HORIZON_DT + self.rotate_inc
                        )
                    else:
                        self.move_inc = np.abs(np.random.normal(0, args.STEP_SIZE/4))
                        controller.step(
                            action=movement,
                            moveMagnitude=args.STEP_SIZE + self.move_inc
                        )
                    if controller.last_event.metadata['lastActionSuccess']:
                        break
                self.previous_movement = movement
            self.movement_failed = False



        else:
            assert(False) # wrong movement mode

        event = controller.last_event
        return event