import os
from ai2thor.controller import Controller
import ipdb
st = ipdb.set_trace
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import time
import math
import pickle
import utils.improc
from utils.utils import *
import utils.geom
import pandas as pd
import sys
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F 
import torch.optim as optim
import re
import numpy as np
import torch
import sys
import numpy as np
import torch
from ai2thor.util.metrics import (
    get_shortest_path_to_object_type
)
import utils.basic
from backend import saverloader
from nets.visual_search_network import VSN
from arguments import args
from models.aithor_visualsearch_base import Ai2Thor_Base
from nets.solq import DDETR

def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out

def overwrite_eps(model, eps):
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.

    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps

class Ai2Thor(Ai2Thor_Base):

    def __init__(self):

        super(Ai2Thor, self).__init__()
        
        num_classes = len(self.include_classes) - 1 - len(self.special_classes) # remove no object class + special classes

        self.model = VSN(
            num_classes, #len(self.classes_to_save), 
            # self.rot_to_ind, self.hor_to_ind, self.ind_to_rot, self.ind_to_hor, 
            args.include_rgb, 
            self.fov,args.Z,args.Y,args.X,
            class_to_save_to_id=self.class_to_save_to_id,
            do_masked_pos_loss=args.do_masked_pos_loss, 
            )
        self.model.cuda()

        self.start_step = 1

        # lr = 1e-4
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=args.lr_vsn)

        self.lr_scheduler = None

        if args.load_model:
            self.start_step =saverloader.load_from_path(args.load_model_path, self.model, self.optimizer, strict=(not args.load_strict_false), lr_scheduler=self.lr_scheduler)

        for name, param in self.model.named_parameters():
            print(name, "requires grad?", param.requires_grad)

        self.max_iters = args.max_iters
        self.log_freq = args.log_freq

        # self.run_episodes()

    def run_episodes(self):

        self.ep_idx = 0

        if args.eval_object_nav:
            self.evaluate_object_goal_navigation()
            return 
        
        for iter_ in range(self.start_step, self.max_iters):

            total_loss = None

            print("Begin iter", iter_)
            print("set name:", args.set_name)

            self.summ_writer = utils.improc.Summ_writer(
                writer=self.writer,
                global_step=iter_,
                log_freq=self.log_freq,
                fps=8,
                just_gif=False)

            self.iter_ = iter_
            
            total_loss = self.run_train()

            if total_loss is not None:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            print(f"loss for iter {iter_} is: {total_loss}")

            # val 
            if args.run_val:
                if iter_ % args.val_freq == 0:
                    with torch.no_grad():
                        self.run_val()

                    self.writer.flush()

            if self.lr_scheduler is not None:
                if iter_ % args.lr_scheduler_freq== 0:
                    self.lr_scheduler.step()

            if iter_ % args.save_freq == 0:
                saverloader.save_checkpoint(self.model, self.checkpoint_path, iter_, self.optimizer, keep_latest=args.keep_latest, lr_scheduler=self.lr_scheduler)
                
        self.controller.stop()
        time.sleep(1)

    def run_train(self):

        total_loss = torch.tensor(0.0).cuda() #to(self.device0) #.cuda()

        mapnames = []
        for batch in range(args.scenes_per_batch):


            if self.random_select: # select random house set 
                rand = np.random.randint(len(self.mapnames_train))
                mapname = self.mapnames_train[rand]
            else:
                house_select = next(house_iter)

            print("MAPNAME=", mapname)
            mapnames.append(mapname)

        n = np.random.randint(args.n_train_mapping_obs)
        input_dict = self.load_agent_output(n, mapnames, mode='train')

        if self.summ_writer.save_this:
            do_inference = True
            do_inference_gt = True
        else:
            do_inference = False  
            do_inference_gt = False

        forward_dict = self.model(
            input_dict["rgb_batch"], input_dict["xyz_batch"], input_dict["camX0_T_camX_batch"], 
            args.Z, args.Y, args.X, 
            input_dict["vox_util_batch"],input_dict["targets_batch"], 
            mode='train',
            summ_writer=self.summ_writer,
            do_inference=do_inference,
            objects_track_dict_batch=input_dict["objects_track_dict_batch"],
            )
        loss = forward_dict['total_loss']
        total_loss += loss

        self.summ_writer.summ_scalar('train/total_loss', total_loss)

        if do_inference:
            self.run_inference(
                forward_dict,
                input_dict["targets_batch"], input_dict["vox_util_batch"],
                input_dict["nav_pts_batch"], input_dict["camX0_T_camX_batch"],
                mapnames,
                input_dict["classes_order"],
                mode='train'
                )

        return total_loss

    def load_agent_output(self, n_obs, mapnames, objects_per_scene=None, mode='train'):
        '''
        Prepares input batch for training
        '''

        if objects_per_scene is None:
            objects_per_scene = args.objects_per_scene

        targets_batch = []
        vox_util_batch = []
        rgb_batch = []
        xyz_batch = []
        origin_T_camX_batch = []
        origin_T_camX0_batch = []
        objects_track_dict_batch = []
        nav_pts_batch = []
        classes_order = []
        for episode in range(len(mapnames)):

            mapname = mapnames[episode]

            self.controller.reset(scene=mapname)
            scene_centroid = {'x': 0., 'y': 0., 'z': 0.} #self.controller.last_event.metadata["sceneBounds"]["center"]

            # (1) Load mapping phase images (can generate this with mode='generate_mapping_obs')
            root = os.path.join(args.mapping_obs_dir, mapname) 
            pickle_fname = f'{n_obs}.p'
            fname = os.path.join(root, pickle_fname)
            with open(fname, 'rb') as f:
                obs_dict = pickle.load(f)

            # (2) Preprocess mapping observations
            rgb = torch.from_numpy(obs_dict['rgb']).cuda().float()
            xyz = torch.from_numpy(obs_dict['xyz']).cuda().float()
            origin_T_camX = torch.from_numpy(obs_dict['camX0_T_camX']).cuda().float()
            origin_T_camX0_batch.append(torch.from_numpy(obs_dict['origin_T_camX0']).cuda().float())
            camX0_candidate = 0 #np.random.choice(obs_dict['camX0_candidates'])

            inds_all = np.arange(rgb.shape[0])
            inds_all = np.delete(inds_all, camX0_candidate)
            
            num_sample = args.num_views_mapping_sample-1 #20 - 1
            if inds_all.shape[0]<num_sample:
                inds = inds_all
            else:
                inds = np.random.choice(inds_all, size=num_sample, replace=False)

            inds = np.insert(inds, 0, camX0_candidate, axis=0)
            
            rgb = rgb[inds].cuda()
            xyz = xyz[inds].cuda()
            origin_T_camX = origin_T_camX[inds].cuda()
            # yaw_camX0 = obs_dict['yaw'][inds[0]]
            # yaw = obs_dict['yaw'][inds]
            # pitch = obs_dict['pitch'][inds]

            if args.do_add_semantic:
                objects_track_dict = obs_dict['objects_track_dict']
            else:
                objects_track_dict = None

            rgb_batch.append(rgb)
            xyz_batch.append(xyz)
            origin_T_camX_batch.append(origin_T_camX)
            objects_track_dict_batch.append(objects_track_dict)

            # (3) next, get objects in the room to target as supervision
            objects = self.controller.last_event.metadata['objects']
            object_classes = list([objects[i]['objectType'] for i in range(len(objects))])
            inds_shuffle = np.arange(len(object_classes))
            np.random.shuffle(inds_shuffle)

            if args.keep_target_aithor_ref:
                # pass
                scene_centroid = torch.tensor(list(scene_centroid.values())).view(1,1,3).cuda()
            else:
                # convert from aithor origin to new camX0
                camX0_T_camX = origin_T_camX[0]
                origin_T_camX0 = torch.from_numpy(obs_dict['origin_T_camX0']).cuda().float()
                origin_T_camX0_new = torch.matmul(origin_T_camX0, camX0_T_camX) #camX is new camX0
                camX0_new_T_origin = utils.geom.safe_inverse_single(origin_T_camX0_new)

                # # convert scene centroid froma aithor to new camX0
                # scene_centroid = torch.tensor(list(scene_centroid.values())).view(1,1,3).cuda()
                # scene_centroid = utils.geom.apply_4x4(camX0_new_T_origin.unsqueeze(0), scene_centroid.float())
                # #.squeeze(1)
                # scene_centroid[:,1] = -scene_centroid[:,1]  

            obj_info_all = {}
            obj_obtained = 0
            for obj_i in range(len(inds_shuffle)):
                obj_cur = object_classes[inds_shuffle[obj_i]]
                if obj_cur not in self.classes_to_save:
                    continue

                print(f"Targeted {obj_cur} for supervision")

                # if args.predict_obj_locations:
                obj_meta = objects[inds_shuffle[obj_i]]
                obj_center = torch.from_numpy(np.array(list(obj_meta['axisAlignedBoundingBox']['center'].values()))).unsqueeze(0)
                obj_corners = torch.from_numpy(np.array(list(obj_meta['axisAlignedBoundingBox']['cornerPoints'])))
                obj_position = torch.cat([obj_center, obj_corners], dim=0)
                if args.keep_target_aithor_ref:
                    pass
                else:
                    # convert to new reference frame
                    obj_position = utils.geom.apply_4x4(camX0_new_T_origin.unsqueeze(0), obj_position.unsqueeze(1).cuda().float()).squeeze(1)
                    obj_position[:,1] = -obj_position[:,1]
                    # camX0_T_origin = utils.geom.safe_inverse_single(origin_T_camX[0])
                    # xyz_pos_camX0 = utils.geom.apply_4x4(camX0_T_origin, obj_position.unsqueeze(0))
                    # obj_same = []
                    # for id_ in objects_track_dict.keys():
                    #     if objects_track_dict[id_]['label']==obj_cur:
                    #         obj_same.append(objects_track_dict[id_]['locs'])
                    # st()
                      
                    # plt.figure()
                    # plt.imshow(self.controller.last_event.frame)
                    # plt.savefig('images/test.png') 

                obj_info_all[obj_i] = {'obj_position':obj_position}

                # else:
                #     assert(False) # what are we predicting?

                obj_info_all[obj_i]['obj_id'] = self.class_to_save_to_id[obj_cur]
                classes_order.append(obj_cur)
                obj_obtained += 1
                if obj_obtained==objects_per_scene:
                    break

            if mode=='train':
                add_noise=True
            else:
                add_noise=False
            
            # prepare supervision
            targets, vox_util = self.model.prepare_supervision(
                obj_info_all,
                origin_T_camX,
                xyz,
                scene_centroid,
                # yaw_camX0,
                args.Z,args.Y,args.X,
                add_noise=add_noise,
                )
            targets_batch.append(targets)
            vox_util_batch.append(vox_util)

            self.controller.reset(scene=mapname)

            event = self.controller.step(action="GetReachablePositions")
            nav_pts = event.metadata["actionReturn"]
            nav_pts = np.array([list(d.values()) for d in nav_pts])
            nav_pts_batch.append(nav_pts)

        input_dict = {}
        input_dict["rgb_batch"] = rgb_batch
        input_dict["xyz_batch"] = xyz_batch
        input_dict["camX0_T_camX_batch"] = origin_T_camX_batch
        input_dict["origin_T_camX0_batch"] = origin_T_camX0_batch
        input_dict["vox_util_batch"] = vox_util_batch
        input_dict["targets_batch"] = targets_batch
        input_dict["nav_pts_batch"] = nav_pts_batch
        input_dict["objects_track_dict_batch"] = objects_track_dict_batch
        input_dict["classes_order"] = classes_order

        return input_dict

    def run_val(self):

        self.model.eval()

        total_loss = torch.tensor(0.0).cuda() #to(self.device0) #.cuda()

        mapnames = self.mapnames_val[:args.n_val]

        n = np.random.randint(args.n_val_mapping_obs)
        input_dict = self.load_agent_output(n, mapnames, objects_per_scene=args.objects_per_scene_val, mode='val')

        if self.summ_writer.save_this:
            do_inference = True
            do_inference_gt = True
        else:
            do_inference = False  
            do_inference_gt = False

        forward_dict = self.model(
            input_dict["rgb_batch"], input_dict["xyz_batch"], input_dict["camX0_T_camX_batch"], 
            args.Z, args.Y, args.X, 
            input_dict["vox_util_batch"],input_dict["targets_batch"], 
            summ_writer=self.summ_writer,
            do_inference=do_inference,
            objects_track_dict_batch=input_dict["objects_track_dict_batch"],
            )
        loss = forward_dict['total_loss']
        total_loss += loss

        self.summ_writer.summ_scalar('val/total_loss', total_loss)

        if do_inference:
            self.run_inference(
                forward_dict,
                input_dict["targets_batch"], input_dict["vox_util_batch"],
                input_dict["nav_pts_batch"], input_dict["camX0_T_camX_batch"],
                mapnames,
                input_dict["classes_order"],
                mode='val'
                )

        self.model.train()

        return total_loss
        
    def run_inference(
        self, forward_dict,
        targets_batch, vox_util_batch,
        nav_pts_batch, origin_T_camX_batch,
        mapnames,
        classes_order, max_positions=1, mode='train',
        do_inference_gt=False
    ):
        # perform inference
        if do_inference_gt:
            inference_dict = self.model.inference_from_gt(
            forward_dict['feat_memX0'],
            forward_dict['feat_pos_logits'],
            targets_batch,
            self.Z,self.Y,self.X,
            vox_util_batch,
            nav_pts_batch,
            origin_T_camX_batch,
            classes_order,
            summ_writer=self.summ_writer
            )
        else:
            inference_dict = self.model.inference(
            forward_dict['feat_memX0'],
            forward_dict['feat_pos_logits'],
            targets_batch,
            self.Z,self.Y,self.X,
            vox_util_batch,
            nav_pts_batch,
            origin_T_camX_batch,
            classes_order,
            max_positions=max_positions,
            summ_writer=self.summ_writer
            )

        if inference_dict is not None:
            
            pos_targets = forward_dict['pos_targets']
            if not do_inference_gt:
                pos_map_pred = inference_dict['feat_mem_thresh'].unsqueeze(1)
                pos_map_sigmoid = inference_dict['feat_mem_logits'].unsqueeze(1)
            # pos_map_logits = utils.basic.normalize(pos_map_logits)
            b_inds_pred = inference_dict['batches']
            b_classes = inference_dict['classes']
            keep_inds = inference_dict['succeses']
            class_n = inference_dict['class_n']
            if do_inference_gt:
                rgbs_gt = inference_dict['rgbs']

            if not do_inference_gt:

                # plot position maps 
                pos_map_vis = torch.cat([utils.basic.normalize(pos_map_sigmoid), utils.basic.normalize(pos_map_pred), utils.basic.normalize(pos_targets[keep_inds])], dim=3)
                occ_camX0s_vis = forward_dict['occ_camX0s_vis'][keep_inds]

                # sds = []
                for v in range(len(pos_map_vis)):
                    pos_map_vis_v = pos_map_vis[v:v+1]
                    ind_v = b_inds_pred[v]
                    class_n_cur = class_n[v]

                    if False:
                        from utils.wctb import ThorPositionTo2DFrameTranslator
                        mapname = mapnames[ind_v]
                        self.controller.reset(scene=mapname)
                        self.controller.step({"action": "ToggleMapView"})
                        cam_position = self.controller.last_event.metadata["cameraPosition"]
                        cam_orth_size = self.controller.last_event.metadata["cameraOrthSize"]
                        pos_translator = ThorPositionTo2DFrameTranslator(
                            self.controller.last_event.frame.shape, (cam_position["x"], cam_position["y"], cam_position["z"]), cam_orth_size
                        )
                        overhead_map = self.controller.last_event.frame
                        overhead_map = torch.from_numpy(overhead_map.copy()).cuda().permute(2,0,1).unsqueeze(0)
                        name = f'{mode}_inference/overheadmap_{b_classes[v]}_(right=gt)_{v}'
                        self.summ_writer.summ_rgb(name, overhead_map)

                    # name = f'{mode}_inference/position_{mapnames[ind_v]}_{b_classes[v]}_(right=gt)'
                    name = f'{mode}_inference/position_{b_classes[v]}_(right=gt)_{v}'
                    occ_camX0s_vis_v = torch.cat([occ_camX0s_vis[v], occ_camX0s_vis[v], occ_camX0s_vis[v]], dim=2)
                    if False:
                        vis = self.summ_writer.summ_oned(name=name, im=pos_map_vis_v, norm=False, only_return=True, heatmap=True, overlay_image=occ_camX0s_vis_v)
                        plt.figure(1);plt.clf()
                        plt.imshow(vis[0].permute(1,2,0).cpu().numpy())
                        plt.savefig(f'{args.movie_dir}/vsn_vis_{b_classes[v]}.png')
                    self.summ_writer.summ_oned(name=name, im=pos_map_vis_v, norm=False, only_return=False, heatmap=True, overlay_image=occ_camX0s_vis_v)


    def evaluate_object_goal_navigation(self):

        self.model.eval()

        results = {}

        self.mode = args.object_navigation_policy_name # "vsn_search" "random"

        self.dist_thresh = args.dist_thresh

        # if self.mode=="vsn_search":
        # init detector 
        if args.do_predict_oop:
            # use two head detector
            num_classes = len(self.include_classes) 
            self.label_to_id = {False:0, True:1} # not out of place = False
            self.id_to_label = {0:'IP', 1:'OOP'}
            num_classes2 = 2 # 0 oop + 1 not oop + 2 no object
            # for inference
            self.score_boxes_name = 'pred1'
            self.score_labels_name = 'pred1' # semantic labels
            self.score_labels_name1 = 'pred1' # semantic labels
            self.score_labels_name2 = 'pred2' # which prediction head has the OOP label?
        else:
            num_classes = len(self.include_classes) 
            num_classes2 = None
            self.score_boxes_name = 'pred1' # only one prediction head so same for both
            self.score_labels_name = 'pred1'

        # self.score_threshold_oop = args.score_threshold_oop
        # self.score_threshold = args.confidence_threshold

        load_pretrained = False
        self.ddetr = DDETR(num_classes, load_pretrained, num_classes2=num_classes2).cuda()

        _ = saverloader.load_from_path(args.SOLQ_oop_checkpoint, self.ddetr, None, strict=True)
        self.ddetr.eval()

        fname = f'./metrics/object_nav_{self.mode}_test_{args.tag}.p'

        self.num_search_locs = args.num_search_locs_object # number of search locations to allow

        if args.load_submission:
            if os.path.isfile(fname):
                print("loading", fname)
                with open(fname, 'rb') as f:
                    results = pickle.load(f)
        
        for episode in range(len(self.mapnames_test)):

            # if episode==0:
            #     continue

            mapname = self.mapnames_test[episode]

            if mapname in results.keys(): # already exists
                continue

            self.controller.reset(scene=mapname)

            print("MAPNAME=", mapname)
            with torch.no_grad():
                time_steps, successes, object_types = self.run_eval_object_goal_nav(mapname=mapname)

            print("TIME STEPS:", time_steps)
            print("successful_path:", successes)

            results[mapname] = {}
            results[mapname]["time_steps"] = time_steps
            results[mapname]["successes"] = successes
            results[mapname]["object_types"] = object_types
            
            print("saving", fname)
            with open(fname, 'wb') as f:
                pickle.dump(results, f, protocol=4)

        successes = []
        time_steps = []
        # get total metrics
        for episode in range(len(self.mapnames_test)):
            mapname = self.mapnames_test[episode]
            successes.append(results[mapname]["successes"])
            time_steps.append(results[mapname]["time_steps"])
        successes = np.concatenate(successes)
        time_steps = np.concatenate(time_steps)
        success_rate = np.sum(successes)/len(successes)
        avg_time_steps = np.mean(time_steps)
        print("############# FINAL METRICS ###############")
        print(f"Success rate: {success_rate}")
        print(f"Average time steps: {avg_time_steps}")
        print("###########################################")
        st()
            

if __name__ == '__main__':
    Ai2Thor()
