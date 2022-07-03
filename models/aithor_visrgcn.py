import os
from ai2thor.controller import Controller
import ipdb
st = ipdb.set_trace
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import pickle
# from nets.ddetr import DDETR
from nets.solq import DDETR
import sys
import SOLQ.util.misc as ddetr_utils
from SOLQ.util import box_ops
import time
import math
import pickle
from scipy.spatial.transform import Rotation as R
from utils.wctb import Utils, Relations
import utils.improc
import itertools
import copy
from PIL import Image, ImageDraw
from utils.wctb import ThorPositionTo2DFrameTranslator
import utils.aithor
import torch
from sklearn.metrics import accuracy_score
from nets.visrgcn_mem_cl import RGCN
from models.aithor_visrgcn_base import Ai2Thor_Base
from backend import saverloader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F 
from arguments import args
import re
from tensorboardX import SummaryWriter
from SOLQ.util.misc import nested_tensor_from_tensor_list

class Ai2Thor(Ai2Thor_Base):
    def __init__(self):   

        super(Ai2Thor, self).__init__()

        self.maptype_to_id = {"kitchen":0, "living_room":1, "bedroom":2, "bathroom":3}
        
        num_classes = len(self.include_classes_floor) 
        num_relations = len(self.relations_executors_pairs)

        layer_sizes = [512, 1024, 2048] # ddetr backbone output features sizes per layer index
        visual_feat_size = layer_sizes[args.backbone_layer_ddetr]
        self.visual_feat_size = visual_feat_size

        in_channels = args.in_channels #+ visual_feat_size
        # out_channels = args.out_channels

        self.model = RGCN(in_channels, num_classes, num_relations, use_memory=True, id_to_name=self.id_to_name_floor, name_to_id=self.name_to_id_floor, visual_feat_size=visual_feat_size)
        self.model.cuda().train() #to(self.device0) #.cpu()      

        params_to_optimize = self.model.parameters()
        self.optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr_vrgcn,
                                      weight_decay=args.weight_decay_vrgcn)
        # self.optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

        self.lr_scheduler = None

        self.start_step  = 1

        if args.load_model:
            self.start_step =saverloader.load_from_path(args.load_model_path, self.model, self.optimizer, strict=(not args.load_strict_false), lr_scheduler=None)

        # if args.do_visual_features:
        if args.do_predict_oop:
            num_classes = len(self.include_classes) 
            self.label_to_id = {False:0, True:1} # not out of place = False
            self.id_to_label = {0:'IP', 1:'OOP'}
            num_classes2 = 2 # 0 oop + 1 not oop + 2 no object
            self.score_boxes_name = 'pred1'
            self.score_labels_name = 'pred1'
            self.score_labels_name2 = 'pred2' # which prediction head has the OOP label?
        else:
            num_classes = len(self.include_classes) 
            num_classes2 = None
            self.score_boxes_name = 'pred1' # only one prediction head so same for both
            self.score_labels_name = 'pred1'
        self.ddetr = DDETR(num_classes, load_pretrained=False, num_classes2=num_classes2)
        self.ddetr.cuda()
        print("loading checkpoint for solq...")
        print(f"loading {args.SOLQ_oop_checkpoint}")
        ddetr_path = args.SOLQ_oop_checkpoint 
        _ = saverloader.load_from_path(ddetr_path, self.ddetr, None, strict=(not args.load_strict_false), lr_scheduler=None)
        print("loaded!")
        self.ddetr.eval()

        if args.start_one:
            self.start_step = 1

        self.max_iters = args.max_iters
        self.log_freq = args.log_freq

        self.mem_dict = None

        # self.run_episodes()

    def run_episodes(self):

        if args.do_save_oop_nodes_and_supervision:
            self.save_relations()
            return 

        # # obtain memory from memory houses
        # self.get_memory()
        
        for iter_ in range(self.start_step, self.max_iters):

            print("Begin iter", iter_)

            self.summ_writer = utils.improc.Summ_writer(
                writer=self.writer,
                global_step=iter_,
                log_freq=self.log_freq,
                fps=8,
                just_gif=True)
            
            total_loss = self.run_train()

            if total_loss is not None:
                self.optimizer.zero_grad()
                total_loss.backward()
                if args.clip_max_norm > 0:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_max_norm)
                # else:
                #     grad_total_norm = ddetr_utils.get_total_grad_norm(self.model.parameters(), args.clip_max_norm)
                self.optimizer.step()

            print(f"loss for iter {iter_} is: {total_loss}")

            if args.run_val:
                if iter_ % args.val_freq == 0:
                    with torch.no_grad():
                        self.run_val()

                    self.writer.flush()
            
            if self.lr_scheduler is not None:
                if iter_ % args.lr_scheduler_freq==0:
                    self.lr_scheduler.step()

            if iter_ % args.save_freq == 0:
                saverloader.save_checkpoint(self.model, self.checkpoint_path, iter_, self.optimizer, keep_latest=args.keep_latest, lr_scheduler=self.lr_scheduler)

        self.controller.stop()
        time.sleep(1)

    def save_relations(self):
        '''
        Save out relations to load in later
        '''
        for episode in range(len(self.mapnames_train)):
            print("STARTING EPISODE ", episode)

            mapname = self.mapnames_train[episode]

            print("MAPNAME=", mapname)

            self.controller.reset(scene=mapname)

            self.save_oop_nodes_and_supervision(mapname)

            time.sleep(1)

        for episode in range(len(self.mapnames_val)):
            print("STARTING EPISODE ", episode)

            mapname = self.mapnames_val[episode]

            print("MAPNAME=", mapname)

            self.controller.reset(scene=mapname)

            self.save_oop_nodes_and_supervision(mapname)

            time.sleep(1)

    def get_memory(self):

        if args.load_visual_memex:
            # filename = 'memory_features.p'
            # load_path = os.path.join(args.load_dir, filename)
            if not os.path.exists(args.visual_memex_path):
                print("Could not locate visual memex at specified location! Will now generate it and save at the specified path....\n")
                print("If using pretrained models, please download the memex using the instructions on Github.\n")
            else:
                with open(args.visual_memex_path, 'rb') as handle:
                    self.mem_dict = pickle.load(handle)
                # self.mem_dict = np.load(load_path, allow_pickle=True)
                return 
        
        subj_pred_obj_adj_scene_batch = []
        obj_ids_scene_batch = []
        obj_feats_scene_batch = []
        for episode in range(len(self.mapnames_mem)):
            print("STARTING EPISODE ", episode)

            mapname = self.mapnames_mem[episode]

            print("MAPNAME=", mapname)

            if mapname in ["FloorPlan2"]: # this one fails
                continue

            self.controller.reset(scene=mapname)

            out = self.get_memex_relations()
            subj_pred_obj_adj_scene_batch.append(out[0])
            obj_ids_scene_batch.append(out[1])
            obj_feats_scene_batch.append(out[2])
            
            time.sleep(1)

        mem_dict = self.get_relation_dict(subj_pred_obj_adj_scene_batch, obj_ids_scene_batch)
        mem_dict['obj_feats'] = torch.cat(obj_feats_scene_batch, dim=0)

        self.mem_dict = mem_dict 

        if args.load_visual_memex:
            print(f"Saving memex to {args.visual_memex_path}...")
            with open(args.visual_memex_path, 'wb') as handle:
                pickle.dump(self.mem_dict, handle, protocol=4)
            print("Done.")
    
    def get_relation_dict(self, subj_pred_obj_adj_scene_batch, obj_ids_scene_batch):

        obj_lens_batch = np.array([subj_pred_obj_adj_scene_batch[i].shape[0] for i in range(len(subj_pred_obj_adj_scene_batch))])

        total_len_obj_batch = sum(obj_lens_batch)
        subj_pred_obj_adj_scene_batch_ = torch.zeros((total_len_obj_batch, len(self.rel_to_id), total_len_obj_batch))

        obj_ids_scene_batch = torch.cat(obj_ids_scene_batch, dim=0).cuda()

        prev_obj_i = 0
        for len_i in range(len(obj_lens_batch)):
            obj_lens_batch_i = np.cumsum(obj_lens_batch)[len_i]
            subj_pred_obj_adj_scene_batch_[prev_obj_i:obj_lens_batch_i, :, prev_obj_i:obj_lens_batch_i] = subj_pred_obj_adj_scene_batch[len_i]
            prev_obj_i = obj_lens_batch_i

        subj_pred_obj_adj_scene_batch = subj_pred_obj_adj_scene_batch_.cuda()

        where_adj = torch.where(subj_pred_obj_adj_scene_batch)

        subj_obj_inds_sparse_scene_batch = torch.stack([where_adj[0], where_adj[2]])
        rel_inds_sparse_scene_batch = where_adj[1]

        batch_inds = torch.zeros(len(rel_inds_sparse_scene_batch))
        ind_p = 0
        for len_i in range(len(obj_lens_batch)):
            ind_c = np.cumsum(obj_lens_batch)[len_i]
            where_inds = torch.where(torch.logical_and(subj_obj_inds_sparse_scene_batch[0] >= ind_p, subj_obj_inds_sparse_scene_batch[0] < ind_c))
            batch_inds[where_inds] = len_i*torch.ones(len(where_inds))
            ind_p = ind_c
        
        out_dict = {}
        out_dict['obj_ids'] = obj_ids_scene_batch
        out_dict['subj_obj_inds'] = subj_obj_inds_sparse_scene_batch.long().T
        out_dict['rel_inds'] = rel_inds_sparse_scene_batch.long()
        out_dict['batch_inds'] = batch_inds.long()
        out_dict['obj_lens'] = torch.from_numpy(obj_lens_batch)

        return out_dict

    def load_agent_output(self, n_obs, mapnames):

        # first make sure we have memory MEMEX - stays fixed across all maps
        if self.mem_dict is None:
            self.get_memory()

        # loop over maps to get batch
        subj_pred_obj_adj_scene_batch = []
        obj_ids_scene_batch = []
        obj_feats_scene_batch = []
        map_types = []
        ddetr_batch_inds = []
        features_ddetr = []
        ddetr_labels = []
        for episode in range(len(mapnames)):

            mapname = mapnames[episode]
            print(f"Map: {mapname}")

            # Then, get out of place object batch
            out = self.get_oop_nodes(mapname)
            subj_pred_obj_adj_scene_batch.extend(out[0])
            obj_ids_scene_batch.extend(out[1])
            obj_feats_scene_batch.extend(out[2])

            # finally, get scene graph
            # (1) Load mapping phase images (can generate this with mode='generate_mapping_obs')
            root = os.path.join(args.mapping_obs_dir, mapname) 
            pickle_fname = f'{n_obs}.p'
            fname = os.path.join(root, pickle_fname)
            with open(fname, 'rb') as f:
                obs_dict = pickle.load(f)

            # (2) Process mapping images into a scene graph of visual features + labels
            map_type, features_ddetr_, labels_ = self.get_scene_graph(mapname, obs_dict)
            if features_ddetr_ is None:
                print(f"No objects found in scene graph for map={mapname} and n={n_obs}")
                continue
            map_types.append(map_type)
            ddetr_labels.append(labels_)
            features_ddetr.append(features_ddetr_)
            ddetr_batch_inds.append(torch.ones(len(features_ddetr_))*episode)

        # construct input dictionary
        input_dict = self.get_relation_dict(subj_pred_obj_adj_scene_batch, obj_ids_scene_batch)
        input_dict['map_types'] = torch.tensor([self.maptype_to_id[map_types_i] for map_types_i in map_types]).cuda().long()
        input_dict['features_ddetr'] = torch.cat(features_ddetr, dim=0)
        input_dict['ddetr_batch_inds'] = torch.cat(ddetr_batch_inds, dim=0)
        input_dict['ddetr_labels'] = torch.cat(ddetr_labels, dim=0)
        input_dict['obj_feats'] = torch.cat(obj_feats_scene_batch, dim=0)
        
        return input_dict

    def run_train(self):
        total_loss = torch.tensor(0.0).cuda() #to(self.device0) #.cuda()

        mapnames = []
        for episode in range(args.scenes_per_batch):
            # print("STARTING EPISODE ", episode)

            if self.random_select: # select random house set 
                rand = np.random.randint(len(self.mapnames_train))
                mapname = self.mapnames_train[rand]
            else:
                assert(False)
            mapnames.append(mapname)

        n = np.random.randint(args.n_train_mapping_obs)

        input_dict = self.load_agent_output(n, mapnames)

        loss, inference_dict = self.model(input_dict, self.mem_dict, summ_writer=self.summ_writer)
        total_loss += loss

        self.summ_writer.summ_scalar('train/total_loss', total_loss.cpu().item())

        # plot inference
        if self.summ_writer.save_this:
            top_k_classes = inference_dict['top_k_classes']
            input_classes = inference_dict['input_classes']
            for b in range(args.objects_per_scene*args.scenes_per_batch):
                input_class_b = input_classes[b]
                top_k_classes_b = top_k_classes[b]
                text = ''
                for k in range(len(top_k_classes_b)):
                    t_ = input_class_b + ' -> ' + top_k_classes_b[k] + '  \n'
                    text += t_
                name = 'train/' + str(b)
                self.summ_writer.summ_text(name, text)

        return total_loss

    def run_val(self):
        self.model.eval()

        total_loss = torch.tensor(0.0).cuda() #to(self.device0) #.cuda()

        # mapnames = []
        # for episode in range(args.scenes_per_batch):
        #     # print("STARTING EPISODE ", episode)

        #     if self.random_select: # select random house set 
        #         rand = np.random.randint(len(self.mapnames_train))
        #         mapname = self.mapnames_train[rand]
        #     else:
        #         assert(False)
        #     mapnames.append(mapname)

        mapnames = self.mapnames_val

        n = np.random.randint(args.n_val_mapping_obs)
        input_dict = self.load_agent_output(n, mapnames)

        loss, inference_dict = self.model(input_dict, self.mem_dict, summ_writer=self.summ_writer)
        total_loss += loss

        self.summ_writer.summ_scalar('val/total_loss', total_loss.cpu().item())

        # plot inference
        if self.summ_writer.save_this:
            top_k_classes = inference_dict['top_k_classes']
            input_classes = inference_dict['input_classes']
            for b in range(args.objects_per_scene*args.scenes_per_batch):
                input_class_b = input_classes[b]
                top_k_classes_b = top_k_classes[b]
                text = ''
                for k in range(len(top_k_classes_b)):
                    t_ = input_class_b + ' -> ' + top_k_classes_b[k] + '  \n'
                    text += t_
                name = 'val/' + str(b)
                self.summ_writer.summ_text(name, text)

        self.model.train()

        return total_loss

    
    def get_scene_graph(
        self, 
        mapname, 
        obs_dict,
        confidence_threshold_scene_graph=args.confidence_threshold_scene_graph,
        ):

        # get room type
        map_type = utils.aithor.get_map_type(mapname)
        # map_types.append(map_type)

        features_ddetr_, labels_ = get_scene_graph_features(
            obs_dict, 
            self.ddetr, 
            self.score_labels_name, 
            self.W, self.H,
            confidence_threshold_scene_graph=confidence_threshold_scene_graph,
            )
        return map_type, features_ddetr_, labels_
        # ddetr_labels.append(labels_)
        # features_ddetr.append(features_ddetr_)
        # ddetr_batch_inds.append(torch.ones(len(features_ddetr_))*episode)

@torch.no_grad()
def get_scene_graph_features(
    obs_dict, 
    ddetr, 
    score_labels_name, 
    W, H, 
    use_backbone=True, 
    confidence_threshold_scene_graph=args.confidence_threshold_scene_graph
    ):
    '''
    This function loads images from an agent exploration and runs ddetr on them, and gets high confidence features from them
    '''

    rgbs = torch.from_numpy(obs_dict['rgb']).cuda()

    if rgbs.shape[0]>args.max_views_scene_graph:
        inds_all = np.arange(rgbs.shape[0])
        inds = np.sort(np.random.choice(inds_all, size=args.max_views_scene_graph, replace=False))
        rgbs = rgbs[inds]

    print(rgbs.shape[0])
    with torch.no_grad():
        if use_backbone:
            # rgb = self.controller.last_event.frame # (H, W, 3)
            # rgbs = (torch.from_numpy(rgb.copy()).float() / 255.0).cuda().permute(2,0,1).unsqueeze(0) # (1, 3, H, W)
            feature_maps = ddetr.model.backbone(nested_tensor_from_tensor_list(rgbs))[0][args.backbone_layer_ddetr].decompose()[0]
        outputs = ddetr(rgbs, do_loss=False, return_features=True)
    
    out = outputs['outputs']
    features = out['features']
    postprocessors = outputs['postprocessors']
    predictions = postprocessors['bbox'](out, torch.as_tensor([W, H]).unsqueeze(0).repeat(len(out['pred_boxes']), 1).cuda(), do_masks=False)
    features_ddetr = []
    labels_ddetr = []
    for b in range(len(predictions)):
        pred_boxes = predictions[score_labels_name][b]['boxes']
        pred_labels = predictions[score_labels_name][b]['labels']
        pred_scores = predictions[score_labels_name][b]['scores']
        keep, count = box_ops.nms(pred_boxes, pred_scores, 0.2, top_k=100)
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]
        keep = pred_scores>args.confidence_threshold_scene_graph
        pred_boxes = pred_boxes[keep].cpu().numpy()
        pred_labels = pred_labels[keep].cpu().numpy()
        pred_scores = pred_scores[keep].cpu().numpy() 

        if len(pred_boxes)==0:
            continue

        # if args.use_features:
        if use_backbone:
            feature_map = feature_maps[b].unsqueeze(0)
            feats = []
            for o_i in range(len(pred_boxes)):
                obj_bbox = pred_boxes[o_i]
                obj_bbox = (int(obj_bbox[0]), int(obj_bbox[1]), int(obj_bbox[2]), int(obj_bbox[3])) #obj_bbox.astype(np.int32)
                obj_bbox = np.array(obj_bbox)
                feature_crop = torchvision.ops.roi_align(feature_map, [torch.from_numpy(obj_bbox / 8).float().unsqueeze(0).cuda()], output_size=(32,32))
                pooled_obj_feat = F.adaptive_avg_pool2d(feature_crop, (1,1)).squeeze(-1).squeeze(-1)
                feats.append(pooled_obj_feat)
            feats = torch.stack(feats, dim=0)
        else:
            assert(False) # remove using query features and replaced with backbone
        labels = torch.from_numpy(pred_labels).cuda().long() 
        labels_ddetr.append(labels)
        features_ddetr.append(feats)
        # batch_inds.append(torch.ones(len(feats))*b)
        # plt.figure()
        # plt.imshow(rgbs[b].permute(1,2,0).cpu().numpy())
        # plt.savefig('images/test.png')
    if len(features_ddetr)==0:
        return None, None
    features_ddetr = torch.cat(features_ddetr, dim=0).squeeze(1)
    labels_ddetr = torch.cat(labels_ddetr, dim=0)
    return features_ddetr, labels_ddetr

        
    
    

if __name__ == '__main__':
    Ai2Thor()
