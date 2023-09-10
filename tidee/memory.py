import ipdb
st = ipdb.set_trace
from arguments import args
from utils.wctb import Utils, Relations
from utils.wctb import ThorPositionTo2DFrameTranslator
import utils.aithor
from nets.visrgcn_mem_cl import RGCN
from models.aithor_visrgcn import get_scene_graph_features
from backend import saverloader
import os
import pickle
import torch
from SOLQ.util.misc import nested_tensor_from_tensor_list
import torchvision
import torch.nn.functional as F
import torchvision.ops
from nets.solq import DDETR
import sys
from SOLQ.util import box_ops
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Memory():

    def __init__(
        self, 
        include_classes, 
        name_to_id, 
        id_to_name, 
        W,H,
        ddetr=None,
        task=None,
        ): 

        self.W, self.H = W,H

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

        self.maptype_to_id = {"kitchen":0, "living_room":1, "bedroom":2, "bathroom":3}

        # self.aithor_base = aithor_base
        self.include_classes = include_classes
        self.name_to_id = name_to_id
        self.id_to_name = id_to_name

        self.include_classes_floor = self.include_classes + ['Floor']
        self.name_to_id_floor = self.name_to_id
        self.name_to_id_floor['Floor'] = len(self.include_classes_floor) - 1
        self.id_to_name_floor = self.id_to_name
        self.id_to_name_floor[len(self.include_classes_floor) - 1] = 'Floor' 

        self.task = task

        if args.do_most_common_memory:
            self.get_memory()

            # self.general_receptacles_classes = ['CounterTop', 'DiningTable', 'CoffeeTable', 'SideTable', 'Sink', 'TowelHolder',
            #     'Desk','Bed',  'TVStand', 'Sofa', 'ArmChair', 'SinkBasin', 
            #     'Drawer', 'Chair', 'Shelf', 'Dresser', 'Fridge', 'Ottoman',   'DogBed', 'ShelvingUnit', 'Cabinet',
            #     'StoveBurner', 'Microwave', 'CoffeeMachine', 'GarbageCan', 
            #     'Toaster', 'LaundryHamper', 'Stool',  'Bathtub', 'Footstool', 'BathtubBasin', 'Safe','Box', 
            #     'Dresser', 'Desk', 'SideTable', 'DiningTable', 'TVStand', 'CoffeeTable', 'CounterTop', 'Shelf',
            #     'Pot', 'Pan', 'Bowl', 'Microwave', 'Fridge', 'Plate', 'Sink', 'SinkBasin', 'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'Desk', 'CounterTop', 'GarbageCan', 'Dresser',
            #     'Sofa', 'ArmChair', 'Dresser', 'Desk', 'Bed', 'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop'
            #     'Box', 'Ottoman', 'Dresser', 'Desk', 'Cabinet', 'DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Shelf', 'Drawer', 'GarbageCan', 'Safe', 'Sofa', 'ArmChair',
            #     'LaundryHamper', 'Desk', 'Toilet', 'Cart', 'BathtubBasin', 'Bathtub', 'Sink', 'SinkBasin','HandTowelHolder','StoveBurner','Cart'
            #     ]

            self.receptacles_classes = [
                'HandTowelHolder', 'CoffeeTable', 'Sofa', 'Drawer', 'Toaster', 'Bowl', 'Safe', 
                'Ottoman', 'Pan', 'Footstool', 'TowelHolder', 'StoveBurner', 'Stool', 'CounterTop', 
                'CoffeeMachine', 'DiningTable', 'Chair', 'CounterTopBox', 'Cabinet', 'LaundryHamper', 
                'Toilet', 'GarbageCan', 'Plate', 'Desk', 'Cart', 'ShelvingUnit', 'Box', 'Shelf', 
                'DogBed', 'TVStand', 'ArmChair', 'Fridge', 'BathtubBasin', 'Dresser', 'Sink', 
                'SideTable', 'Bed', 'Pot', 'SinkBasin', 'Bathtub', 'Microwave'
                ]

        elif args.do_visual_memex:

            self.score_boxes_name = 'pred1'
            self.score_labels_name = 'pred1'
            self.score_labels_name2 = 'pred2' # which prediction head has the OOP label?

            self.ddetr = ddetr
        
            num_classes = len(self.include_classes_floor) 
            num_relations = len(self.relations_executors_pairs)

            layer_sizes = [512, 1024, 2048] # ddetr backbone output features sizes per layer index
            visual_feat_size = layer_sizes[args.backbone_layer_ddetr]
            self.visual_feat_size = visual_feat_size

            in_channels = args.in_channels #+ visual_feat_size
            # out_channels = args.out_channels

            self.model = RGCN(in_channels, num_classes, num_relations, use_memory=True, id_to_name=self.id_to_name_floor, name_to_id=self.name_to_id_floor, visual_feat_size=visual_feat_size)
            self.model.cuda()

            _ = saverloader.load_from_path(args.visual_memex_checkpoint, self.model, None, strict=True, lr_scheduler=None)

            # obtain memory from memory houses
            self.get_memory()

    def get_memory(self):

        if os.path.exists(args.visual_memex_path):
            with open(args.visual_memex_path, 'rb') as handle:
                self.mem_dict = pickle.load(handle)
            return 
        else:
            assert(False) # please download memex data or generate memex data in aithor_visrgcn.py

    def process_scene_graph(self, obs_dict, mapname):
        '''
        Get scene graph from map type and mapping phase images
        '''
        
        map_type, features_ddetr, labels = self.get_scene_graph(mapname, obs_dict)
        if features_ddetr is None:
            for ti in range(3):
                map_type, features_ddetr, labels = self.get_scene_graph(
                    mapname, 
                    obs_dict, 
                    confidence_threshold_scene_graph=args.confidence_threshold_scene_graph-(.1*ti),
                    )
                if features_ddetr is not None:
                    break
        
        # for mi in range(len(obs_dict["rgb"])):
        #     plt.figure(1); plt.clf()
        #     plt.imshow(obs_dict["rgb"][mi].transpose((1,2,0)))
        #     plt.savefig(f'images/test{mi}.png')
        # st()

        self.input_dict = {}
        self.input_dict['map_types'] = torch.tensor([self.maptype_to_id[map_type]]).cuda().long()
        self.input_dict['features_ddetr'] = features_ddetr
        self.input_dict['ddetr_batch_inds'] = torch.ones(len(features_ddetr))*0
        self.input_dict['ddetr_labels'] = labels

    @torch.no_grad()
    def get_scene_graph(
        self, 
        mapname, 
        obs_dict,
        confidence_threshold_scene_graph=args.confidence_threshold_scene_graph
        ):

        # get room type
        map_type = utils.aithor.get_map_type(mapname)
        features_ddetr_, labels_ = get_scene_graph_features(
            obs_dict, 
            self.ddetr, 
            self.score_labels_name, 
            self.W, self.H,
            confidence_threshold_scene_graph=confidence_threshold_scene_graph
            )
        return map_type, features_ddetr_, labels_


    def get_most_common_objs_in_memory(self, oop_dict, obs_dict, vis=None):
        oopType = oop_dict['object_name']
        oopID = self.name_to_id[oopType]
        batch_inds = self.mem_dict['batch_inds']
        mem_ids = self.mem_dict['obj_ids']
        subj_obj_inds = self.mem_dict['subj_obj_inds']
        rel_inds = self.mem_dict['rel_inds']

        # filter by supported by relation
        supported_ind = self.rel_to_id['supported-by']
        supported_where = torch.where(rel_inds==supported_ind)[0]
        subj_obj_inds = subj_obj_inds[supported_where,:]
        
        where_oop_id = torch.where(mem_ids==oopID)[0]
        object_types_related = []
        for mem_i in where_oop_id:
            where_oop_id_graph = torch.where(subj_obj_inds[:,0]==mem_i)[0]
            related_oop_ids = subj_obj_inds[where_oop_id_graph,1]
            object_types_related_ = list(set([self.id_to_name[int(mem_ids[i].cpu().numpy())] for i in list(related_oop_ids.cpu().numpy())]))
            object_types_related.extend(object_types_related_)
        object_types_related_ = []
        for object_types_related_i in object_types_related:
            if object_types_related_i in self.receptacles_classes:
                object_types_related_.append(object_types_related_i)
        ids = np.array([self.name_to_id[object_types_related_[i]] for i in range(len(object_types_related_))])
        vals, indices, counts = np.unique(ids, return_counts=True, return_index=True)
        ids_best = vals[np.flip(np.argsort(counts))]
        search_classes = [object_types_related_[i] for i in indices[np.flip(np.argsort(counts))]]

        return [search_classes]

    def run_visual_memex(self, oop_dict, obs_dict, vis=None):

        mapname = self.task.controller.scene.split('_')[0]

        # (2) Process mapping images into a scene graph of visual features + labels
        map_type, features_ddetr, ddetr_labels = self.get_scene_graph(mapname, obs_dict)

        # construct input dictionary
        # input_dict = self.get_relation_dict(subj_pred_obj_adj_scene_batch, obj_ids_scene_batch)
        input_dict = {}
        input_dict['map_types'] = torch.tensor([self.maptype_to_id[map_type]]).cuda().long()
        input_dict['features_ddetr'] = features_ddetr
        input_dict['ddetr_batch_inds'] = torch.ones(len(features_ddetr)).cuda()*0
        input_dict['ddetr_labels'] = ddetr_labels
        input_dict['obj_feats'] = oop_dict['features']
        input_dict['obj_ids'] = torch.tensor([oop_dict['label']]).cuda()
        input_dict['obj_lens'] = []

        _, inference_dict = self.model(input_dict, self.mem_dict, summ_writer=None, do_loss=False)
        
        print(f"Top K prediction for {oop_dict['object_name']} are:\n {inference_dict['top_k_classes'][0]}")

        if vis is not None and args.visualize_memex:
            top_k_classes_b = inference_dict['top_k_classes'][0]
            vis.add_rgcn_visual(self.id_to_name[int(oop_dict['label'])], top_k_classes_b)
        
        return inference_dict

