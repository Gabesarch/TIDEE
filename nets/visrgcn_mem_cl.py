import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from torchvision import models
from arguments import args
from torch.autograd import Variable
import ipdb
st = ipdb.set_trace
from torch_geometric.nn import RGCNConv, GCNConv
from itertools import combinations  

class RGCN(nn.Module):
    # def __init__(self, fea_size, dropout=False, gate_width=1, use_kernel_function=False):
    def __init__(
        self, 
        in_channels,  
        num_classes, 
        num_relations, 
        use_memory=True, 
        id_to_name=None, 
        name_to_id=None,
        visual_feat_size=None
        ):
        super(RGCN, self).__init__()
        
        visual_feat_size_after_proj = args.visual_feat_size_after_proj
         
        self.num_relations = num_relations

        self.num_classes = num_classes

        self.use_memory = use_memory

        if id_to_name is not None:
            self.id_to_name = id_to_name
            self.name_to_id = name_to_id

        out_channels = in_channels

        if not args.without_memex:
            # rgcn memory layers
            self.rgcn1_mem = RGCNConv(in_channels, out_channels, num_relations=num_relations)
            self.rgcn2_mem = RGCNConv(in_channels, out_channels, num_relations=num_relations)
            self.rgcn3_mem = RGCNConv(in_channels, out_channels, num_relations=num_relations)
        
        if not args.remove_sg_layers:
            self.gcn1_sg = GCNConv(in_channels, out_channels)
            self.gcn2_sg = GCNConv(in_channels, out_channels)
            self.gcn3_sg = GCNConv(in_channels, out_channels)

        num_relations = 0

        num_relations += 1 # edge type for bridging memex to oop node
        self.bridge_ind_memory = 0

        num_relations += 1 # edge type for bridging map type node
        self.bridge_ind_maptype = 1

        num_relations += 1 # edge type for bridging ddetr feats to oop node
        self.bridge_ind_ddetr = 2

        num_relations += 1 # edge type for bridging scene graph to memory graph
        self.bridge_ind_scene_memory = 3

        # rgcn bridge layers
        self.rgcn1_bridge = RGCNConv(in_channels, out_channels, num_relations=num_relations)
        self.rgcn2_bridge = RGCNConv(in_channels, out_channels, num_relations=num_relations)
        self.rgcn3_bridge = RGCNConv(in_channels, out_channels, num_relations=num_relations)
        self.rgcn4_bridge = RGCNConv(in_channels, out_channels, num_relations=num_relations)

        self.obj_embeddings = nn.Embedding(num_classes, in_channels - visual_feat_size_after_proj)
        self.scene_type_embeddings = nn.Embedding(4, in_channels) # four scene types in aithor

        # resnet
        # self.resnet = models.resnet50(pretrained=True)
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        # # optimize layer 4
        # for param in self.resnet.layer4.parameters():
        #     param.requires_grad = True
        # alter fc layer

        # num_ftrs = self.resnet.fc.in_features
        # self.resnet.fc = nn.Linear(num_ftrs, in_channels)
        # norm
        self.pre_mean=torch.as_tensor([0.485, 0.456, 0.406]).cuda().reshape(1,3,1,1)
        self.pre_std=torch.as_tensor([0.229, 0.224, 0.225]).cuda().reshape(1,3,1,1)

        # class head
        self.obj_predictor = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, num_classes),
        )
        # if args.do_visual_features:
        # layer_sizes = [512, 1024, 2048]
        self.ddetr_proj = nn.Sequential(
            nn.Linear(visual_feat_size, visual_feat_size_after_proj),
        )

        self.dropout = nn.Dropout(p=0.2)

        self.gamma = 2.
        self.alpha = 0.25

        self.ce = torch.nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        self.k = 5


    def forward(self, input_dict, mem_dict, summ_writer=None, do_loss=True):
        
        subj_obj_inds_mem = mem_dict['subj_obj_inds'].clone().T
        rel_inds_mem = mem_dict['rel_inds'].clone()
        
        obj_feats_mem = mem_dict['obj_feats'].clone()

        # get target object indices
        if len(input_dict['obj_lens'])>0:
            target_obj_inds_batch = torch.from_numpy(np.array([0] + list(np.cumsum(input_dict['obj_lens'].cpu().numpy())))[:-1]).cuda()
        else:
            target_obj_inds_batch = torch.from_numpy(np.array([0]))
        obj_labels_gt = input_dict['obj_ids'][target_obj_inds_batch]
        obj_ids_input = input_dict['obj_ids']

        num_targets = len(obj_labels_gt)

        
        if 'batch_inds' not in input_dict.keys():
            num_objs_per_batch = 1 # for inference
            num_batches = torch.unique(input_dict['ddetr_batch_inds']).shape[0]
        else:
            num_objs_per_batch = args.objects_per_scene
            num_batches = args.scenes_per_batch

        # get embeddings for target and memory objects
        obj_embeddings_target = self.obj_embeddings(obj_labels_gt)
        
        # add visual embeddings to the obj embeddings
        obj_feats_tgt = input_dict['obj_feats'][target_obj_inds_batch]
        obj_feats_tgt = self.ddetr_proj(obj_feats_tgt)
        obj_feats_mem = self.ddetr_proj(obj_feats_mem)
        obj_embeddings_target = torch.cat([obj_embeddings_target, obj_feats_tgt], dim=1)

        if args.without_memex:
            num_memory_nodes = 0
            mem_x = torch.tensor([]).cuda()
        else:
            obj_ids_mem = mem_dict['obj_ids'].clone()
            obj_embeddings_memory = self.obj_embeddings(obj_ids_mem)
            obj_embeddings_memory = torch.cat([obj_embeddings_memory, obj_feats_mem], dim=1)
            num_memory_nodes = len(obj_embeddings_memory)

        # obj_embeddings_target += obj_feats_tgt
        # obj_embeddings_memory += obj_feats_mem

        subj_obj_inds_bridge = []
        rel_inds_bridge = []
        
        # prepare pass from memory to target object (connect target to memory based on class labels)
        if not args.without_memex:
            print("Connecting memory to target")
            for obj_i in range(num_targets):
                obj_t = obj_labels_gt[obj_i]
                # find where object exists in memory
                where_id_mem = torch.where(obj_t==obj_ids_mem)
                mem_matches = where_id_mem[0] + num_targets
                # create sparse adjacency matrix
                subj_obj_inds_bridge_ = torch.stack([mem_matches, torch.ones(len(mem_matches)).cuda().long()*obj_i])
                # create bridging edge type
                rel_inds_bridge_ = torch.ones(subj_obj_inds_bridge_.shape[1]).cuda() * self.bridge_ind_memory
                
                subj_obj_inds_bridge.append(subj_obj_inds_bridge_)
                rel_inds_bridge.append(rel_inds_bridge_)

        # prepare pass from map type to target object
        print("Connecting map type to target")
        map_ids = input_dict['map_types']
        map_features = self.scene_type_embeddings(map_ids)
        for b in range(num_batches):
            maptype_featind = num_targets + num_memory_nodes + b # map index is after target and memory features
            target_inds = torch.from_numpy(np.arange(num_objs_per_batch) + b*num_objs_per_batch).cuda().long()
            # create sparse adjacency matrix
            subj_obj_inds_bridge_ = torch.stack([torch.ones(len(target_inds)).long().cuda()*maptype_featind, target_inds])
            # create bridging edge type
            rel_inds_bridge_ = torch.ones(subj_obj_inds_bridge_.shape[1]).cuda() * self.bridge_ind_maptype
            
            subj_obj_inds_bridge.append(subj_obj_inds_bridge_)
            rel_inds_bridge.append(rel_inds_bridge_)

        print("Connecting scene to target")
        # prepare pass from scene to target object
        map_ids = input_dict['map_types']
        features_ddetr_visual = input_dict['features_ddetr']
        features_ddetr_visual = self.ddetr_proj(features_ddetr_visual)
        ddetr_labels = input_dict['ddetr_labels']
        features_ddetr_category = self.obj_embeddings(ddetr_labels)
        features_ddetr = torch.cat([features_ddetr_visual, features_ddetr_category], dim=1)
        ddetr_batch_inds = input_dict['ddetr_batch_inds']

        

        if not args.remove_sg_layers:
            # create all to all connections for passing within scene graph
            subj_obj_inds_SG = []
            for b in range(num_batches):
                feat_inds_b = torch.where(ddetr_batch_inds==b)[0]
                subj_obj_inds_SG_ = torch.tensor(list(combinations(range(feat_inds_b[0], feat_inds_b[-1]+1),2))).t().cuda() # fully connected
                subj_obj_inds_SG.append(subj_obj_inds_SG_)
            subj_obj_inds_SG = torch.cat(subj_obj_inds_SG, dim=1).long()
        
        # prepare pass from scene graph to target object
        for b in range(num_batches):
            feat_inds_b = torch.where(ddetr_batch_inds==b)[0] + num_targets + num_memory_nodes + len(map_features)
            feat_inds_b_repeat = feat_inds_b.repeat(num_objs_per_batch).cuda().long()
            target_inds = torch.from_numpy(np.repeat(np.arange(num_objs_per_batch) + b*num_objs_per_batch, len(feat_inds_b))).cuda().long()

            # create sparse adjacency matrix
            subj_obj_inds_bridge_ = torch.stack([feat_inds_b_repeat, target_inds])
            # create bridging edge type
            rel_inds_bridge_ = torch.ones(subj_obj_inds_bridge_.shape[1]).cuda() * self.bridge_ind_ddetr
            
            subj_obj_inds_bridge.append(subj_obj_inds_bridge_)
            rel_inds_bridge.append(rel_inds_bridge_)

        # if not args.remove_connect_mem_with_scene:
        # prepare pass from memory to target object (connect target to memory based on class labels)
        if not args.without_memex:
            print("Connecting memory to scene")
            for obj_i in range(ddetr_labels.shape[0]):
                obj_t = ddetr_labels[obj_i]
                # find where object exists in memory
                where_id_mem = torch.where(obj_t==obj_ids_mem)
                mem_matches = where_id_mem[0] + num_targets

                scene_ind = num_targets + num_memory_nodes + len(map_features) + obj_i
                # create sparse adjacency matrix

                # create link from mem to scene
                subj_obj_inds_bridge_ = torch.stack([mem_matches, torch.ones(len(mem_matches)).cuda().long()*scene_ind])

                # create link from scene to mem
                subj_obj_inds_bridge_ = torch.cat([subj_obj_inds_bridge_, torch.stack([torch.ones(len(mem_matches)).cuda().long()*scene_ind, mem_matches])], dim=1)

                # create bridging edge type
                rel_inds_bridge_ = torch.ones(subj_obj_inds_bridge_.shape[1]).cuda() * self.bridge_ind_scene_memory
                
                subj_obj_inds_bridge.append(subj_obj_inds_bridge_)
                rel_inds_bridge.append(rel_inds_bridge_)

        subj_obj_inds_bridge = torch.cat(subj_obj_inds_bridge, dim=1).long()
        rel_inds_bridge = torch.cat(rel_inds_bridge, dim=0).long()

        ##########%%%%%%%%% GCNs %%%%%%%################
        if not args.without_memex:
            # do memex graph message passing
            mem_x = self.rgcn1_mem(x=obj_embeddings_memory, edge_index=subj_obj_inds_mem, edge_type=rel_inds_mem)
            mem_x = F.relu(mem_x)
            # mem_x = F.dropout(mem_x, p=0.2, training=self.training)
            mem_x = self.dropout(mem_x)
            mem_x = self.rgcn2_mem(x=mem_x, edge_index=subj_obj_inds_mem, edge_type=rel_inds_mem)
            mem_x = F.relu(mem_x)
            # mem_x = F.dropout(mem_x, p=0.2, training=self.training)
            mem_x = self.dropout(mem_x)
            mem_x = self.rgcn3_mem(x=mem_x, edge_index=subj_obj_inds_mem, edge_type=rel_inds_mem)
            mem_x = F.relu(mem_x)
        
        if not args.remove_sg_layers:
            # do scene graph message passing
            features_ddetr = self.gcn1_sg(x=features_ddetr, edge_index=subj_obj_inds_SG)
            features_ddetr = F.relu(features_ddetr)
            # SG_x = F.dropout(SG_x, p=0.2, training=self.training)
            features_ddetr = self.dropout(features_ddetr)
            features_ddetr = self.gcn2_sg(x=features_ddetr, edge_index=subj_obj_inds_SG)
            features_ddetr = F.relu(features_ddetr)
            # SG_x = F.dropout(SG_x, p=0.2, training=self.training)
            features_ddetr = self.dropout(features_ddetr)
            features_ddetr = self.gcn3_sg(x=features_ddetr, edge_index=subj_obj_inds_SG)
            features_ddetr = F.relu(features_ddetr)

        # create node embeddings for bridging
        obj_embeddings = torch.cat([obj_embeddings_target, mem_x, map_features, features_ddetr], dim=0)

        # message pass memory + scene -> target objs
        bridge_x = self.rgcn1_bridge(x=obj_embeddings, edge_index=subj_obj_inds_bridge, edge_type=rel_inds_bridge)
        bridge_x = F.relu(bridge_x)
        # bridge_x = F.dropout(bridge_x, p=0.2, training=self.training)
        bridge_x = self.dropout(bridge_x)
        bridge_x = self.rgcn2_bridge(x=bridge_x, edge_index=subj_obj_inds_bridge, edge_type=rel_inds_bridge)
        bridge_x = F.relu(bridge_x)
        # bridge_x = F.dropout(bridge_x, p=0.2, training=self.training)
        bridge_x = self.dropout(bridge_x)
        bridge_x = self.rgcn3_bridge(x=bridge_x, edge_index=subj_obj_inds_bridge, edge_type=rel_inds_bridge)
        bridge_x = F.relu(bridge_x)
        # bridge_x = F.dropout(bridge_x, p=0.2, training=self.training)
        bridge_x = self.dropout(bridge_x)
        bridge_x = self.rgcn4_bridge(x=bridge_x, edge_index=subj_obj_inds_bridge, edge_type=rel_inds_bridge)
        bridge_x = F.relu(bridge_x)

        # logits
        obj_class_logits_target = self.obj_predictor(bridge_x[:num_targets])

        if do_loss:
            # prepare object predictor supervision 
            ce_targets = []
            for obj_i in range(num_targets):
                batch_inds_i = input_dict['batch_inds']==obj_i
                obj_supervision_ids = torch.unique(input_dict['subj_obj_inds'][batch_inds_i,1])
                obj_supervision_ids = obj_ids_input[obj_supervision_ids]
                ce_supervision = torch.zeros(self.num_classes).cuda()
                ce_supervision[obj_supervision_ids] = 1 # put mass 1 at each obj index that target is related to
                ce_targets.append(ce_supervision)
            ce_targets = torch.stack(ce_targets)

            loss = F.binary_cross_entropy_with_logits(obj_class_logits_target, ce_targets.float(), reduction='none')
            loss = loss.mean()
            # loss = focal_loss(BCE_loss, ce_targets, self.gamma, self.alpha)
        else:
            loss = None

        # get probabilities
        obj_class_logits_target_ = obj_class_logits_target.detach().cpu()
        
        # hack to remove floor
        if True:
            print("NOTE: Visual RGCN removing floor in model file. Remove this if want to include floor.")
            floor_id = self.name_to_id['Floor']
            obj_class_logits_target_ = obj_class_logits_target_[:,:floor_id]

        probs = self.softmax(obj_class_logits_target_)


        # top k classes
        top_k_inds = torch.topk(probs, self.k, dim=1).indices
        top_k_classes = []
        for obj_i in range(num_targets):
            top_k_inds_i = top_k_inds[obj_i].cpu().numpy()
            top_classes_i = []
            for k in range(self.k):
                class_ = self.id_to_name[top_k_inds_i[k]]
                top_classes_i.append(class_)
            top_k_classes.append(top_classes_i)
        
        inference_dict = {}
        inference_dict['probs'] = probs
        inference_dict['top_k_classes'] = top_k_classes
        inference_dict['input_classes'] = [self.id_to_name[obj_labels_gt_i] for obj_labels_gt_i in list(obj_labels_gt.cpu().numpy())]

        # print(inference_dict['top_k_classes'])
        # print(inference_dict['input_classes'])

        return loss, inference_dict

def focal_loss(bce_loss, targets, gamma, alpha):
    """Binary focal loss, mean.

    Per https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/5 with
    improvements for alpha.
    :param bce_loss: Binary Cross Entropy loss, a torch tensor.
    :param targets: a torch tensor containing the ground truth, 0s and 1s.
    :param gamma: focal loss power parameter, a float scalar.
    :param alpha: weight of the class indicated by 1, a float scalar.
    """
    p_t = torch.exp(-bce_loss)
    alpha_tensor = (1 - alpha) + targets * (2 * alpha - 1)  # alpha if target = 1 and 1 - alpha if target = 0
    f_loss = alpha_tensor * (1 - p_t) ** gamma * bce_loss
    return f_loss.mean()

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()
        






