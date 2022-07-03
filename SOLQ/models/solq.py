# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

import numpy as np
import cv2
from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from .deformable_transformer_cp import build_deforamble_transformer as build_cp_deforamble_transformer
from .dct import ProcessorDCT
from detectron2.structures import BitMasks
from detectron2.layers import paste_masks_in_image
from detectron2.utils.memory import retry_if_cuda_oom
import copy
import functools

import ipdb
st = ipdb.set_trace
print = functools.partial(print, flush=True)
import matplotlib.pyplot as plt

import hyperparams as hyp

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SOLQ(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, with_vector=False, processor_dct=None, vector_hidden_dim=256):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.with_vector = with_vector
        self.processor_dct = processor_dct
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if self.with_vector:
            print(f'Training with vector_hidden_dim {vector_hidden_dim}.', flush=True)
            self.vector_embed = MLP(hidden_dim, vector_hidden_dim, self.processor_dct.n_keep, 3)
        self.num_feature_levels = num_feature_levels

        # projection for multiscale object queries 
        self.multiscale_query_proj = nn.Linear(hidden_dim*num_feature_levels, hidden_dim)

        # # learned embeddings for target/supporters
        # if hyp.pred_one_object and hyp.use_supporters:
        #     self.target_emb = nn.Embedding(50, num_pos_feats)

        # if not two_stage:
        #     self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.vector_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].bias.data[2:], -2.0)
            self.vector_embed = nn.ModuleList([self.vector_embed for _ in range(num_pred)])

        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        if hyp.do_self_atn_for_queries:
            self.transformer.multiscale_query_proj = self.multiscale_query_proj

    def forward(self, samples, instance_masks, targets=None, target_idxs=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples: batched images, of shape [batch_size x nviews x 3 x H x W]
               - instance_masks: object masks in list batch, nviews, HxW

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        assert(not (targets is None and hyp.filter_inds))

        assert(not (hyp.do_self_atn_for_queries and not hyp.image_features_self_attend)) # cant have do_self_atn_for_queries without image_features_self_attend
        
        B, S, C, H, W = samples.shape
        samples = samples.view(B*S, C, H, W).unbind(0)
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples) 

        num_feature_levs = len(features)

        if hyp.pred_one_object:
            # bring target object to first index
            if target_idxs is None:
                num_objs = [len(instance_masks[i][0]) for i in range(len(instance_masks))]
                target_idxs = np.random.randint(num_objs)

        srcs = []
        masks = []
        poss = []
        # objects to track
        obj_feats = []
        obj_poss = []
        obj_masks = []
        # for l, feat in enumerate(features):
        for l in range(self.num_feature_levels):
            if l > num_feature_levs - 1:
                if l == num_feature_levs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
            else:
                feat = features[l]
                pos_l = pos[l]
                src, mask = feat.decompose()
                src = self.input_proj[l](src)
            _, C_l, H_l, W_l = src.shape
            src = src.view(B,S,C_l,H_l,W_l)
            mask = mask.view(B,S,H_l,W_l)
            _, C_pos_l, _, _ = pos_l.shape
            pos_l = pos_l.view(B,S,C_pos_l,H_l,W_l)
            obj_feat = []
            obj_pos_l = []
            obj_mask = []
            for b in range(B):
                # take object masks in first view 
                obj_m = instance_masks[b][0]
                pos_l_m = pos_l[b][0].view(C_pos_l,H_l,W_l)
                if hyp.pred_one_object:
                    # bring target object to first index
                    if hyp.use_supporters:
                        num_masks = len(obj_m)
                        mask_loop_inds = list(np.arange(num_masks))
                        mask_loop_inds.insert(0, mask_loop_inds.pop(target_idxs[b]))
                    else:
                        num_masks = 1
                        mask_loop_inds = target_idxs[b]
                else:
                    num_masks = len(obj_m)
                    mask_loop_inds = list(np.arange(num_masks))
                obj_m_0_ = F.interpolate(obj_m[None].float(), mode='bilinear', size=src.shape[-2:])
                obj_m_0 = obj_m_0_.to(torch.bool)[0]
                # obj_m_0 = obj_m_0.unsqueeze(1).repeat(1,C_l,1,1)
                if not hyp.do_self_atn_for_queries:
                    src_0 = src[b,0]
                # where_masks = torch.where(obj_m)
                obj_feat_ = []
                obj_pos_l_ = []
                # print(num_masks)
                if hyp.filter_inds:
                    st()
                    targets[b][s]["filter_inds"]
                for n in range(num_masks):
                    obj_m_0_n = obj_m_0[n] #.view(H_l,W_l)
                    where_masks = torch.where(obj_m_0_n)#[0]
                    # pos encoding
                    if len(where_masks[0])==0:
                        # Hack - if no points in interpolated mask, take nearest point to median of mask
                        y_m = int((torch.median(torch.where(obj_m[n])[0]) * src.shape[-2:][0]) / obj_m[n].shape[0])
                        x_m = int((torch.median(torch.where(obj_m[n])[1]) * src.shape[-2:][1]) / obj_m[n].shape[1]) # 480 / x = 8 / y
                        where_masks = (torch.tensor([y_m], dtype=torch.int64).cuda(), torch.tensor([x_m], dtype=torch.int64).cuda())
                    half_m = int(len(where_masks[0])/2)
                    # src_0 = src_0.view(C_l,H_l,W_l)
                    
                    if hyp.do_self_atn_for_queries:
                        obj_feat_.append(where_masks)
                    else:
                        src_0_m = src_0[:,where_masks[0],where_masks[1]].mean(dim=1)
                        obj_feat_.append(src_0_m)
                    pos_obj = pos_l_m[:,where_masks[0][half_m], where_masks[1][half_m]]
                    obj_pos_l_.append(pos_obj)
                num_pad = hyp.max_objs - num_masks
                pad = torch.zeros(num_pad, C_l).cuda()
                if not hyp.do_self_atn_for_queries:
                    obj_feat_ = torch.stack(obj_feat_)
                    obj_feat_ = torch.cat([obj_feat_, pad], dim=0)
                obj_pos_l_ = torch.stack(obj_pos_l_)
                pad_pos_l = torch.zeros(num_pad, C_pos_l).cuda()
                obj_pos_l_ = torch.cat([obj_pos_l_, pad_pos_l], dim=0)
                obj_mask_ = torch.ones(hyp.max_objs).to(torch.bool).cuda()
                obj_mask_[:num_masks] = False
                obj_mask.append(obj_mask_)
                obj_feat.append(obj_feat_)
                obj_pos_l.append(obj_pos_l_)
            
            if hyp.do_self_atn_for_queries:
                obj_feats.append(obj_feat)
                srcs.append(src.view(B*S,C_l,H_l,W_l))
                masks.append(mask.view(B*S,H_l,W_l))
                poss.append(pos_l.view(B*S,C_pos_l,H_l,W_l))
            else:
                obj_feats.append(torch.stack(obj_feat))
                srcs.append(src[:,1:].view(B*(S-1),C_l,H_l,W_l))
                masks.append(mask[:,1:].view(B*(S-1),H_l,W_l))
                poss.append(pos_l[:,1:].view(B*(S-1),C_pos_l,H_l,W_l))
            obj_poss.append(torch.stack(obj_pos_l))
            obj_masks.append(torch.stack(obj_mask))
            assert mask is not None
        
        # for now let's concat the object features and learn a linear projection
        if not hyp.do_self_atn_for_queries:
            obj_feats = torch.cat(obj_feats, dim=2)
            # project multiscale to correct size for query input
            obj_feats = self.multiscale_query_proj(obj_feats)
        obj_poss = obj_poss[0] # just take the first pos encoding as this will be most precise
        obj_masks = obj_masks[0] # all object masks are the same  

        # if hyp.pred_one_object:
        #     # bring target object to first index
        #     with torch.no_grad():
        #         num_objs = torch.sum(~obj_masks, dim=1).detach().cpu().numpy()
        #         if target_idxs is None:
        #             target_idxs = np.random.randint(num_objs)
            
        #     obj_feats_ = obj_feats[np.arange(len(obj_feats)),target_idxs,:].unsqueeze(1)
        #     obj_poss_ = obj_poss[np.arange(len(obj_feats)),target_idxs,:].unsqueeze(1)
        #     if hyp.use_supporters:
        #         # print("Predicting one object WITH supporters")
        #         mask_supporters = torch.ones((obj_feats.shape[0], obj_feats.shape[1]), dtype=bool)
        #         mask_supporters[np.arange(len(obj_feats)),target_idxs] = False

        #         # feats
        #         obj_mask_supporters = mask_supporters.unsqueeze(2).repeat(1,1,obj_feats.shape[2])
        #         obj_supporters = obj_feats[obj_mask_supporters].view(obj_feats.shape[0], hyp.max_objs-1, obj_feats.shape[2])
        #         obj_feats = torch.cat([obj_feats_, obj_supporters], dim=1)

        #         # pos encodings
        #         pos_mask_supporters = mask_supporters.unsqueeze(2).repeat(1,1,obj_poss.shape[2])
        #         pos_supporters = obj_poss[pos_mask_supporters].view(obj_poss.shape[0], hyp.max_objs-1, obj_poss.shape[2])
        #         obj_poss = torch.cat([obj_poss_, pos_supporters], dim=1)

        #         # mask
        #         pass
        #     else:
        #         # print("Predicting one object WITHOUT supporters")
        #         # if dont use supporters, everything is zero except first token
        #         # feats
        #         obj_feats = torch.zeros_like(obj_feats).cuda()
        #         obj_feats[:, 0:1, :] = obj_feats_

        #         # pos
        #         obj_poss = torch.zeros_like(obj_poss).cuda()
        #         obj_poss[:, 0:1, :] = obj_poss_

        #         # mask - only first object is valid
        #         obj_masks = torch.ones_like(obj_masks, dtype=bool).cuda()
        #         obj_masks[:, 0:1] = False                
        
        # if self.num_feature_levels > len(srcs):
        #     _len_srcs = len(srcs)
        #     for l in range(_len_srcs, self.num_feature_levels):
        #         if l == _len_srcs:
        #             src = self.input_proj[l](features[-1].tensors)
        #         else:
        #             src = self.input_proj[l](srcs[-1])
        #         m = samples.mask
        #         mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
        #         pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
        #         srcs.append(src)
        #         masks.append(mask)
        #         pos.append(pos_l)

        query_embeds = None
        # if not self.two_stage:
        #     query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, _, _ = self.transformer(
                                                                                                        srcs, 
                                                                                                        masks, 
                                                                                                        poss, 
                                                                                                        query_embeds, 
                                                                                                        obj_srcs=obj_feats, 
                                                                                                        obj_masks=obj_masks,
                                                                                                        obj_pos_embeds=obj_poss,
                                                                                                        B=B,
                                                                                                        S=S,
                                                                                                        )

        if hyp.pred_one_object and hyp.supervise_only_one:
            hs = hs[:,:,0:1,:] # only keep target object to decode

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if hyp.do_deformable_atn_decoder:
                if lvl == 0:
                    reference = init_reference
                else:
                    reference = inter_references[lvl - 1]
                reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if hyp.do_deformable_atn_decoder:
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        if self.with_vector:
            outputs_vectors = []
            for lvl in range(hs.shape[0]):
                outputs_vector = self.vector_embed[lvl](hs[lvl])
                outputs_vectors.append(outputs_vector)
            outputs_vector = torch.stack(outputs_vectors)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.with_vector:
            out.update({'pred_vectors': outputs_vector[-1]})
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_vector)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        if hyp.pred_one_object:
            out["target_idxs"] = target_idxs
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_vector):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_vectors': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_vector[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, 
                with_vector=False, 
                processor_dct=None, 
                vector_loss_coef=0.7, 
                no_vector_loss_norm=False,
                vector_start_stage=0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.with_vector = with_vector
        self.processor_dct = processor_dct
        self.vector_loss_coef = vector_loss_coef
        self.no_vector_loss_norm = no_vector_loss_norm
        self.vector_start_stage = vector_start_stage

        print(f'Training with {6-self.vector_start_stage} vector stages.')

        print(f"Training with vector_loss_coef {self.vector_loss_coef}.")

        if not self.no_vector_loss_norm:
            print('Training with vector_loss_norm.')

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # also allows for no objects in view
        target_classes_o = []
        for t, (_, J) in zip(targets, indices):
            if len(J)>0:
                target_classes_o.append(t["labels"][J])
        target_classes_o = torch.cat(target_classes_o)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        # target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # also allows for no objects in view
        target_boxes = []
        for t, (_, i) in zip(targets, indices):
            if len(i)>0:
                target_boxes.append(t['boxes'][i])
        target_boxes = torch.cat(target_boxes, dim=0)

        if target_boxes.shape[0] == 0:
            losses = {
                "loss_bbox": src_boxes.sum() * 0,
                "loss_giou": src_boxes.sum() * 0,
            }
            return losses

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_vectors" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_vectors"]
        src_boxes = outputs['pred_boxes']
        # TODO use valid to mask invalid areas due to padding in loss
        # target_boxes = torch.cat([box_ops.box_cxcywh_to_xyxy(t['boxes'][i]) for t, (_, i) in zip(targets, indices)], dim=0)
        # target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        
        # also allows for no objects in view
        target_boxes = []
        target_masks_ = []
        valid = []
        for t, (_, i) in zip(targets, indices):
            if len(i)>0:
                target_boxes.append(box_ops.box_cxcywh_to_xyxy(t['boxes'][i]))
            target_masks_.append(t["masks"])
        target_boxes = torch.cat(target_boxes, dim=0)
        target_masks, valid = nested_tensor_from_tensor_list(target_masks_).decompose()

        target_masks = target_masks.to(src_masks)
        src_vectors = src_masks[src_idx]
        src_boxes = src_boxes[src_idx]
        target_masks = target_masks[tgt_idx]
        
        # scale boxes to mask dimesnions
        N, mask_w, mask_h = target_masks.shape
        target_sizes = torch.as_tensor([mask_w, mask_h]).unsqueeze(0).repeat(N, 1).cuda()
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        target_boxes = target_boxes * scale_fct

        # for b in range(target_masks.shape[0]):
        #     # for m in range(target_masks.shape[1]):
        #     plt.figure(1); plt.clf()
        #     plt.imshow(target_masks[b].cpu().numpy())
        #     plt.savefig('images/test.png')
        #     st()

        # for b in range(target_masks.shape[0]):
        #     # for m in range(target_masks.shape[1]):
        #     plt.figure(1)
        #     plt.clf()
        #     mask = np.expand_dims(np.float32(target_masks[b].cpu().numpy()), axis=2).repeat(3,2)
        #     box = target_boxes[b] * mask.shape[0]
        #     mask = cv2.rectangle(mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(0, 1, 0), 1)
        #     plt.imshow(mask)
        #     plt.savefig('images/test.png')
        #     st()


        # crop gt_masks
        n_keep, gt_mask_len = self.processor_dct.n_keep, self.processor_dct.gt_mask_len
        gt_masks = BitMasks(target_masks)
        gt_masks = gt_masks.crop_and_resize(target_boxes, gt_mask_len).to(device=src_masks.device).float()
        target_masks = gt_masks

        # for b in range(target_masks.shape[0]):
        #     # for m in range(target_masks.shape[1]):
        #     plt.figure(1)
        #     plt.clf()
        #     mask = np.expand_dims(np.float32(target_masks[b].cpu().numpy()), axis=2).repeat(3,2)
        #     # box = target_boxes[b] * mask.shape[0]
        #     # mask = cv2.rectangle(mask, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(0, 1, 0), 1)
        #     plt.imshow(mask)
        #     plt.savefig('images/test.png')
        #     st()

        if target_masks.shape[0] == 0:
            losses = {
                "loss_vector": src_vectors.sum() * 0
            }
            return losses

        # perform dct transform
        target_vectors = []
        for i in range(target_masks.shape[0]):
            gt_mask_i = ((target_masks[i,:,:] >= 0.5)* 1).to(dtype=torch.uint8) 
            gt_mask_i = gt_mask_i.cpu().numpy().astype(np.float32)
            coeffs = cv2.dct(gt_mask_i)
            coeffs = torch.from_numpy(coeffs).flatten()
            coeffs = coeffs[torch.tensor(self.processor_dct.zigzag_table)]
            gt_label = coeffs.unsqueeze(0)
            target_vectors.append(gt_label)

        target_vectors = torch.cat(target_vectors, dim=0).to(device=src_vectors.device)
        losses = {}
        if self.no_vector_loss_norm:
            losses['loss_vector'] = self.vector_loss_coef * F.l1_loss(src_vectors, target_vectors, reduction='none').sum() / num_boxes
        else:
            losses['loss_vector'] = self.vector_loss_coef * F.l1_loss(src_vectors, target_vectors, reduction='mean')
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
        # return indices[0]

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
        # return indices[1]

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        if hyp.pred_one_object:
            # for when only predicting one target
            target_idxs = outputs["target_idxs"]

        B = len(targets)
        S = len(targets[0])

        targets_ = []
        indices = []
        for b in range(B):
            for s in range(S):
                if s>0:

                    ######### PREPARE FOR PREDICTING TARGET ONLY ##########
                    if hyp.pred_one_object:
                        # for when only predicting one target
                        t = targets[b][s]
                        f = t["filter_inds"]
                        target_idx = target_idxs[b]
                        where_target_idx = torch.where(f==target_idx)[0]
                        if target_idx not in f:
                            indices.append((torch.tensor([], dtype=torch.long).cuda(), torch.tensor([], dtype=torch.long).cuda()))
                        else:
                            indices.append((torch.tensor([0], dtype=torch.long).cuda(), torch.tensor([0], dtype=torch.long).cuda()))
                        for k in t.keys():
                            if len(where_target_idx)==0:
                                if k=="masks":
                                    t[k] = t[k][0:1] # for mask loss we need to pad here for tgt_idx
                                else:
                                    t[k] = t[k][where_target_idx] # take target
                            else:
                                t[k] = t[k][where_target_idx:where_target_idx+1]
                        targets_.append(t)
                    ######################################################

                    ########## PREPARE FOR PREDICTING ALL OBJECTS (TARGET & SUPPORT) ##########
                    else:
                        targets_.append(targets[b][s])
                        if hyp.filter_inds:
                            indices.append((torch.arange(len(targets[b][s]["filter_inds"])), torch.arange(len(targets[b][s]["filter_inds"]))))
                        else:
                            indices.append((targets[b][s]["filter_inds"], torch.arange(len(targets[b][s]["filter_inds"]))))

                else:
                    if hyp.pred_one_object:
                        # for when only predicting one target
                        # for plotting, also filter first view targets
                        t = targets[b][s]
                        target_idx = target_idxs[b]
                        for k in t.keys():
                            t[k] = t[k][target_idx:target_idx+1]
                    ######################################################
        targets = targets_ # remove first view which is target view

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks' and i < self.vector_start_stage:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            # indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, processor_dct=None):
        super().__init__()
        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, outputs, target_sizes, do_masks=True):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox, out_vector = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_vectors']
        
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        if False:
            prob = out_logits.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 25, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
            if self.processor_dct is not None:
                n_keep = self.processor_dct.n_keep
                vectors = torch.gather(out_vector, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, n_keep))
        else:
            prob = out_logits.sigmoid()
            # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 25, dim=1)

            scores = prob.max(dim=2).values
            # topk_boxes = topk_indexes // out_logits.shape[2]
            labels = prob.max(dim=2).indices
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
            if self.processor_dct is not None:
                n_keep = self.processor_dct.n_keep
                vectors = out_vector #torch.gather(out_vector, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, n_keep))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        if self.processor_dct is not None and do_masks:
            masks = []
            n_keep, gt_mask_len = self.processor_dct.n_keep, self.processor_dct.gt_mask_len
            b, r, c = vectors.shape
            for bi in range(b):
                outputs_masks_per_image = []
                for ri in range(r):
                    # here visual for training
                    idct = np.zeros((gt_mask_len ** 2))
                    idct[:n_keep] = vectors[bi,ri].cpu().numpy()
                    idct = self.processor_dct.inverse_zigzag(idct, gt_mask_len, gt_mask_len)
                    re_mask = cv2.idct(idct)
                    max_v = np.max(re_mask)
                    min_v = np.min(re_mask)
                    re_mask = np.where(re_mask>(max_v+min_v) / 2., 1, 0)
                    re_mask = torch.from_numpy(re_mask)[None].float()
                    outputs_masks_per_image.append(re_mask)
                outputs_masks_per_image = torch.cat(outputs_masks_per_image, dim=0).to(out_vector.device)
                # here padding local mask to global mask
                outputs_masks_per_image = retry_if_cuda_oom(paste_masks_in_image)(
                    outputs_masks_per_image,  # N, 1, M, M
                    boxes[bi],
                    (img_h[bi], img_w[bi]),
                    threshold=0.5,
                )
                outputs_masks_per_image = outputs_masks_per_image.unsqueeze(1).cpu()
                masks.append(outputs_masks_per_image)

        if self.processor_dct is None or not do_masks:
            results1 = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        else:
            results1 = [{'scores': s, 'labels': l, 'boxes': b, 'masks': m} for s, l, b, m in zip(scores, labels, boxes, masks)]

        results = {'pred1':results1}

        return results


class PostProcessSegm(nn.Module):
    def __init__(self, threshold=0.5, processor_dct=None):
        super().__init__()
        self.threshold = threshold
        self.processor_dct = processor_dct

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args, num_classes):
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     num_classes = 250
    device = torch.device(args.device)

    if 'swin' in args.backbone:
        from .swin_transformer import build_swin_backbone
        backbone = build_swin_backbone(args) 
    else:
        backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args) if not args.checkpoint else build_cp_deforamble_transformer(args)
    if args.with_vector:
        processor_dct = ProcessorDCT(args.n_keep, args.gt_mask_len)
    model = SOLQ(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        with_vector=args.with_vector, 
        processor_dct=processor_dct if args.with_vector else None,
        vector_hidden_dim=args.vector_hidden_dim
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_vector"] = 1
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, 
                                                                        with_vector=args.with_vector, 
                                                                        processor_dct=processor_dct if args.with_vector else None,
                                                                        vector_loss_coef=args.vector_loss_coef,
                                                                        no_vector_loss_norm=args.no_vector_loss_norm,
                                                                        vector_start_stage=args.vector_start_stage)
    criterion.to(device)
    # postprocessors = {'bbox': PostProcess(processor_dct=processor_dct if (args.with_vector and args.eval) else None)}
    postprocessors = {'bbox': PostProcess(processor_dct=processor_dct if (args.with_vector) else None)}

    if args.masks: # and args.eval:
        postprocessors['segm'] = PostProcessSegm(processor_dct=processor_dct if args.with_vector else None)
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
