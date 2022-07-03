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

from SOLQ.util import box_ops
from SOLQ.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

import numpy as np
import cv2
from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer_solq_base import build_deforamble_transformer
from .deformable_transformer_cp import build_deforamble_transformer as build_cp_deforamble_transformer
from .dct import ProcessorDCT
from detectron2.structures import BitMasks
from detectron2.layers import paste_masks_in_image
from detectron2.utils.memory import retry_if_cuda_oom
import copy
import functools

import ipdb
st = ipdb.set_trace
from arguments import args
print = functools.partial(print, flush=True)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SOLQ(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, with_vector=False, processor_dct=None, vector_hidden_dim=256, num_classes2=None):
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
        if args.do_predict_oop:
            self.class_embed2 = nn.Linear(hidden_dim, num_classes2)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        if self.with_vector:
            print(f'Training with vector_hidden_dim {vector_hidden_dim}.', flush=True)
            self.vector_embed = MLP(hidden_dim, vector_hidden_dim, self.processor_dct.n_keep, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
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
        if args.do_predict_oop:
            self.class_embed2.bias.data = torch.ones(num_classes2) * bias_value
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
            if args.do_predict_oop:
                self.class_embed2 = _get_clones(self.class_embed2, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            if args.do_predict_oop:
                self.class_embed2 = nn.ModuleList([self.class_embed2 for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if self.with_vector:
            nn.init.constant_(self.vector_embed.layers[-1].bias.data[2:], -2.0)
            self.vector_embed = nn.ModuleList([self.vector_embed for _ in range(num_pred)])

        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            if args.do_predict_oop:
                self.transformer.decoder.class_embed2 = self.class_embed2
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor, return_features=False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
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
        B, C, H, W = samples.shape
        samples = samples.view(B, C, H, W).unbind(0)
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        # hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, _, _ = self.transformer(srcs, masks, pos, query_embeds)
        transformer_out = self.transformer(srcs, masks, pos, query_embeds)
        hs = transformer_out['hs']
        init_reference = transformer_out['init_reference_out']
        inter_references = transformer_out['inter_references_out']
        enc_outputs_class = transformer_out['enc_outputs_class']
        enc_outputs_coord_unact = transformer_out['enc_outputs_coord_unact']
        if args.do_predict_oop:
            enc_outputs_class2 = transformer_out['enc_outputs_class2']

        outputs_classes = []
        outputs_classes2 = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            if args.do_predict_oop:
                outputs_class2 = self.class_embed2[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            if args.do_predict_oop:
                outputs_classes2.append(outputs_class2)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        if args.do_predict_oop:
            outputs_class2 = torch.stack(outputs_classes2) 
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

        if args.do_predict_oop:
            out['pred_logits2'] = outputs_class2[-1]
            out = self._set_aux_loss_addition(out, outputs_class2, 'pred_logits2')

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
            if args.do_predict_oop:
                out['enc_outputs']['pred_logits2'] = enc_outputs_class2

        if return_features:
            features = hs[-1]
            out['features'] = features
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_vector):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_vectors': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_vector[:-1])]

    @torch.jit.unused
    def _set_aux_loss_addition(self, out, outputs_class, outputs_class_name):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        for l in range(len(out['aux_outputs'])):
            out['aux_outputs'][l][outputs_class_name] = outputs_class[l]
        return out


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
                vector_start_stage=0, 
                num_classes2=None
                ):
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
        self.num_classes2 = num_classes2
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
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
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

    def loss_labels2(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits2' in outputs
        src_logits = outputs['pred_logits2']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels2"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes2,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # this handles the no object class
        
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce2': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error2'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
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
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

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

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'labels2': self.loss_labels2,
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

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

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
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks' and i < self.vector_start_stage:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    if loss == 'labels2':
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
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                if loss == 'labels2':
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
    def forward(self, outputs, target_sizes, do_masks=True, return_features=False, features=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox, out_vector = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_vectors'] #, outputs['batch_inds']
        
        # 
        # out_batches = 
        
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        if True:
            prob = out_logits.sigmoid()
            topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 50, dim=1)
            scores = topk_values
            topk_boxes = topk_indexes // out_logits.shape[2]
            labels = topk_indexes % out_logits.shape[2]
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

            if args.do_predict_oop:
                out_logits2 = outputs['pred_logits2']
                prob2 = out_logits2.sigmoid()
                topk_boxes2 = topk_boxes.unsqueeze(2).repeat(1,1,out_logits2.shape[2])
                scores2 = torch.gather(prob2, 1, topk_boxes2)
                labels2 = torch.argmax(scores2, dim=2)
                scores2 = torch.max(scores2, dim=2).values

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
            masks = torch.stack(masks)

        # N = scores.shape[1]
        # _, N, C, H, W = masks.shape
        # # reshape to (B, S-1, ...)
        # scores = scores.reshape(B, S-1, N)
        # labels = labels.reshape(B, S-1, N)
        # boxes = boxes.reshape(B, S-1, N, 4)
        # masks = masks.reshape(B, S-1, N, C, H, W)

        if return_features and features is not None:
            features_keep = torch.gather(features, 1, topk_boxes.unsqueeze(-1).repeat(1,1,features.shape[-1]))


        if self.processor_dct is None or not do_masks:
            results1 = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
            if args.do_predict_oop:
                results2 = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores2, labels2, boxes)]
        else:
            results1 = [{'scores': s, 'labels': l, 'boxes': b, 'masks': m} for s, l, b, m in zip(scores, labels, boxes, masks)]
            if args.do_predict_oop:
                results2 = [{'scores': s, 'labels': l, 'boxes': b, 'masks':m} for s, l, b, m in zip(scores2, labels2, boxes, masks)]

        results = {'pred1':results1}
        if args.do_predict_oop:
            results['pred2'] = results2

        # return results

        if return_features:
            return results, features_keep
        else:
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


def build(args, num_classes, num_classes2):
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
        vector_hidden_dim=args.vector_hidden_dim,
        num_classes2=num_classes2,
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    if args.do_predict_oop:
        weight_dict['loss_ce2'] = args.cls_loss_coef
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
    if args.do_predict_oop:
        losses += ["labels2"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha, 
                                                                        with_vector=args.with_vector, 
                                                                        processor_dct=processor_dct if args.with_vector else None,
                                                                        vector_loss_coef=args.vector_loss_coef,
                                                                        no_vector_loss_norm=args.no_vector_loss_norm,
                                                                        vector_start_stage=args.vector_start_stage,
                                                                        num_classes2=num_classes2)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(processor_dct=processor_dct if (args.with_vector) else None)}

    if args.masks and args.eval:
        postprocessors['segm'] = PostProcessSegm(processor_dct=processor_dct if args.with_vector else None)
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors