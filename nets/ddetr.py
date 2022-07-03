import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from torchvision import models

import ipdb
st = ipdb.set_trace

import sys
import hyperparams as hyp
import argparse

if hyp.ddetr_files:
    sys.path.append('Deformable-DETR')
    from utils.parser import get_args_parser
elif hyp.solq_files:
    sys.path.append('SOLQ')
    from utils.parser_solq import get_args_parser

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as ddetr_utils
# import datasets.samplers as samplers
# from engine import evaluate, train_one_epoch
# if hyp.two_pred_heads:
#     from models import build_two_head as build_model
# else:  
#     from models import build_model
from util import box_ops

from models import build_two_head as build_model

import random



parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

# fix the seed for reproducibility
seed = args.seed + ddetr_utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class DDETR(nn.Module):
	# def __init__(self, fea_size, dropout=False, gate_width=1, use_kernel_function=False):
    def __init__(self, num_classes, load_pretrained=False, num_classes2=None, num_classes_openness=None, no_two_heads=False):
        super(DDETR, self).__init__()

        if hyp.solq_files:
            args.masks = True
            args.with_box_refine = True
            args.meta_arch = 'deformable_detr'
        
        if hyp.do_segmentation:
            args.masks = True
            if hyp.freeze_ddetr_backbone:
                args.frozen_weights = True

        # if no_two_heads:
        #     from models import build_model
        # elif hyp.two_pred_heads:
        #     if hyp.do_objectness:
        #         print("USING OBJECTNESS")
        #         from models import build_two_head_objectness as build_model
        #     else:
        #         from models import build_two_head as build_model
        # else:
        #     from models import build_model


        # print(args)
        # if num_classes2 is not None:
        model, criterion, postprocessors = build_model(args, num_classes, num_classes2, num_classes_openness=num_classes_openness)
        # else:
        #     model, criterion, postprocessors = build_model(args, num_classes)

        if load_pretrained:
            # load pretrained model
            print('loading BASE pretrained ddetr...')
            PATH = '/projects/katefgroup/viewpredseg/checkpoints/ddetr_pretrained/r50_deformable_detr-checkpoint.pth'
            checkpoint = torch.load(PATH)
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['model']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # learn from scratch embeddings with class size info
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "class_embed" not in k}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "query_embed" not in k}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "tgt_embed" not in k}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(pretrained_dict, strict=False)
            print('done')

        # # load pretrained model
        # PATH = '/projects/katefgroup/viewpredseg/checkpoints/ddetr_pretrained/r50_deformable_detr-checkpoint.pth'
        # checkpoint = torch.load(PATH)
        # model_dict = model.state_dict()
        # pretrained_dict = checkpoint['model']
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if "class_embed" not in k}
        # # pretrained_dict = {k: v for k, v in pretrained_dict.items() if "query_embed" not in k}
        # model_dict.update(pretrained_dict) 
        # model.load_state_dict(pretrained_dict, strict=False)

        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors

        # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        # self.coco_evaluator = CocoEvaluator(base_ds, iou_types)

        self.nms_threshold = 0.4

        self.softmax = nn.Softmax(dim=1)

    def forward(self, rgb, targets=None, egomotion=None, batch_tags=None, training=True, summ_writer=None, do_loss=True, return_features=False):

        outputs, hs = self.model(rgb, return_features=True)

        if do_loss:
            loss_dict = self.criterion(outputs, targets)

            weight_dict = self.criterion.weight_dict

            # losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses = []
            for k in loss_dict.keys():
                if k in weight_dict:
                    losses.append(loss_dict[k] * weight_dict[k])
                    # print(k)
                elif ("class_error" in k) or ("cardinality_error" in k):
                    pass
                else:
                    print("LOSS IS ", k)
                    assert(False) # not in weight dict
            losses = sum(losses)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = ddetr_utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()
        else:
            losses = None
            loss_dict_reduced = None
            loss_dict_reduced_unscaled = None
            loss_dict_reduced_scaled = None
            losses_reduced_scaled = None

        # if True:
        #     orig_target_sizes = torch.tensor(rgb.shape[-2:]).unsqueeze(0).repeat(rgb.shape[0], 1).cuda()
        #     results = self.postprocessors['bbox'](outputs, orig_target_sizes)
        #     for i in range(len(results)):
        #         keep, count = box_ops.nms(results[i]['boxes'], results[i]['scores'], self.nms_threshold, top_k=100)
        #         results[i]['nms_boxes'] = results[i]['boxes'][keep]
        out_dict = {}
        out_dict['outputs'] = outputs
        out_dict['losses'] = losses
        out_dict['loss_dict_reduced'] = loss_dict_reduced
        out_dict['loss_dict_reduced_unscaled'] = loss_dict_reduced_unscaled
        out_dict['loss_dict_reduced_scaled'] = loss_dict_reduced_scaled
        out_dict['postprocessors'] = self.postprocessors
        if return_features:
            out_dict['features'] = hs

        return out_dict

        # if return_features:
        #     return outputs, losses, loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, self.postprocessors, hs
        # else:
        #     return outputs, losses, loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, self.postprocessors