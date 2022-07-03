import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import utils.improc
import utils.misc
import utils.basic
from utils.basic import print_stats

import mmdet.models.backbones.resnet as resnet
import mmdet.models.necks.fpn as fpn
import torchvision.models as models
import numpy as np

class CompressNet(nn.Module):
    def __init__(self):
        super(CompressNet, self).__init__()

        print('CompressNet...')

        self.backbone = resnet.ResNet(
            depth=50,
            num_stages=4,
            out_indices=(0,1,2,3),
            frozen_stages=2,
            style='pytorch',
        ).cuda()
        #self.backbone.init_weights(pretrained='torchvision://resnet50')
        self.neck = fpn.FPN(
            in_channels=[256,512,1024,2048],
            out_channels=256,
            start_level=0,
            num_outs=5,
        )
        if False:
            self.net = nn.Conv2d(
                in_channels=1024, 
                out_channels=1,
                # out_channels=128,
                kernel_size=3, 
                stride=1,
                padding=1)
        else:
            # self.net = nn.Conv2d(
            #     in_channels=1024, 
            #     out_channels=128,
            #     kernel_size=2, 
            #     stride=2,
            #     padding=0)

            self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels=1024, 
                    out_channels=512,
                    kernel_size=2, 
                    stride=2,
                    padding=0),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=512, 
                    out_channels=512,
                    kernel_size=2, 
                    stride=2,
                    padding=0),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=512, 
                    out_channels=128,
                    kernel_size=2, 
                    stride=2,
                    padding=0),
            )
            
        print(self.net)
        
        self.dict_len = 1000
        self.neg_pool = utils.misc.SimplePool(self.dict_len, version='pt')
        self.ce = torch.nn.CrossEntropyLoss()

    def compute_seg_loss(self, pred, pos, neg):
        # pred is B x C x H x W
        # seg is B x C x H x W

        loss_im = torch.zeros_like(pred[:,0:1])
        
        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        pos_loss = utils.basic.reduce_masked_mean(loss, pos)
        neg_loss = utils.basic.reduce_masked_mean(loss, neg)

        total_loss = torch.mean(pos_loss + neg_loss)
        return total_loss
        
    def compute_ce_loss(self, emb_q, emb_k, emb_n):
        # print('emb_q', emb_q.shape)
        # print('emb_k', emb_k.shape)
        B, C = list(emb_q.shape)
        
        self.neg_pool.update(emb_n.cpu())
        # print('neg_pool len:', len(self.neg_pool))
        emb_n = self.neg_pool.fetch().cuda()

        N2, C2 = list(emb_n.shape)
        assert (C2 == C)

        # print('emb_q', emb_q.shape)
        # print('emb_k', emb_k.shape)
        
        N = emb_q.shape[0]
        l_pos = torch.bmm(emb_q.view(N,1,-1), emb_k.view(N,-1,1))

        # print('l_pos', l_pos.shape)
        l_neg = torch.mm(emb_q, emb_n.T)
        # print('l_neg', l_neg.shape)
        
        l_pos = l_pos.view(N, 1)
        # print('l_pos', l_pos.shape)
        logits = torch.cat([l_pos, l_neg], dim=1)

        labels = torch.zeros(N, dtype=torch.long).cuda()

        temp = 0.07
        emb_loss = self.ce(logits/temp, labels)
        # print('emb_loss', emb_loss.detach().cpu().numpy())
        return emb_loss
        
    def compute_loss(self, pred_pos, pred_neg):
        pred = torch.cat([pred_pos, pred_neg], dim=0)

        label = torch.cat([torch.ones_like(pred_pos),
                           torch.zeros_like(pred_neg)], dim=0)
        pos = (label==1).float()
        neg = (label==0).float()

        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        pos_loss = utils.basic.reduce_masked_mean(loss, pos)
        neg_loss = utils.basic.reduce_masked_mean(loss, neg)

        balanced_loss = (pos_loss + neg_loss)*0.5

        # return balanced_loss
        # return pos_loss
        return neg_loss

    def merge_feats(self, feats):
        # feat0 = F.interpolate(feats[0], scale_factor=
        feat0 = feats[0]
        feat1 = F.interpolate(feats[1], scale_factor=2, mode='bilinear')
        feat2 = F.interpolate(feats[2], scale_factor=4, mode='bilinear')
        feat3 = F.interpolate(feats[3], scale_factor=8, mode='bilinear')
        super_feat = torch.cat([feat0, feat1, feat2, feat3], dim=1)
        return super_feat

    def forward_on_rgb(self, rgb):
        rgb = utils.improc.back2color(rgb).float().cuda()
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        mean_ = torch.from_numpy(np.array(img_norm_cfg['mean'])).reshape(1, 3, 1, 1).float().cuda()
        std_ = torch.from_numpy(np.array(img_norm_cfg['std'])).reshape(1, 3, 1, 1).float().cuda()
        rgb_ = (rgb - mean_) / std_
        level_outputs = self.backbone(rgb_)
        level_outputs = self.neck(level_outputs)
        super_feat = self.merge_feats(level_outputs)
        # print('super_feat', super_feat.shape)
        compress_feat = self.net(super_feat)
        # print('compress_feat', compress_feat.shape)
        compress_feat = torch.mean(compress_feat, dim=[2,3])
        # print('compress_feat', compress_feat.shape)
        compress_feat = utils.basic.l2_normalize(compress_feat, dim=1)
        return compress_feat

    def dense_forward_on_rgb(self, rgb):
        rgb = utils.improc.back2color(rgb).float().cuda()
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        mean_ = torch.from_numpy(np.array(img_norm_cfg['mean'])).reshape(1, 3, 1, 1).float().cuda()
        std_ = torch.from_numpy(np.array(img_norm_cfg['std'])).reshape(1, 3, 1, 1).float().cuda()
        rgb_ = (rgb - mean_) / std_
        level_outputs = self.backbone(rgb_)
        level_outputs = self.neck(level_outputs)
        super_feat = self.merge_feats(level_outputs)
        # print('super_feat', super_feat.shape)
        compress_feat = self.net(super_feat)
        # print('compress_feat', compress_feat.shape)
        compress_feat = utils.basic.l2_normalize(compress_feat, dim=1)

        # rgb_ torch.Size([1, 3, 512, 768])
        # super_feat torch.Size([1, 1024, 128, 192])
        # compress_feat torch.Size([1, 128, 16, 24])
        
        return compress_feat
        
    # def forward(self, rgb0, rgb1, mask0, summ_writer=None):
    def forward(self, rgb0, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()

        debug_with_seg = False
        if debug_with_seg:
            rgb = utils.improc.back2color(rgb0).float().cuda()
            img_norm_cfg = dict(
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
            mean_ = torch.from_numpy(np.array(img_norm_cfg['mean'])).reshape(1, 3, 1, 1).float().cuda()
            std_ = torch.from_numpy(np.array(img_norm_cfg['std'])).reshape(1, 3, 1, 1).float().cuda()
            rgb_ = (rgb - mean_) / std_
            level_outputs = self.backbone(rgb_)
            level_outputs = self.neck(level_outputs)
            super_feat = self.merge_feats(level_outputs)
            seg = self.net(super_feat)
            seg = F.interpolate(seg, scale_factor=4)

            print('rgb0', rgb0.shape)
            print('seg', seg.shape)
            print('mask0', mask0.shape)
            seg_loss = self.compute_seg_loss(seg, (mask0==1).float(), (mask0==0).float())
            total_loss = utils.misc.add_loss('compress/seg_loss', total_loss, seg_loss, 0.0, summ_writer)
        else:
            feat0 = self.forward_on_rgb(rgb0)
            return feat0
            
            
        
        # rgb = utils.improc.back2color(rgb_both).float().cuda()
        # img_norm_cfg = dict(
        #     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        # mean_ = torch.from_numpy(np.array(img_norm_cfg['mean'])).reshape(1, 3, 1, 1).float().cuda()
        # std_ = torch.from_numpy(np.array(img_norm_cfg['std'])).reshape(1, 3, 1, 1).float().cuda()
        # rgb_ = (rgb - mean_) / std_
        # level_outputs = self.backbone(rgb_)
        # level_outputs = self.neck(level_outputs)
        # super_feat = self.merge_feats(level_outputs)

        # compress_feat = self.net(super_feat)
        # feat0, feat1 = torch.split(compress_feat, 2, dim=0)
        # feat0 = torch.mean(feat0, dim=[1,2])
        # feat1 = torch.mean(feat1, dim=[1,2])
        
        # classif_pos = self.net(torch.cat([super_feat0, super_feat1], dim=1))
        # classif_pos = torch.mean(classif_pos, dim=[1,2])

        # classif_neg = self.net(torch.cat([super_feat0, torch.roll(super_feat1, 1, 0)], dim=1))
        # classif_neg = torch.mean(classif_neg, dim=[1,2])

        # print_stats('classif_pos', classif_pos)
        # # print_stats('classif_neg', classif_neg)

        # classif_loss = self.compute_loss(classif_pos, classif_neg)
        
        

        # classif_sig_pos = F.sigmoid(classif_pos)
        # classif_sig_neg = F.sigmoid(classif_neg)
        # classif_acc_pos = torch.mean((classif_sig_pos > 0.5).float())
        # classif_acc_neg = torch.mean((classif_sig_neg > 0.5).float())
        # classif_acc_bal = (classif_acc_pos + classif_acc_neg)/2.0
        
        # summ_writer.summ_scalar('compress/acc_pos', classif_acc_pos)
        # summ_writer.summ_scalar('compress/acc_neg', classif_acc_neg)
        # summ_writer.summ_scalar('compress/acc_bal', classif_acc_bal)
        
        # return total_loss

