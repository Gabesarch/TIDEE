import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")
import utils.geom
import utils.misc
import utils.basic

import mmdet.models.backbones.resnet as resnet
import mmdet.models.necks.fpn as fpn

import torchvision.models as models

EPS = 1e-4
class Seg2dNet(nn.Module):
    def __init__(self, num_classes=2):
        super(Seg2dNet, self).__init__()

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
        self.num_classes = num_classes
        self.net = nn.Conv2d(
            in_channels=1024, 
            out_channels=self.num_classes,
            kernel_size=1, 
            stride=1)
        print(self.net)

    def compute_loss(self, pred, seg, valid, hard):
        # pred is B x C x H x W
        # seg is B x C x H x W
        # valid is B x C x H x W

        target = torch.max(seg, dim=1)[1]
        
        # loss = F.cross_entropy(pred, target, reduction='none')

        loss_im = torch.zeros_like(pred[:,0:1])
        
        losses = []
        # next i want to gather up the loss for each valid class, and balance these into a total
        for cls in list(range(self.num_classes)):
            pos = seg[:,cls]
            # neg = 1.0 - pos

            weights2d = torch.ones(1, 1, 3, 3, device=torch.device('cuda'))
            pos_wide = F.conv2d(pos.unsqueeze(1), weights2d, padding=1).clamp(0, 1).squeeze(1)
            neg = 1.0 - pos_wide
            
            har = hard[:,cls]
            val = valid[:,cls]
            pre = pred[:,cls]

            label = pos*2.0 - 1.0
            a = -label * pre
            b = F.relu(a)
            loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))
            
            # print('loss', loss.shape)
            # print('pos', pos.shape)
            # print('neg', neg.shape)
            # print('valid', valid.shape)
            # print('hard', hard.shape)
            
            pos_loss = utils.basic.reduce_masked_mean(loss, pos*val)
            neg_loss = utils.basic.reduce_masked_mean(loss, neg*val)
            hard_loss = utils.basic.reduce_masked_mean(loss, har*val)

            if torch.sum(pos*val) >= 1:
                losses.append(pos_loss)
                loss_im = loss_im + (loss*pos*val).unsqueeze(1)
            if torch.sum(neg*val) >= 1:
                losses.append(neg_loss)
                loss_im = loss_im + (loss*neg*val).unsqueeze(1)
            if torch.sum(hard*val) >= 1:
                losses.append(hard_loss)
                loss_im = loss_im + (loss*har*val).unsqueeze(1)
        total_loss = torch.mean(torch.stack(losses))
        return total_loss, loss_im
    
    def merge_feats(self, feats):
        # feat0 = F.interpolate(feats[0], scale_factor=
        feat0 = feats[0]
        feat1 = F.interpolate(feats[1], scale_factor=2, mode='bilinear')
        feat2 = F.interpolate(feats[2], scale_factor=4, mode='bilinear')
        feat3 = F.interpolate(feats[3], scale_factor=8, mode='bilinear')
        super_feat = torch.cat([feat0, feat1, feat2, feat3], dim=1)
        return super_feat
        
    def forward(self, rgb, seg_g=None, valid=None, hard=None, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()
        # B, C, H, W = list(feat.shape)
        # seg_e = self.net(feat)
        
        # print('seg_g', seg_g.shape)
        # print('valid', valid.shape)

        # for ind, feat in enumerate(feats):
        #     print('ori shape at %d' % ind, feat.shape)
        
        B = rgb.shape[0]

        if summ_writer is not None:
            summ_writer.summ_rgb('seg2d/input', rgb)
        
        rgb = utils.improc.back2color(rgb).float().cuda()
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        mean_ = torch.from_numpy(np.array(img_norm_cfg['mean'])).reshape(1, 3, 1, 1).float().cuda()
        std_ = torch.from_numpy(np.array(img_norm_cfg['std'])).reshape(1, 3, 1, 1).float().cuda()
        rgb_ = (rgb - mean_) / std_
        level_outputs = self.backbone(rgb_)
        level_outputs = self.neck(level_outputs)
        super_feat = self.merge_feats(level_outputs)
        
        # for ind, feat in enumerate(new_feats):
        #     print('new shape at %d' % ind, feat.shape)
        #     summ_writer.summ_feat('seg2d/feat_input_%d' % ind, feat, pca=True)

        # super_feat = torch.cat(new_feats, dim=1)
        # print('super_feat', super_feat.shape)
        # seg_e_preds = []
        
        # for idx in list(range(len(new_feats))):
        #     seg_e = self.net(new_feats[idx])
        #     print('seg_e', seg_e.shape)
        #     seg_e_preds.append(seg_e)
        # seg_e = 
        #     # print('ins_pred_', ins_pred_.shape)
        #     # print('cate_pred_', cate_pred_.shape)
        #     ins_pred.append(ins_pred_)
        #     cate_pred.append(cate_pred_)
        seg_e = self.net(super_feat)
        # print('seg_e', seg_e.shape)

        # smooth loss
        dy, dx = utils.basic.gradient2d(seg_e, absolute=True)
        smooth_im = torch.mean(dy+dx, dim=1, keepdims=True)
        if summ_writer is not None:
            summ_writer.summ_oned('seg2d/smooth_loss', smooth_im)
        smooth_loss = torch.mean(smooth_im)
        total_loss = utils.misc.add_loss('seg2d/smooth_loss', total_loss, smooth_loss, 0.1, summ_writer)
        
        if seg_g is not None:
            # print('seg_g', seg_g.shape)
            # print('valid', valid.shape)
            # print('hard', hard.shape)
            prob_loss, loss_im = self.compute_loss(seg_e, seg_g, valid, hard)
            total_loss = utils.misc.add_loss('seg2d/all_prob_loss', total_loss, prob_loss, 1.0, summ_writer)
        
        if summ_writer is not None and summ_writer.save_this:
            # contrary to the typical vis, what i want here is:
            # if the confidence of a class is above some thresh, show it
            seg_e_sig = F.sigmoid(seg_e)
            for k in list(range(self.num_classes)):
                summ_writer.summ_oned('seg2d/seg_e_%d' % k, seg_e_sig[:,k:k+1], norm=False)


            if seg_g is not None:
                # assume free_g and valid are also not None

                # collect some accuracy stats
                seg_e_binary = seg_e_sig.round()
                fg_match = ((seg_g==1.0)*(seg_e_binary==1.0)).float()
                bg_match = ((seg_g==0.0)*(seg_e_binary==0.0)).float()
                acc_fg = utils.basic.reduce_masked_mean(fg_match, valid)
                acc_bg = utils.basic.reduce_masked_mean(bg_match, valid)
                acc_bal = (acc_fg + acc_bg)*0.5

                if summ_writer is not None:
                    summ_writer.summ_scalar('unscaled_seg2d/acc_fg', acc_fg.cpu().item())
                    summ_writer.summ_scalar('unscaled_seg2d/acc_bg', acc_bg.cpu().item())
                    summ_writer.summ_scalar('unscaled_seg2d/acc_bal', acc_bal.cpu().item())
                

            utils.basic.print_stats('seg_e_sig', seg_e_sig)
            # seg_e_sig = F.interpolate(seg_e_sig, scale_factor=4, mode='bilinear')
            for thr in [0.8, 0.9, 0.95]:
                seg_e_hard = (seg_e_sig > thr).float()
                single_class = (torch.sum(seg_e_hard, dim=1, keepdim=True)==1).float()
                seg_e_hard = seg_e_hard * single_class
                bkg = torch.sum(seg_e_hard, dim=1, keepdim=True)==0
                seg_e_vis = torch.cat([bkg, seg_e_hard], dim=1)
                seg_e_vis = torch.max(seg_e_vis, dim=1)[1]
                summ_writer.summ_seg('seg2d/all_seg_e_%.2f' % thr, seg_e_vis)
            # summ_writer.summ_soft_seg('seg2d/seg_e_alt', F.softmax(seg_e, dim=1))

            if seg_g is not None:
                summ_writer.summ_oned('seg2d/loss', loss_im)
                
                bkg = torch.sum(seg_g, dim=1, keepdim=True)==0
                seg_g_vis = torch.cat([bkg, seg_g], dim=1)
                seg_g_vis_fullres = F.interpolate(seg_g_vis, scale_factor=4, mode='bilinear')
                seg_g_vis = torch.max(seg_g_vis, dim=1)[1]
                seg_g_vis_fullres = torch.max(seg_g_vis_fullres, dim=1)[1]
                summ_writer.summ_seg('seg2d/all_seg_g', seg_g_vis)
                # summ_writer.summ_seg('seg2d/all_seg_g_fullres', seg_g_vis_fullres)

                for k in list(range(self.num_classes)):
                    pos = seg_g[:,k:k+1]
                    neg = 1.0 - pos
                    val = valid[:,k:k+1]
                    # summ_writer.summ_oned('seg2d/label_%d_pos' % k, pos*val, norm=False)
                    # summ_writer.summ_oned('seg2d/label_%d_neg' % k, neg*val, norm=False)
                    # summ_writer.summ_oned('seg2d/label_%d_val' % k, val, norm=False)
                    summ_writer.summ_oned('seg2d/label_%d' % k, (pos-neg*2)*val, norm=True)
                
        return total_loss, seg_e
    
