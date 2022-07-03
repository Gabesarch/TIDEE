# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
import functools
print = functools.partial(print, flush=True)

import ipdb
st = ipdb.set_trace

# from .rpe_attention import RPEMultiheadAttention, irpe

import hyperparams as hyp

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, normalize_before=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_feature_levels = num_feature_levels

        if hyp.image_features_self_attend:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
            self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        if hyp.do_deformable_atn_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, dec_n_points)
            self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # learned embeddings for target/supporters
        if hyp.pred_one_object and hyp.use_supporters and hyp.use_target_pos_emb:
            self.target_embed = nn.Parameter(torch.Tensor(1, d_model))
            self.support_embed = nn.Parameter(torch.Tensor(1, d_model))


        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        print(f'Training with {activation}.')

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None,obj_srcs=None, obj_masks=None, obj_pos_embeds=None,B=None,S=None,init_queries=False):
        # assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        if hyp.rel_pos_encodings:
            abs_pos = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if hyp.rel_pos_encodings:
                # if use relative encodings, then we only add to input the level embedding
                lvl_pos_embed = self.level_embed[lvl].view(1, 1, -1).expand(pos_embed.shape[0], pos_embed.shape[1], self.d_model)
                abs_pos.append(pos_embed)
            else:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        if hyp.rel_pos_encodings:
            abs_pos = torch.cat(abs_pos, 1)
        else:
            abs_pos = None

        if hyp.image_features_self_attend:
            # (1) encoder - self attention with image features
            memory = self.encoder(
                src=src_flatten, 
                spatial_shapes=spatial_shapes, 
                level_start_index=level_start_index, 
                valid_ratios=valid_ratios, 
                pos=lvl_pos_embed_flatten, 
                padding_mask=mask_flatten
                )

            # if queries should be obtained after self-attention, then we need to obtain them here
            if hyp.do_self_atn_for_queries and init_queries:
                print("do_self_atn_for_queries")
                B2, E, C = memory.shape

                # only keep from view_2 to view_n for subsequent processing - use view 1 after self atn for queries
                mask_flatten = mask_flatten.view(B,S,E)[:,1:,:] #.contiguous().view(B*(S-1),E)
                lvl_pos_embed_flatten = lvl_pos_embed_flatten.view(B,S,E,C)[:,1:,:] #.contiguous().view(B*(S-1),E,C)
                valid_ratios = valid_ratios.view(B,S,4,2)[:,1:,:] #.contiguous().view(B*(S-1),4,2)
                memory = memory.view(B,S,E,C)
                memory_0 = memory[:,0,:,:].view(B,E,C) # memory frame 0
                memory = memory[:,1:,:,:] #.contiguous().view(B*(S-1),E,C) 
                
                obj_feats = []
                for start_idx in range(self.num_feature_levels):
                    where_masks_lvl = obj_srcs[start_idx]
                    H_l, W_l = spatial_shapes[start_idx]
                    start_ = level_start_index[start_idx]
                    if start_idx==self.num_feature_levels-1:
                        memory_lvl = memory_0[:,start_:,:].view(B,H_l,W_l,C)
                    else:
                        end_ = level_start_index[start_idx+1]
                        memory_lvl = memory_0[:,start_:end_,:].view(B,H_l,W_l,C)
                    obj_feat = []
                    for b in range(B):
                        where_masks_lvl_b = where_masks_lvl[b]
                        num_masks = len(where_masks_lvl_b)
                        obj_feat_ = []
                        for n in range(num_masks):
                            where_masks = where_masks_lvl_b[n]
                            src_0_m = memory_lvl[b,where_masks[0],where_masks[1]].mean(dim=0)
                            obj_feat_.append(src_0_m)
                        num_pad = hyp.max_objs - num_masks
                        pad = torch.zeros(num_pad, self.d_model).cuda()
                        if len(obj_feat_)>0:
                            obj_feat_ = torch.stack(obj_feat_)
                        else:
                            print("NO OBJS")
                            obj_feat_ = torch.tensor(obj_feat_).cuda()
                        obj_feat_ = torch.cat([obj_feat_, pad], dim=0)
                        obj_feat.append(obj_feat_)
                    obj_feats.append(torch.stack(obj_feat))
                obj_feats = torch.cat(obj_feats, dim=2)
                # project multiscale to correct size for query input
                obj_feats = self.multiscale_query_proj(obj_feats)
                obj_srcs = obj_feats
                bs, _, _ = obj_srcs.shape
        else:
            memory = src_flatten

        # seg_memory, seg_mask = memory[:,level_start_index[-1]:,:], mask_flatten[:,level_start_index[-1]:]
        # seg_memory = seg_memory.permute(0,2,1).view(bs,c,h,w)
        # seg_mask = seg_mask.view(bs,h,w)

        if False:
            # prepare obj inputs
            src_flatten_obj = []
            mask_flatten_obj = []
            lvl_pos_embed_flatten_obj = []
            # spatial_shapes = []
            for lvl, (src, mask, pos_embed) in enumerate(zip(obj_srcs, obj_masks, obj_pos_embeds)):
                # bs, c, h, w = src.shape
                # spatial_shape = (h, w)
                # spatial_shapes.append(spatial_shape)
                src = src.flatten(2)#.transpose(1, 2)
                mask = mask.flatten(1)
                pos_embed = pos_embed.flatten(2)#.transpose(1, 2)
                if hyp.pred_one_object and hyp.use_supporters:
                    st()
                    targ_supp_pos_emb = self.support_embed.view(1, 1, -1).expand(pos_embed.shape[0], pos_embed.shape[1], self.d_model)
                    lvl_pos_embed = pos_embed + targ_supp_pos_emb
                else:
                    lvl_pos_embed = pos_embed #+ self.level_embed[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten_obj.append(lvl_pos_embed)
                src_flatten_obj.append(src)
                mask_flatten_obj.append(mask)
            src_flatten_obj = torch.cat(src_flatten_obj, 1)
            mask_flatten_obj = torch.cat(mask_flatten_obj, 1)
            lvl_pos_embed_flatten_obj = torch.cat(lvl_pos_embed_flatten_obj, 1)
        else:
            src_flatten_obj = obj_srcs
            mask_flatten_obj = obj_masks
            if hyp.pred_one_object and hyp.use_supporters and hyp.use_target_pos_emb:
                # add to indicate which object is the target adn which are supporters
                targ_supp_pos_emb = torch.cat([self.target_embed.view(1, 1, -1).expand(obj_pos_embeds.shape[0], 1, self.d_model), self.support_embed.view(1, 1, -1).expand(obj_pos_embeds.shape[0], obj_pos_embeds.shape[1]-1, self.d_model)], dim=1)
                lvl_pos_embed_flatten_obj = obj_pos_embeds + targ_supp_pos_emb
            else:
                lvl_pos_embed_flatten_obj = obj_pos_embeds

        # spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # # prepare input for decoder
        # bs, _, c = memory.shape
        # if self.two_stage:
        #     output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

        #     # hack implementation for two-stage Deformable DETR
        #     enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        #     enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

        #     topk = self.two_stage_num_proposals
        #     topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        #     topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        #     topk_coords_unact = topk_coords_unact.detach()
        #     reference_points = topk_coords_unact.sigmoid()
        #     init_reference_out = reference_points
        #     pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        #     query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        # else:
        #     query_embed, tgt = torch.split(query_embed, c, dim=1)
        #     query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        #     tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        #     reference_points = self.reference_points(query_embed).sigmoid()
        #     init_reference_out = reference_points

        # (2) cross attention with object features from frame 1
        # reference_points = self.reference_points(src_flatten_obj).sigmoid()
        # init_reference_out = reference_points
        hs = []
        batch_inds = []
        for view in range(S-1):
            memory_ = memory[:,view,:,:] # take next view
            mask_flatten_ = mask_flatten[:,view,:]
            lvl_pos_embed_flatten_ = lvl_pos_embed_flatten[:,view,:]
            # (3) decoder - for decoding 
            if hyp.do_deformable_atn_decoder:
                # hs_, inter_references = self.decoder(
                #     tgt=src_flatten_obj, 
                #     reference_points=reference_points, 
                #     src=memory,
                #     src_spatial_shapes=spatial_shapes, 
                #     src_level_start_index=level_start_index, 
                #     src_valid_ratios=valid_ratios, 
                #     query_pos=lvl_pos_embed_flatten_obj, 
                #     src_padding_mask=mask_flatten # pad tokens
                #     )
                # inter_references_out = inter_references
                assert(False) # not implemented yet
            else:
                # query_embed = src_flatten_obj
                # tgt = torch.zeros_like(query_embed)
                hs_ = self.decoder(
                    tgt=src_flatten_obj, # query feature input
                    memory=memory_, # memory feature input
                    memory_key_padding_mask=mask_flatten_, # input mask for memory
                    pos=lvl_pos_embed_flatten_, # positional embedding for memory
                    tgt_mask=None, # attention mask for query - restricts key/value pairs - usually not needed
                    tgt_key_padding_mask=mask_flatten_obj, # key mask for query 
                    query_pos=lvl_pos_embed_flatten_obj, # positional embedding for query
                    )
                src_flatten_obj = hs_[-1] # intiialize next view queries as last view's output queries
                hs.append(hs_)
                batch_inds.append(torch.arange(B).cuda())

        # placeholders since we dont use deformable atn and many queries
        init_reference_out = None
        inter_references_out = None
        enc_outputs_class = None
        enc_outputs_coord_unact = None

        hs = torch.cat(hs, dim=1)
        batch_inds = torch.cat(batch_inds, dim=0)

        if self.two_stage:
            return hs, batch_inds, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact #, seg_memory, seg_mask
        return hs, batch_inds, init_reference_out, inter_references_out, None, None #, seg_memory, seg_mask

############# DECODER ###############
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        if hyp.rel_pos_encodings:
            ratio=2.0
            method='product'
            mode='contextual'
            shared_head=True
            rpe_on='k'
            rpe_config = irpe.get_rpe_config(
                    ratio=ratio,
                    method=method,
                    mode=mode,
                    shared_head=shared_head,
                    skip=0,
                    rpe_on=rpe_on,
                )
            if hyp.query_self_attn:  
                self.self_attn = RPEMultiheadAttention(d_model, nhead, dropout=dropout, rpe_config=rpe_config)
            self.multihead_attn = RPEMultiheadAttention(d_model, nhead, dropout=dropout, rpe_config=rpe_config)
        else:
            if hyp.query_self_attn: 
                self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            else:
                print("NOT DOING QUERY SELF ATTENTION")
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        if hyp.query_self_attn: 
            # print("Queries self attending")
            q = k = self.with_pos_embed(tgt, query_pos)
            if hyp.rel_pos_encodings:
                tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), value=tgt.transpose(0, 1), attn_mask=tgt_mask,
                                    key_padding_mask=tgt_key_padding_mask)[0]
            else:
                tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), value=tgt.transpose(0, 1), attn_mask=tgt_mask,
                                    key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2.transpose(0, 1))
            tgt = self.norm1(tgt)

        # print("Queries cross attending to image")
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos).transpose(0, 1),
                                key=self.with_pos_embed(memory, pos).transpose(0, 1),
                                value=memory.transpose(0, 1), attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2.transpose(0, 1))
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    # def forward_pre(self, tgt, memory,
    #                 tgt_mask: Optional[Tensor] = None,
    #                 memory_mask: Optional[Tensor] = None,
    #                 tgt_key_padding_mask: Optional[Tensor] = None,
    #                 memory_key_padding_mask: Optional[Tensor] = None,
    #                 pos: Optional[Tensor] = None,
    #                 query_pos: Optional[Tensor] = None):
    #     tgt2 = self.norm1(tgt)
    #     q = k = self.with_pos_embed(tgt2, query_pos)
    #     tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
    #                           key_padding_mask=tgt_key_padding_mask)[0]
    #     tgt = tgt + self.dropout1(tgt2)
    #     tgt2 = self.norm2(tgt)
    #     tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
    #                                key=self.with_pos_embed(memory, pos),
    #                                value=memory, attn_mask=memory_mask,
    #                                key_padding_mask=memory_key_padding_mask)[0]
    #     tgt = tgt + self.dropout2(tgt2)
    #     tgt2 = self.norm3(tgt)
    #     tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
    #     tgt = tgt + self.dropout3(tgt2)
    #     return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

############# DEFORMABLE ENCODER ###############
class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src



############# DEFORMABLE DECODER ###############
class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                            reference_points,
                            src, src_spatial_shapes, level_start_index, src_padding_mask)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def swish(x):
    return x * torch.sigmoid(x)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "silu":
        return swish
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries)


