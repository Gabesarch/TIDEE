
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import cv2
import ipdb
st = ipdb.set_trace
import torchvision.transforms as T
import math
import numbers
import matplotlib.pyplot as plt
import utils.geom
import utils.vox
from scipy.ndimage import gaussian_filter
from arguments import args
import scipy

class VSN(nn.Module):
    def __init__(
        self, num_classes, 
        # rot_to_ind, hor_to_ind, ind_to_rot, ind_to_hor, 
        include_rgb, 
        fov, Z,Y,X, class_to_save_to_id=None,
        do_masked_pos_loss=True
        ):
        """
        Args:
            num_classes (int): number of classes
            # rot_to_ind: yaw rotation to index dict
            # hor_to_ind: pitch rotation to index dict
            # ind_to_rot: index to yaw rotation dict
            # ind_to_hor: index to pitch rotation dict
            include_rgb: include rgb in feature map (note: memory intensive)
            fov: fov of agent
            Z: width of occupancy 
            Y: height of occupancy
            X: length of occupancy
        """
        super(VSN, self).__init__()

        self.class_to_save_to_id = class_to_save_to_id
        self.num_classes = num_classes

        # self.include_rgb = include_rgb

        assert(not (args.include_rgb and args.do_add_semantic)) # TODO: no support for both currently 

        if args.include_rgb:

            resnet = models.resnet50(pretrained=True).cuda()

            self.resnet_back = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1,
            )
            self.layer2 = resnet.layer2

            # freeze whole thing

            for param in self.resnet_back.parameters():
                param.requires_grad = False

            for param in self.layer2.parameters():
                param.requires_grad = False

            self.channel_red_layer = nn.Sequential(
                nn.Conv2d(512, 256, 3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(256, 128, 3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.LeakyReLU(),
            )

        if args.include_rgb:
            chan_in = 64
        else:
            chan_in = 1

        if args.do_add_semantic:
            chan_in = num_classes

        self.out_dim = 128
        self.depth_agg_layer = nn.Sequential(
            nn.Conv2d(chan_in*Y, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(1024, self.out_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

        self.pos_conv3d = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.out_dim, self.out_dim, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.out_dim, 1, 1, stride=1),
        )

        self.softmax = nn.Softmax(dim=1)

        self.pre_mean=torch.as_tensor([0.485, 0.456, 0.406]).cuda().reshape(1,3,1,1)
        self.pre_std=torch.as_tensor([0.229, 0.224, 0.225]).cuda().reshape(1,3,1,1)

        # bounds for voxel grid

        XMIN = args.voxel_min #-4.0 # right (neg is left)
        XMAX = args.voxel_max #4.0 # right
        YMIN = args.voxel_min #-4.0 # down (neg is up)
        YMAX = args.voxel_max #4.0 # down
        ZMIN = args.voxel_min #-4.0 # forward
        ZMAX = args.voxel_max #4.0 # forward
        self.bounds = torch.tensor([XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX]).cuda()

        ####%%% for object attention
        dim=self.out_dim*self.out_dim
        self.obj_attention = nn.Embedding(num_classes, dim)

        self.pos_coeff = 1.0
        # self.yaw_coeff = 0.7
        # self.pitch_coeff = 0.7

        self.pos_threshold = 0.5 # threshold for inference on positions
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        # huge positive to negaitve class imbalance here...
        self.num_positive_on_average = args.num_positive_on_average

        self.fov = fov

        # self.max_positions = 50 # maximum number of output positions from inference

        self.mask_position_loss = do_masked_pos_loss

        # augmentations

        self.color_jit = T.Compose([T.ToPILImage(),T.ColorJitter(brightness=.5, hue=.3, saturation=0.7), T.ToTensor()])
        self.guassian_noise = AddGaussianNoise(mean=0., std=0.1)
        
        # self.aug_transforms = T.RandomApply(torch.nn.ModuleList([
        #     T.ColorJitter(brightness=.5, hue=.3, saturation=0.7),
        #     AddGaussianNoise(mean=0., std=0.1)
        #     ]), p=0.5)
    
    def forward(self, images_batch, xyz_batch, origin_T_camXs_batch, Z,Y,X, vox_util_batch, targets_batch, objects_track_dict_batch=None, mode='train', summ_writer=None, do_inference=False, do_loss=True, vis=None):
        """
        Args:
        batch indicates list where each element is batch with parameters indicated below
            images: ego-centric views of the object out of place (S, 3, H, W)
            xyz: point cloud (S,N,3)
            pix_T_camX: camera proj matrix (4,4) 
            Z,Y,X: occupancy
            vox_util: occupancy util 
            origin_T_camXs_batch: Sx4x4 N=number of views - tranformation matrix from scene origin to camera 
            targets: targets from prepare_supervision()
        """

        total_loss = torch.tensor(0.0).cuda()

        forward_dict = self.vsn_features_forward(
            images_batch, 
            xyz_batch, 
            origin_T_camXs_batch, 
            # pix_T_camX, 
            Z,Y,X, 
            vox_util_batch, 
            targets_batch, 
            objects_track_dict_batch,
            mode, summ_writer, vis=vis
            )

        if do_loss:
            # huge positive to negaitve class imbalance here... weight = neg/pos
            feat_pos_logits = forward_dict['feat_pos_logits']
            B,N,C,H,W = feat_pos_logits.shape
            feat_pos_logits = feat_pos_logits.view(B*N,C,H,W)


            if self.mask_position_loss:

                def balanced_ce_loss(pred, gt, valid):
                    pos = (gt > 0.95).float()
                    neg = (gt < 0.05).float()

                    label = pos*2.0 - 1.0
                    a = -label * pred
                    b = F.relu(a)
                    loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

                    pos_loss = utils.basic.reduce_masked_mean(loss, pos*valid)
                    neg_loss = utils.basic.reduce_masked_mean(loss, neg*valid)

                    balanced_loss = pos_loss + neg_loss

                    return balanced_loss

                # since supervision is sparse, dont apply loss in empty regions directly around supervision points
                fat_gt_map = utils.improc.dilate2d(forward_dict['pos_targets'])
                ignore_mask = fat_gt_map - forward_dict['pos_targets']
                keep_mask = 1 - ignore_mask
                keep_mask = keep_mask.clamp(0,1)

                position_loss = balanced_ce_loss(feat_pos_logits, forward_dict['pos_targets'], keep_mask)

            else:
                pos_targets = forward_dict['pos_targets']

                visualize_supervision = False
                if visualize_supervision:
                    plt.figure()
                    plt.imshow(pos_targets[0,0].cpu().numpy())
                    plt.savefig('images/test.png')
                    st()
                # pos_targets = torch.from_numpy(pos_targets).cuda()
                pos_weight = [(feat_pos_logits.shape[-1]*feat_pos_logits.shape[-2] - self.num_positive_on_average)/self.num_positive_on_average]
                position_loss = F.binary_cross_entropy_with_logits(feat_pos_logits, pos_targets, reduction='mean',pos_weight=torch.tensor(pos_weight).cuda())

            if summ_writer is not None:
                # summ_writer should be Summ_writer object in utils.improc
                summ_writer.summ_scalar(f'{mode}/unscaled_position', position_loss)
                summ_writer.summ_scalar(f'{mode}/scaled_position', self.pos_coeff*position_loss)
            total_loss = total_loss + self.pos_coeff*position_loss

        forward_dict['total_loss'] = total_loss

        return forward_dict

    def vsn_features_forward(
        self, 
        images_batch, 
        xyz_batch, 
        origin_T_camXs_batch, 
        Z,Y,X, 
        vox_util_batch, 
        targets_batch, 
        objects_track_dict_batch=None, 
        mode='train', 
        summ_writer=None, 
        vis=None
        ):
        
        forward_dict = {}

        batch_size = len(images_batch)

        if args.include_rgb:
            images_per_batch = np.array([len(images_batch[i]) for i in range(batch_size)])
            images = torch.cat(images_batch, dim=0)

            if mode=='train' and args.do_augmentations:
                if np.random.uniform()>0.5:
                    images = torch.stack([self.color_jit(image.cpu()) for image in images], dim=0).cuda()
                if np.random.uniform()>0.5:
                    images = self.guassian_noise(images)

                if summ_writer is not None:
                    if summ_writer.save_this:
                        summ_writer.summ_rgb('train/rgb', images[0:1]-0.5) # improc takes images -0.5

            ########%%%%%%%%% Featurize images
            # normalize images
            feats = (images - self.pre_mean) / self.pre_std
            
            feats = self.resnet_back(feats)
            feats = self.layer2(feats)
            # feats = self.layer3(feats)
            # feats = self.up_layer(feats)
            # feats = self.upsample(feats)
            feats = self.channel_red_layer(feats)

            _,C,H,W = feats.shape

            pix_T_camX = self.get_pix_to_camX(H,W)

            images_per_batch_stop = np.concatenate([np.array([0]), np.cumsum(images_per_batch)])
            feats_batch = [feats[images_per_batch_stop[i]:images_per_batch_stop[i+1]] for i in range(len(images_per_batch))]
        else:
            C = 1

        if args.do_add_semantic:
            C = self.num_classes
            smoothing = GaussianSmoothing(channels=1, kernel_size=3, sigma=3, dim=3).cuda()

        feat_memX0 = []
        attention = []
        pos_targets = []
        batch_inds = []
        occ_camX0s_vis = []
        for b in range(batch_size):
            origin_T_camXs = origin_T_camXs_batch[b]
            xyz = xyz_batch[b]
            vox_util = vox_util_batch[b]
            ############%%%%%%% back project features to 3d occupancy map
            
            camX0s_T_camXs = utils.geom.get_camM_T_camXs(origin_T_camXs.unsqueeze(0), ind=0).squeeze(0)
            camXs_T_camX0s = utils.geom.safe_inverse(camX0s_T_camXs)
            xyz_camX0s = utils.geom.apply_4x4(camX0s_T_camXs, xyz)

            occ_camX0s = vox_util.voxelize_xyz(xyz_camX0s, Z, Y, X, assert_cube=False, device='cuda').unsqueeze(1)


            # make sure this works iwth the increased channels
            B = targets_batch[b]['B'] # number of objects
            if b>0:
                assert(B==B_prev) # number of objects should not change

            if summ_writer is not None:
                if summ_writer.save_this:
                    occ_camX0s_vis_ = vox_util.voxelize_xyz(xyz_camX0s, Z, 4, X, assert_cube=False).unsqueeze(1)
                    occ_camX0s_vis_ = torch.max(occ_camX0s_vis_, dim=0).values
                    # occ_camX0s_vis_ = occ_camX0s_vis_[:,:,:,:,:]
                    occ_camX0s_vis_ = summ_writer.summ_occ(f'', occ_camX0s_vis_, only_return=True)
                    occ_camX0s_vis.append(occ_camX0s_vis_.repeat(B,1,1,1))
            
            if summ_writer is None and vis is not None:
                occ_camX0s_vis_ = vox_util.voxelize_xyz(xyz_camX0s, Z, 4, X, assert_cube=False).unsqueeze(1)
                occ_camX0s_vis_ = torch.max(occ_camX0s_vis_, dim=0).values
                # occ_camX0s_vis_ = occ_camX0s_vis_[:,:,:,:,:]
                occ_camX0s_vis_ = summ_writer.summ_occ(f'', occ_camX0s_vis_, only_return=True)
                occ_camX0s_vis.append(occ_camX0s_vis_.repeat(B,1,1,1))


            # do I need unsqueeze?
            if args.include_rgb:
                feats_ = feats_batch[b]
                pix_T_camX_ = pix_T_camX.unsqueeze(0).repeat(camX0s_T_camXs.shape[0],1,1)
                pixX_T_camX0 = utils.basic.matmul2(pix_T_camX_, camXs_T_camX0s)
                feat_memX0_ = vox_util.unproject_rgb_to_mem(feats_, Z, Y, X, pixX_T_camX0, assert_cube=False).unsqueeze(1)
                feat_memX0_ = utils.basic.reduce_masked_mean(feat_memX0_, occ_camX0s, dim=0)
            else:
                feat_memX0_ = torch.max(occ_camX0s, dim=0).values

            if args.do_add_semantic:
                save_xyz = False
                camX0_T_origin = utils.geom.safe_inverse_single(origin_T_camXs[0])
                objects_track_dict = objects_track_dict_batch[b]
                # semantic_map = torch.ones((self.num_classes, Z,Y,X)).cuda()

                # get centroids and labels to add to map
                # adapted from get_centroids_and_labels() in object_tracker
                labels = []
                for key in list(objects_track_dict.keys()):
                    cur_dict = objects_track_dict[key]
                    labels.append(cur_dict['label'])
                labels = list(set(labels))
                centroids = {}
                for label in labels:
                    centroids[label] = []
                for key in list(objects_track_dict.keys()):
                    cur_dict = objects_track_dict[key]
                    centroids[cur_dict['label']].append(cur_dict['locs'])

                feat_memX0_ = feat_memX0_.repeat(1,self.num_classes,1,1,1)
                
                # build semantic map
                if save_xyz:
                    positions_objects =[]
                for key in list(centroids.keys()):
                    if key not in self.class_to_save_to_id:
                        continue
                    if len(centroids[key])==0:
                        continue
                    # get centroids from memory
                    positions = torch.from_numpy(np.array(centroids[key])).cuda().unsqueeze(0).float()
                    xyz_pos_camX0 = utils.geom.apply_4x4(camX0_T_origin.unsqueeze(0), positions)
                    occ_pos_camX0, vox_inds_expand, vox_inds = vox_util.voxelize_xyz(xyz_pos_camX0, Z, Y, X, return_inds=True, assert_cube=False)
                    if torch.sum(occ_pos_camX0)==0:
                        continue
                    # smooth out centroids
                    occ_pos_camX0 = smoothing(occ_pos_camX0)
                    occ_pos_camX0 = (occ_pos_camX0/torch.max(occ_pos_camX0))*2 # normalize so guassian addition is not so small

                    id_ind = self.class_to_save_to_id[key]
                    # semantic_map[id_ind,:,:,:] = semantic_map[id_ind,:,:,:] + occ_pos_camX0.squeeze(0).squeeze(0)
                    feat_memX0_[:,id_ind,:,:,:] = feat_memX0_[:,id_ind,:,:,:] + occ_pos_camX0.squeeze(0)
                    if save_xyz:
                        positions_objects.append(xyz_pos_camX0)

                    visualize_semantic_channel = False
                    if visualize_semantic_channel:
                        plt.figure()
                        plt.imshow(torch.max(feat_memX0_[:,id_ind,:,:,:], dim=2).values.squeeze().cpu().numpy())
                        plt.colorbar()
                        plt.savefig('images/test3.png')
                        st()

                if save_xyz:
                    # for saving out point cloud visualization
                    object_pos = torch.cat(positions_objects,dim=0).squeeze().cpu().numpy()
                    xyz_camX0s = xyz_camX0s.cpu().numpy()
                    with open('images/xyz.npy', 'wb') as f:
                        np.save(f, xyz_camX0s)
                    with open('images/object_pos.npy', 'wb') as f:
                        np.save(f, object_pos)
                    st()
                    xyz = xyz_camX0s.reshape(xyz_camX0s.shape[0]*xyz_camX0s.shape[1], 3)
                    points = np.vstack((xyz[:,0], xyz[:,1], xyz[:,2])).transpose()
                    points = points[points[:,0]<10,:]
                    points = points[points[:,1]<4,:]
                    points = points[points[:,2]<10,:]
                    points = points[points[:,0]>-10,:]
                    points = points[points[:,1]>-10,:]
                    points = points[points[:,2]>-10,:]
            
            feat_memX0_ = feat_memX0_.repeat(B,1,1,1,1)
            feat_memX0.append(feat_memX0_)

            obj_ids = targets_batch[b]['obj_ids'].cuda().long()
            attention_ = self.obj_attention(obj_ids).view(B,self.out_dim,self.out_dim)
            attention.append(attention_)

            # position supervision 
            pos_targets_= targets_batch[b]['occ_sup_camX0']
            # collapse height
            pos_targets.append(pos_targets_)
            batch_inds.append(torch.ones(B)*b)
            
            B_prev = B

        num_objects = B

        feat_memX0 = torch.cat(feat_memX0, dim=0)
        attention = torch.cat(attention, dim=0)
        pos_targets = torch.cat(pos_targets, dim=0)
        batch_inds = torch.cat(batch_inds, dim=0)

        if summ_writer is not None:
            if summ_writer.save_this:
                if args.include_rgb:
                    summ_writer.summ_feat(f'{mode}/feat_memX0', feat_memX0, pca=True, only_return=False)
                else:
                    summ_writer.summ_feat(f'{mode}/feat_memX0', feat_memX0, pca=False, only_return=False)
                occ_agg = torch.max(occ_camX0s, dim=0).values
                summ_writer.summ_occ(f'{mode}/occX0', occ_agg)
                
                occ_camX0s_vis = torch.cat(occ_camX0s_vis, dim=0)
                forward_dict['occ_camX0s_vis'] = occ_camX0s_vis

        if vis is not None:
            occ_camX0s_vis = torch.cat(occ_camX0s_vis, dim=0)
            forward_dict['occ_camX0s_vis'] = occ_camX0s_vis



        # channel dimension!!
        B2,_,_,_,_ = feat_memX0.shape
        feat_memX0 = feat_memX0.view(B2,C,Z,Y,X)

        if False:
            # more visualization
            # dense_xyz_camX0s = utils.geom.apply_4x4(camX0s_T_camXs, xyz)
            # xyz_acc = dense_xyz_camX0s.reshape(1, dense_xyz_camX0s.shape[0]*dense_xyz_camX0s.shape[1], 3)
            # occXs = vox_util.voxelize_xyz(xyz_acc, Z, Y, X).squeeze()
            # occXs = torch.max(occXs, dim=0).values.squeeze(0)
            # occ_X0 = torch.max(occ_X0, dim=1).values
            # occ_X0 = occ_X0.cpu().numpy()
            # for i in range(len(occXs)):
            vis = torch.max(feat_memX0, dim=0).values
            vis = torch.max(vis, dim=1).values
            vis = vis.detach().cpu().numpy()
            plt.figure()
            plt.imshow(vis)
            plt.savefig(f'images/test.png')
            st()

        #C*vertical_dim
        feat_memX0 = feat_memX0.permute(0,1,3,2,4) # aggregate over depth channel
        feat_memX0 = feat_memX0.contiguous().view(B2,C*Y,Z,X)
        feat_memX0 = self.depth_agg_layer(feat_memX0)

        if summ_writer is not None:
            summ_writer.summ_feat(f'{mode}/feat_memX0_depth_agg', feat_memX0, pca=True)

        ####%%%%%%%%% Apply object attention        
        feat_memX0 = feat_memX0.view(B2,self.out_dim,-1)    
        feat_memX0 = torch.bmm(attention, feat_memX0) #BxCxC BxCxZ*X -> BxCxZ*X
        feat_memX0 = feat_memX0.view(B2,self.out_dim,Z,X)

        ###%%% reduce to 1 channel + apply position loss
        feat_pos_logits = self.pos_conv3d(feat_memX0)
        feat_pos_logits = feat_pos_logits.view(batch_size, num_objects, 1, feat_pos_logits.shape[-1], feat_pos_logits.shape[-2])
        
        if summ_writer is not None:
            summ_writer.summ_feat(f'{mode}/position_supervision', pos_targets, pca=False, only_return=False)

        forward_dict['feat_pos_logits'] = feat_pos_logits
        forward_dict['pos_targets'] = pos_targets
        forward_dict['feat_memX0'] = feat_memX0        

        return forward_dict
        
    def inference(self,feat_mem,feat_pos_logits,targets_batch,Z,Y,X,vox_util_batch,nav_pts_batch,origin_T_camXs_batch,classes_order,max_positions=20,summ_writer=None):
        '''
        feat_pos_logits: position logits from forward pass
        nav_pts_batch: list (length batch size) where each element is Nx3 navigable points in the environment (in origin coords)
        vox_util_batch: list of vox_util for each batch
        Z,Y,X: voxel dimension sizes
        '''

        

        with torch.no_grad():
            
            inference_dict = {}

            feat_mem_sigmoid = self.sigmoid(feat_pos_logits)

            batch_size = feat_mem_sigmoid.shape[0] # batch size
            num_objects = feat_mem_sigmoid.shape[1] # number of objects

            mem = []
            mem_b = []
            closest_nav_points = []
            classes = []
            batches = []
            class_n = []
            succeses = []
            feat_mem_thresh = []
            feat_mem_logits = []
            for b in range(batch_size):
                vox_util = vox_util_batch[b]
                nav_pts = nav_pts_batch[b]
                origin_T_camXs = origin_T_camXs_batch[b]
                for n in range(num_objects):
                    feat_mem_probs_b = feat_mem_sigmoid[b,n].squeeze()
                    pos_pred_mem_z, pos_pred_mem_x = torch.where(feat_mem_probs_b>self.pos_threshold)
                    # xyz_mem_pos = torch.stack([x_where, y_where, z_where]).t().unsqueeze(0).float()
                    if pos_pred_mem_z.nelement()==0:
                        succeses.append(False)
                        continue
                    else:
                        succeses.append(True)
                    
                    scores = feat_mem_probs_b[pos_pred_mem_z, pos_pred_mem_x]
                    arg_scores = torch.argsort(scores)
                    mem_ = torch.stack([pos_pred_mem_x, torch.ones(len(pos_pred_mem_z)).cuda().long()*(Y//2), pos_pred_mem_z]).t().float()
                    
                    xyz_mem_pos = torch.stack([pos_pred_mem_x, torch.ones(len(pos_pred_mem_z)).cuda().long()*(Y//2), pos_pred_mem_z]).t().unsqueeze(0).float()
                    xyz_camX0_pos = vox_util.Mem2Ref(xyz_mem_pos, Z,Y,X,assert_cube=False)
                    xyz_origin_pos = utils.geom.apply_4x4(origin_T_camXs[0:1], xyz_camX0_pos)
                    xyz_origin_pos = xyz_origin_pos.squeeze(0).detach().cpu().numpy()
                    closest_nav_points_, _ = utils.aithor.get_closest_navigable_point(torch.from_numpy(xyz_origin_pos).cuda(), torch.from_numpy(nav_pts).cuda())

                    # get unique positions (due to voxel spatial size can have duplicates)
                    unique, unique_mapping = torch.unique(closest_nav_points_, dim=0, return_inverse=True)
                    arg_unqiue_maxscores = torch.zeros(unique.shape[0]).long()
                    for u in range(unique.shape[0]):
                        where_unique_u = torch.where(unique_mapping==u)[0]
                        scores_u = scores[where_unique_u]
                        ind_m = torch.argmax(scores_u)
                        ind_u_maxscore = where_unique_u[ind_m]
                        arg_unqiue_maxscores[u] = ind_u_maxscore.long()

                    # sort by highest score
                    arg_unqiue_maxscores = arg_unqiue_maxscores[torch.flip(torch.argsort(scores[arg_unqiue_maxscores]),[0])]

                    # sort by score
                    if arg_unqiue_maxscores.shape[0] > max_positions:
                        arg_unqiue_maxscores = arg_unqiue_maxscores[:max_positions]
                    closest_nav_points_ = closest_nav_points_[arg_unqiue_maxscores]
                    mem_ = mem_[arg_unqiue_maxscores]

                    mem_b_ = torch.ones(mem_.shape[0]).cuda()*b*num_objects+n

                    classes.append(classes_order[b*num_objects+n])
                    batches.append(b)
                    class_n.append(n)
                    mem.append(mem_)
                    mem_b.append(mem_b_)
                    closest_nav_points.append(closest_nav_points_)
                    feat_mem_probs_b_vis = feat_mem_probs_b>self.pos_threshold
                    feat_mem_thresh.append(feat_mem_probs_b_vis)
                    feat_mem_logits.append(feat_mem_probs_b)
            if sum(succeses)==0:
                return None
            b_samps = torch.cat(mem_b, dim=0).long()
            feat_mem_thresh = torch.stack(feat_mem_thresh, dim=0).float()
            feat_mem_logits = torch.stack(feat_mem_logits, dim=0).float()

        inference_dict['closest_nav_points'] = closest_nav_points
        inference_dict['feat_mem_thresh'] = feat_mem_thresh
        inference_dict['feat_mem_logits'] = feat_mem_logits
        inference_dict['classes'] = classes
        inference_dict['batches'] = batches
        inference_dict['succeses'] = succeses
        inference_dict['class_n'] = class_n

        return inference_dict

    def inference_alt(
        self,
        feat_mem,
        feat_pos_logits,
        targets_batch,
        Z,Y,X,
        vox_util_batch,
        origin_T_camXs_batch,
        summ_writer=None
        ):
        '''
        feat_pos_logits: position logits from forward pass
        nav_pts_batch: list (length batch size) where each element is Nx3 navigable points in the environment (in origin coords)
        vox_util_batch: list of vox_util for each batch
        Z,Y,X: voxel dimension sizes
        '''

        inference_dict = {}

        feat_mem_sigmoid = self.sigmoid(feat_pos_logits)

        batch_size = feat_mem_sigmoid.shape[0] # batch size
        num_objects = feat_mem_sigmoid.shape[1] # number of objects

        mem = []
        mem_b = []
        closest_nav_points = []
        classes = []
        batches = []
        class_n = []
        succeses = []
        feat_mem_thresh = []
        feat_mem_logits = []
        xyz_origin_poss = []
        for b in range(batch_size):
            vox_util = vox_util_batch[b]
            origin_T_camXs = origin_T_camXs_batch[b]
            for n in range(num_objects):
                feat_mem_probs_b = feat_mem_sigmoid[b,n].squeeze()

                s1, s2 = feat_mem_probs_b.shape
                pos_mem_z, pos_mem_x = np.where(np.ones((s1,s2)))

                pos_mem_z = torch.from_numpy(pos_mem_z).cuda()
                pos_mem_x = torch.from_numpy(pos_mem_x).cuda()
                
                print('get xyz origin')
                xyz_mem_pos = torch.stack([pos_mem_x, torch.ones(len(pos_mem_z)).cuda().long()*(Y//2), pos_mem_z]).t().unsqueeze(0).float()
                xyz_camX0_pos = vox_util.Mem2Ref(xyz_mem_pos, Z,Y,X,assert_cube=False, device='cuda')
                xyz_origin_pos = utils.geom.apply_4x4(origin_T_camXs[0:1], xyz_camX0_pos)

                batches.append(b)
                class_n.append(n)
                xyz_origin_poss.append(xyz_origin_pos)
                feat_mem_probs_b_vis = feat_mem_probs_b>self.pos_threshold
                feat_mem_thresh.append(feat_mem_probs_b_vis)
                feat_mem_logits.append(feat_mem_probs_b)


        feat_mem_thresh = torch.stack(feat_mem_thresh, dim=0).float()
        feat_mem_logits = torch.stack(feat_mem_logits, dim=0).float()
        xyz_origin_poss = torch.cat(xyz_origin_poss, dim=0)

        feat_mem_sigmoid = feat_mem_logits[0].cpu().numpy()
        thresh_mem = feat_mem_sigmoid>args.vsn_threshold
        thresh_mem = scipy.ndimage.binary_erosion(thresh_mem, iterations=args.erosion_iters)
        if np.sum(thresh_mem)<args.num_search_locs_object: # number of valid points less than requested
            # if too little, do not do erosion
            thresh_mem = feat_mem_sigmoid>args.vsn_threshold
        if np.sum(thresh_mem)<args.num_search_locs_object:
            # if still too few, then just take highest scoring
            A, B = thresh_mem.shape
            thresh_mem = thresh_mem.flatten()
            argsort_thresh_mem = np.argsort(-thresh_mem)
            thresh_mem = np.zeros_like(thresh_mem)
            thresh_mem[argsort_thresh_mem[:args.num_search_locs_object]] = 1.
            thresh_mem = thresh_mem.reshape([A,B])
            thresh_mem = feat_mem_sigmoid>args.vsn_threshold

        xyz_origin_poss_i = xyz_origin_poss[0, thresh_mem.flatten(), :]
        where_thresh = np.where(thresh_mem)
        pts = np.stack([where_thresh[0], where_thresh[1]]).T
        farthest_pts, farthest_pts_inds = utils.samp.farthest_point_sampling(pts, args.num_search_locs_object)
        xyz_origin_select = xyz_origin_poss_i[farthest_pts_inds[:,0]]
        scores_fp = []
        for fp in range(len(farthest_pts)):
            fp_ = farthest_pts[fp].astype(int)
            scores_fp.append(feat_mem_sigmoid[fp_[0], fp_[1]])
        scores_fp = np.array(scores_fp)
        scores_argsort = np.flip(np.argsort(scores_fp))
        xyz_origin_select = xyz_origin_select[scores_argsort.copy(), :]

        inference_dict['feat_mem_thresh'] = feat_mem_thresh
        inference_dict['feat_mem_logits'] = feat_mem_logits
        inference_dict['class_n'] = class_n
        inference_dict['xyz_origin_poss'] = xyz_origin_poss
        inference_dict['xyz_origin_select'] = xyz_origin_select
        inference_dict['thresh_mem'] = thresh_mem
        inference_dict['farthest_pts'] = farthest_pts
        

        return inference_dict


    def inference_from_gt(
        self,
        feat_mem,
        feat_pos_logits,
        targets_batch,
        Z,Y,X,
        vox_util_batch,
        origin_T_camXs_batch,
        classes_order,
        max_positions=5,
        summ_writer=None
        ):

        with torch.no_grad():
            
            inference_dict = {}

            batch_size = len(targets_batch) #feat_mem_sigmoid.shape[0] # batch size
            num_objects = targets_batch[0]['B'] #feat_mem_sigmoid.shape[1] # number of objects

            mem = []
            mem_b = []
            closest_nav_points = []
            classes = []
            batches = []
            class_n = []
            succeses = []
            yaw_angles = []
            pitch_angles = []
            yaw_angles_inds_gt = []
            rgbs = []
            for b in range(batch_size):
                vox_util = vox_util_batch[b]
                nav_pts = nav_pts_batch[b]
                origin_T_camXs = origin_T_camXs_batch[b]
                occ_pos_camX0 = targets_batch[b]['occ_sup_camX0']
                vox_inds = targets_batch[b]['vox_inds']
                rots_camX0 = targets_batch[b]['rots']
                hors_camX0 = targets_batch[b]['horizons']
                yaw_camX0 = targets_batch[b]['yaw_camX0']
                pos_origins = targets_batch[b]['positions_origin']
                rgb = targets_batch[b]['rgb']
                for n in range(num_objects):
                    vox_inds_n = vox_inds[n]
                    rgb_n = rgb[n]
                    positions_origin_n = pos_origins[n]

                    feat_mem_probs_b = occ_pos_camX0[n].squeeze()
                    
                    # feat_mem_probs_b = torch.max(feat_mem_probs_b, dim=1)[0]
                    pos_pred_mem_z, pos_pred_mem_x = torch.where(feat_mem_probs_b>self.pos_threshold)
                    if pos_pred_mem_z.nelement()==0:
                        succeses.append(False)
                        continue
                    else:
                        succeses.append(True)
                    
                    scores = feat_mem_probs_b[pos_pred_mem_z, pos_pred_mem_x]
                    arg_scores = torch.argsort(scores)
                    mem_ = torch.stack([pos_pred_mem_x, torch.ones(len(pos_pred_mem_z)).cuda().long()*(Y//2), pos_pred_mem_z]).t().float()
                    
                    xyz_mem_pos = torch.stack([pos_pred_mem_x, torch.ones(len(pos_pred_mem_z)).cuda().long()*(Y//2), pos_pred_mem_z]).t().unsqueeze(0).float()
                    xyz_camX0_pos = vox_util.Mem2Ref(xyz_mem_pos, Z,Y,X,assert_cube=False)
                    xyz_origin_pos = utils.geom.apply_4x4(origin_T_camXs[0:1], xyz_camX0_pos)
                    xyz_origin_pos = xyz_origin_pos.squeeze(0).detach().cpu().numpy()
                    closest_nav_points_, _ = utils.aithor.get_closest_navigable_point(torch.from_numpy(xyz_origin_pos).cuda(), torch.from_numpy(nav_pts).cuda())

                    # get unique positions (due to voxel spatial size can have duplicates)
                    unique, unique_mapping = torch.unique(closest_nav_points_, dim=0, return_inverse=True)
                    arg_unqiue_maxscores = torch.zeros(unique.shape[0]).long()
                    for u in range(unique.shape[0]):
                        where_unique_u = torch.where(unique_mapping==u)[0]
                        scores_u = scores[where_unique_u]
                        ind_m = torch.argmax(scores_u)
                        ind_u_maxscore = where_unique_u[ind_m]
                        arg_unqiue_maxscores[u] = ind_u_maxscore.long()

                    # sort by score
                    if arg_unqiue_maxscores.shape[0] > max_positions:
                        arg_unqiue_maxscores = arg_unqiue_maxscores[:max_positions]
                    closest_nav_points_ = closest_nav_points_[arg_unqiue_maxscores]
                    mem_ = mem_[arg_unqiue_maxscores]
                    rots_n_origin = rots_n_origin[arg_unqiue_maxscores]
                    hors_n_origin = hors_n_origin[arg_unqiue_maxscores]
                    rgb_n = rgb_n[arg_unqiue_maxscores]
                    rots_n_origin2 = rots_n_origin2[arg_unqiue_maxscores]
                    vox_inds_n = vox_inds_n[arg_unqiue_maxscores]

                    mem_b_ = torch.ones(mem_.shape[0]).cuda()*b*num_objects+n

                    yaw_angles_inds_gt.append(rots_n_origin2)
                    classes.append(classes_order[b*num_objects+n])
                    batches.append(b)
                    class_n.append(n)
                    mem.append(mem_)
                    mem_b.append(mem_b_)
                    closest_nav_points.append(closest_nav_points_)
                    yaw_angles.append(rots_n_origin)
                    pitch_angles.append(hors_n_origin)
                    rgbs.append(rgb_n)
            if sum(succeses)==0:
                return None
            b_samps = torch.cat(mem_b, dim=0).long()

        inference_dict['closest_nav_points'] = closest_nav_points
        inference_dict['classes'] = classes
        inference_dict['batches'] = batches
        inference_dict['succeses'] = succeses
        inference_dict['class_n'] = class_n
        inference_dict['rgbs'] = rgbs

        return inference_dict


    def prepare_supervision(
        self, 
        obj_info_all, 
        origin_T_camXs,
        xyz,scene_centroid,
        # yaw_camX0,
        Z,Y,X,
        add_noise=False,
        std=0.5,
        inference=False,
        ):
        '''
        Prepare supervised for VSN
        obj_info_all has object position info in aithor coordinate frame
        '''

        camX0_T_origin = utils.geom.safe_inverse_single(origin_T_camXs[0])
        camX0s_T_camXs = utils.geom.get_camM_T_camXs(origin_T_camXs.unsqueeze(0), ind=0).squeeze(0)
        xyz = utils.geom.apply_4x4(camX0s_T_camXs, xyz)
        scene_centroid_origin = torch.tensor(list(scene_centroid.values())).view(1,1,3).cuda()
        scene_centroid_camX0 = utils.geom.apply_4x4(camX0_T_origin.unsqueeze(0), scene_centroid_origin).squeeze(0)
        if add_noise:
            scene_centroid_camX0 += torch.normal(torch.tensor([0.,0.,0.]), torch.tensor([std,std,std])).cuda().unsqueeze(0)

        vox_util = utils.vox.Vox_util(Z, Y, X, '', scene_centroid_camX0, bounds=self.bounds, assert_cube=False)
        B = len(obj_info_all)
        keys = list(obj_info_all.keys())

        targets = {}
        targets['B'] = B
        targets['occ_sup_camX0'] = torch.zeros(B,1,Z,X).cuda()
        targets['obj_ids'] = torch.zeros(B).cuda()
        targets['xyz_pos_camX0'] = []
        targets['rots'] = []
        targets['horizons'] = []
        targets['rots_gt'] = []
        targets['horizons_gt'] = []
        targets['vox_inds'] = []
        targets['b_inds'] = []
        targets['rgb'] = []
        targets['positions_origin'] = []
        if inference:
            for b in range(B):
                key = keys[b]
                obj_info = obj_info_all[key]
                targets['obj_ids'][b] = torch.tensor(obj_info['obj_id']).cuda()
            return targets, vox_util
        for b in range(B):
            key = keys[b]
            obj_info = obj_info_all[key]

            positions = obj_info['obj_position'].unsqueeze(0).float().cuda()
            xyz_pos_camX0 = utils.geom.apply_4x4(camX0_T_origin, positions)
            occ_pos_camX0, vox_inds_expand, vox_inds = vox_util.voxelize_xyz(xyz_pos_camX0, Z, Y, X, return_inds=True, assert_cube=False)
            occ_pos_camX0 = torch.max(occ_pos_camX0, dim=3).values
            occ_pos_camX0 = gaussian_filter(occ_pos_camX0.cpu().numpy(), sigma=3)
            occ_pos_camX0 = torch.from_numpy(occ_pos_camX0).cuda()
            
            if False:
                occ_poss = torch.max(occ_pos_camX0.squeeze(), dim=1).values
                # occ_X0 = torch.max(occ_X0, dim=1).values
                occ_poss = occ_poss.detach().cpu().numpy()
                plt.figure()
                plt.imshow(occ_poss)
                plt.savefig('images/test2.png')

            inds_keep = np.arange(positions.shape[1])
            inds_keep = inds_keep[vox_inds>0]
            vox_inds_ = vox_inds[inds_keep]
            inds_keep = inds_keep[np.unique(vox_inds_, return_index=True)[1]]
            positions = positions[:,inds_keep]

            targets['occ_sup_camX0'][b] = occ_pos_camX0
            targets['obj_ids'][b] = torch.tensor(obj_info['obj_id']).cuda()
            targets['xyz_pos_camX0'].append(xyz_pos_camX0)
            targets['vox_inds'].append(torch.from_numpy(vox_inds).cuda().long())
            targets['positions_origin'].append(positions)

        return targets, vox_util

    def get_pix_to_camX(self,H,W):
        self.fov = 90
        hfov = float(self.fov) * np.pi / 180.
        pix_T_camX = np.array([
            [(W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        pix_T_camX[0,2] = W/2.
        pix_T_camX[1,2] = H/2.
        pix_T_camX = torch.from_numpy(pix_T_camX).cuda().float()
        return pix_T_camX

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=1)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).cuda() * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



        
