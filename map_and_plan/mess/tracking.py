import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import skimage
import skimage.io
from PIL import Image

from .seg2dnet import Seg2dNet
from .compressnet import CompressNet

import utils.obj
import utils.py
import utils.samp
import utils.geom
import utils.misc
import utils.improc
import utils.basic
import utils.track

from .config import objects_traj_lib_path, seg2dnet_checkpoint_path, compressnet_checkpoint_path

# SIZE_train = 64
# SIZE_val = 64
# SIZE_test = 64
# groups['mcs_4-2-4_bounds_test'] = [
#     'XMIN_test = -4.0', # right (neg is left)
#     'XMAX_test = 4.0', # right
#     'YMIN_test = -2.0', # down (neg is up)
#     'YMAX_test = 2.0', # down
#     'ZMIN_test = -4.0', # forward
#     'ZMAX_test = 4.0', # forward
#     'Z_test = %d' % (int(SIZE_test*4)),
#     'Y_test = %d' % (int(SIZE_test*2)),
#     'X_test = %d' % (int(SIZE_test*4)),
# ]

class Tracking():
    def __init__(self, output):
        self.seg2dnet = Seg2dNet(num_classes = 1).cuda()
        seg2dnet_checkpoint = torch.load(seg2dnet_checkpoint_path)
        seg2dnet_state_dict = self.seg2dnet.state_dict()
        print("loading seg2dnet...")
        for load_para_name, para in seg2dnet_checkpoint['model_state_dict'].items():
            model_para_name = '.'.join(load_para_name.split('.')[1:])
            if model_para_name in seg2dnet_state_dict.keys():
                if not (seg2dnet_state_dict[model_para_name].shape == para.data.shape):
                    print(model_para_name, load_para_name)
                    print('param in ckpt', para.data.shape)
                    print('param in state dict', seg2dnet_state_dict[model_para_name].shape)
                assert(seg2dnet_state_dict[model_para_name].shape == para.data.shape), model_para_name
                seg2dnet_state_dict[model_para_name].copy_(para.data)
            else:
                print('warning: %s is not in the state dict of the current model' % model_para_name)
        self.seg2dnet.eval()
        self.set_requires_grad(self.seg2dnet, False)

        self.compressnet = CompressNet().cuda()
        compressnet_checkpoint = torch.load(compressnet_checkpoint_path)
        compressnet_state_dict = self.compressnet.state_dict()
        print("loading compressnet...")
        for load_para_name, para in compressnet_checkpoint['model_state_dict'].items():
            if "slow" in load_para_name:
                continue
            model_para_name = '.'.join(load_para_name.split('.')[1:])
            if model_para_name in compressnet_state_dict.keys():
                if not (compressnet_state_dict[model_para_name].shape == para.data.shape):
                    print(model_para_name, load_para_name)
                    print('param in ckpt', para.data.shape)
                    print('param in state dict', compressnet_state_dict[model_para_name].shape)
                assert(compressnet_state_dict[model_para_name].shape == para.data.shape), model_para_name
                compressnet_state_dict[model_para_name].copy_(para.data)
            else:
                print('warning: %s is not in the state dict of the current model' % model_para_name)
        self.compressnet.eval()
        self.set_requires_grad(self.compressnet, False)


        self.score_pool = utils.misc.SimplePool(1000, version='np')

        torch.autograd.set_detect_anomaly(True)
        self.include_image_summs = True

        self.Z1 = 64*4
        self.Y1 = 64*2
        self.X1 = 64*4
        self.S = 100 # max length of the video
        self.s0 = -1

        self.H = 400 # resolution of the input data
        self.W = 600
        
        self.H2 = int(self.H/2)
        self.W2 = int(self.W/2)
        self.Z2 = int(self.Z1/2)
        self.Y2 = int(self.Y1/2)
        self.X2 = int(self.X1/2)

        scene_centroid_x = 0.0
        scene_centroid_y = -0.5 # up a bit, to eliminate some voxels under ground
        scene_centroid_z = 6.0
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()

        self.vox_util = utils.vox.Vox_util(
            self.Z1, self.Y1, self.X1, 
            scene_centroid=self.scene_centroid,
            assert_cube=True)

        self.weights2d = torch.ones(1, 1, 3, 3, device=torch.device('cuda'))
        self.weights3d = torch.ones(1, 1, 3, 3, 3, device=torch.device('cuda'))

        self.N_max = 30
        self.lrtlists_camX = torch.zeros((self.S, self.N_max, 19), dtype=torch.float32).cuda()
        self.scorelists = torch.zeros((self.S, self.N_max), dtype=torch.float32).cuda()
        self.masklists = torch.zeros((self.S, self.N_max, self.H, self.W), dtype=torch.float32).cuda()
        self.featlists = torch.zeros((self.S, self.N_max, 128), dtype=torch.float32).cuda()

        self.depth_camXs = torch.zeros((1, 0, self.H, self.W), dtype=torch.float32).cuda()
        self.xyz_camXs = torch.zeros((1, 0, self.H*self.W, 3), dtype=torch.float32).cuda()
        self.sem_seq = torch.zeros((0, self.H, self.W, 3), dtype=torch.float32).cuda()

        self.rgb_vis = []
        self.mask_vis = []
        self.seg_vis = []
        self.bg_vis = []
        self.boxes_vis_p = []
        self.boxes_vis_b = []
        self.center_vis = []

        self.vis_halfmemX_list = []
        self.center_halfmemX_list = []

        self.obj_list = []
        
        self.val_halfmemXs = []
        self.val_halfmemXs_vis = []

        self.dot_match_thresh = 0.7

        traj_lib = np.load(objects_traj_lib_path)
        self.traj_lib = torch.from_numpy(traj_lib).float().cuda()
        self.traj_lib = self.traj_lib[:,:self.S]

        self.voe_value_list = np.zeros((self.S), dtype=np.float32)
        self.voe_xy_list = np.zeros((self.S, 2), dtype=np.float32)
        self.voe_heatmap_list = np.zeros((self.S, self.H, self.W), dtype=np.float32)

    def get_box_from_mask(self, mask, xyz, use_std=True):
        H, W = mask.shape
        HW, D = xyz.shape
        assert(D == 3)
        assert(HW==H*W)
        assert(self.H*self.W==HW)

        mask_ = mask.reshape(-1)
        xyz = xyz[mask_ > 0]

        grid_y, grid_x = utils.py.meshgrid2d(self.H, self.W)
        grid_y_ = grid_y.reshape(-1)
        grid_x_ = grid_x.reshape(-1)
        x_pix = grid_x_[mask_ > 0].astype(np.int32)
        y_pix = grid_y_[mask_ > 0].astype(np.int32)
        x_pix = np.unique(x_pix)
        y_pix = np.unique(y_pix)

        y_pix_min = np.min(y_pix)
        x_pix_min = np.min(x_pix)
        y_pix_max = np.max(y_pix)
        x_pix_max = np.max(x_pix)

        h_pix = y_pix_max - y_pix_min
        w_pix = x_pix_max - x_pix_min

        box2d = np.stack([y_pix_min/self.H,
                          x_pix_min/self.W,
                          y_pix_max/self.H,
                          x_pix_max/self.W], axis=0)

        if (np.sum(mask) > 64 and  
            (h_pix > 6) and # slightly larger mask req in hard mode, for safety against occluders
            (w_pix > 6) and
            (w_pix < self.W/2) and # also max size, i.e., don't track the wall
            (h_pix < self.H/2) 
        ):
            valid = 1.0

            x_min = np.min(xyz[:,0])
            x_max = np.max(xyz[:,0])
            y_min = np.min(xyz[:,1])
            y_max = np.max(xyz[:,1])
            z_min = np.min(xyz[:,2])
            z_max = np.max(xyz[:,2])

            cx = (x_max + x_min)/2.0
            cy = (y_max + y_min)/2.0
            cz = (z_max + z_min)/2.0

            cx_ = np.median(xyz[:,0])
            cy_ = np.median(xyz[:,1])
            cz_ = np.median(xyz[:,2])
            c_med = np.array([cx_, cy_, cz_]).reshape(1, 3)
            xyz_med = xyz - c_med
            xyz_norm = np.linalg.norm(xyz_med, axis=1)
            xyz_std = np.sqrt(np.var(xyz_norm))

            lx = x_max - x_min
            ly = y_max - y_min
            lz = z_max - z_min

            rx, ry, rz = 0.0, 0.0, 0.0

            xyz = np.array([cx,cy,cz]).reshape(3)
            size = np.array([lx, ly, lz]).reshape(3)
            rot = size*0
            box3d = np.concatenate([xyz, size, rot], axis=0)
            score = 1.0

            print('xyz_std', xyz_std)
            if xyz_std > 0.3:
                score = 0.0
                
            print('got an object')
        else:
            box3d = np.zeros((9), dtype=np.float32)
            score = 0.0
            box2d = np.zeros((4), dtype=np.float32)

        return box3d, score, box2d

    def get_objects_from_seg_easy(self, sem_py, safe_bg, seg_e, xyz_cam_py):
        sem_py = sem_py * (1 - safe_bg.reshape(self.H, self.W, 1).cpu().numpy())
        sem_py_ = sem_py.reshape(-1, 3)
        colors_here = np.unique(sem_py_, axis=0)
        N_max = len(colors_here)

        boxlist = np.zeros((N_max, 9), dtype=np.float32)
        masklist = np.zeros((N_max, self.H, self.W), dtype=np.float32)
        scorelist = np.zeros((N_max), dtype=np.float32)

        for ci, color_here in enumerate(colors_here):

            mask = (np.sum(sem_py == np.reshape(color_here, (1, 1, 3)), axis=2)==3).astype(np.uint8)
            seg_ = seg_e.reshape(self.H, self.W).cpu().numpy()

            if np.sum(mask*seg_) > np.sum(mask)/4.0: # at least a fraction is marked by our net
                # use the amodal mask to gather points
                box, score, _ = self.get_box_from_mask(mask, xyz_cam_py)
                boxlist[ci] = box
                scorelist[ci] = score
                masklist[ci] = mask*score
            # end conds over mask
        # end loop over colors_here

        boxlist = torch.from_numpy(boxlist).float().cuda().unsqueeze(0) # 1 x N x 9
        scorelist = torch.from_numpy(scorelist).float().cuda().unsqueeze(0) # 1 x N
        masklist = torch.from_numpy(masklist).float().cuda().unsqueeze(0) # 1 x N x H x W

        # overall mask (for vis mostly)
        mask = torch.max(masklist, dim=1)[0].unsqueeze(1)

        boxlist = boxlist[:,scorelist[0] > 0] # 1 x ? x 9
        masklist = masklist[:,scorelist[0] > 0]
        scorelist = scorelist[:,scorelist[0] > 0]
        N = scorelist.shape[1]
        
        return boxlist, masklist, scorelist, N, mask

    def get_objects_from_seg_hard(self, safe_bg, seg_e, xyz_cam_py):
        
        seg_e = seg_e.reshape(self.H, self.W)

        # connected components
        component_label = skimage.measure.label(seg_e.cpu().numpy())
        N_max = np.max(component_label)+1
        
        boxlist = np.zeros((N_max, 9), dtype=np.float32)
        masklist = np.zeros((N_max, self.H, self.W), dtype=np.float32)
        scorelist = np.zeros((N_max), dtype=np.float32)

        boxlist2d = []

        for ci in range(N_max):
            mask = component_label==ci

            # avoid the bkg component
            mask_ = mask.reshape(-1)
            grid_y, grid_x = utils.py.meshgrid2d(self.H, self.W)
            grid_y_ = grid_y.reshape(-1)
            grid_x_ = grid_x.reshape(-1)
            x_pix = grid_x_[mask_ > 0].astype(np.int32)
            y_pix = grid_y_[mask_ > 0].astype(np.int32)
            x_pix = np.unique(x_pix)
            y_pix = np.unique(y_pix)
            y_pix_min = np.min(y_pix)
            x_pix_min = np.min(x_pix)
            y_pix_max = np.max(y_pix)
            x_pix_max = np.max(x_pix)
            h_pix = y_pix_max - y_pix_min
            w_pix = x_pix_max - x_pix_min
            if h_pix < self.H/2 and w_pix < self.W/2:
                mask = torch.from_numpy(mask).float().reshape(1, 1, self.H, self.W).cuda()

                mask = mask * (1 - safe_bg)

                # erode slightly, for safety
                weights2d = torch.ones(1, 1, 3, 3, device=torch.device('cuda'))
                mask = 1.0 - (F.conv2d(1.0 - mask, weights2d, padding=1)).clamp(0, 1)
                mask = mask.reshape(self.H, self.W).cpu().numpy()

                # use the amodal mask to gather points
                if np.sum(mask) > 16:
                    box3d, score, box2d = self.get_box_from_mask(mask, xyz_cam_py, use_std=True)

                    if len(boxlist2d):
                        ious = utils.box.boxlist_2d_iou(np.reshape(box2d, (1, 4)), np.stack(boxlist2d, axis=0))
                        # print('ious', ious)
                        if np.max(ious) > 0.2:
                            score = 0
                    if score > 0:
                        boxlist2d.append(box2d)

                    boxlist[ci] = box3d
                    scorelist[ci] = score
                    masklist[ci] = mask*score

        boxlist = torch.from_numpy(boxlist).float().cuda().unsqueeze(0) # 1 x N x 9
        scorelist = torch.from_numpy(scorelist).float().cuda().unsqueeze(0) # 1 x N
        masklist = torch.from_numpy(masklist).float().cuda().unsqueeze(0) # 1 x N x H x W

        # overall mask (for vis mostly)
        mask = torch.max(masklist, dim=1)[0].unsqueeze(1)

        boxlist = boxlist[:,scorelist[0] > 0] # 1 x ? x 9
        masklist = masklist[:,scorelist[0] > 0]
        scorelist = scorelist[:,scorelist[0] > 0]
        N = scorelist.shape[1]
        
        return boxlist, masklist, scorelist, N, mask

    def get_seg_e(self, rgb):
        # first make the rgb a nicer resolution
        _, _, H, W = list(rgb.shape)
        H_, W_ = 512, 768
        rgb = F.interpolate(rgb, (H_, W_))
        back_rgb = utils.improc.back2color(rgb)
        #skimage.io.imsave("rgb_{}.png".format(self.s0), rgb.permute(0, 2, 3, 1).cpu().numpy().reshape(H_, W_, 3))
        seg2d_loss, seg_e = self.seg2dnet(rgb)
        seg_e = F.interpolate(F.sigmoid(seg_e), (self.H, self.W), mode='bilinear')
        #skimage.io.imsave("seg_{}.png".format(self.s0), seg_e.cpu().numpy().reshape(self.H, self.W))
        return seg_e

    def act(self, output):
        self.s0 += 1

        rgb_camX = torch.from_numpy(np.array(output.image_list[-1])).permute(2,0,1).unsqueeze(0).cuda()
        rgb_camX = utils.improc.preprocess_color(rgb_camX)

        def parse_intrinsics(camera):
            W, H = camera['camera_aspect_ratio']
            fov = camera['camera_fov']
            focal_pix = (W * 0.5) / np.tan(fov * 0.5 * np.pi/180)
            x0 = W/2.0
            y0 = H/2.0
            pix_T_cam = utils.py.merge_intrinsics(
                focal_pix, focal_pix, x0, y0)
            return pix_T_cam

        camera_info = {}
        camera_info['camera_aspect_ratio'] = output.camera_aspect_ratio
        camera_info['camera_clipping_planes'] = output.camera_clipping_planes
        camera_info['camera_fov'] = output.camera_field_of_view
        camera_info['camera_height'] = output.camera_height
        pix_T_cam = torch.from_numpy(parse_intrinsics(camera_info).reshape(1,4,4)).cuda()

        # Set level 1 (hard) or 2 (easy)
        hard_mode = False
        if len(output.object_mask_list) == 0:
            hard_mode = True

        assoc_mode = None
        if hard_mode:
            assoc_mode = 'most_confident'
        else:
            assoc_mode = 'largest'

        sem = None
        if hard_mode:
            seg_e = self.get_seg_e(rgb_camX)
            seg_e = (seg_e > 0.9).float()
        else:
            sem = torch.from_numpy(np.array(output.object_mask_list[-1])).unsqueeze(0).cuda()
            self.sem_seq = torch.cat([self.sem_seq, sem], dim=0)
            # use the seg to clarify edges in the rgb map
            sem = self.sem_seq[self.s0] # H x W x 3
            sem_im = sem.permute(2, 0, 1).unsqueeze(0).cuda() # 1 x 3 x H x W
            sem_im = utils.improc.preprocess_color(sem_im)
            seg_e = self.get_seg_e((rgb_camX+sem_im)/2.0)
            seg_e = (seg_e > 0.9).float()

        # depth and xyz
        depth = torch.from_numpy(np.array(output.depth_map_list[-1])).unsqueeze(0).unsqueeze(0).cuda()
        xyz_camX = utils.geom.depth2pointcloud(depth, pix_T_cam).unsqueeze(1).float().cuda() # depth: 1,1,H,W; pix_T_cam: 1,3,3 / 1,4,4; xyz_camX: 1, N, 3

        # depth and xyz list
        self.depth_camXs = torch.cat([self.depth_camXs, depth], dim=1)
        self.xyz_camXs = torch.cat([self.xyz_camXs, xyz_camX], dim=1)

        # form bkg estimates
        depth_bg0 = torch.max(self.depth_camXs, dim=1)[0]
        depth_bg1 = torch.max(depth, dim=3)[0].reshape(1, 1, self.H, 1).repeat(1,1,1, self.W)
        bg0 = (torch.abs(depth - depth_bg0) < 0.05).float()
        bg1 = (torch.abs(depth - depth_bg1) < 0.05).float()
        safe_bg = (bg0 + bg1).clamp(0,1)

        xyz_cam_py = xyz_camX.squeeze(0).squeeze(0).cpu().numpy()

        if hard_mode:
            boxlist, masklist, scorelist, N, mask = self.get_objects_from_seg_hard(safe_bg, seg_e, xyz_cam_py)
        else:
            boxlist, masklist, scorelist, N, mask = self.get_objects_from_seg_easy(sem.cpu().numpy(), safe_bg, seg_e, xyz_cam_py)

        occ_memX = self.vox_util.voxelize_xyz(xyz_camX.squeeze(0), self.Z1, self.Y1, self.X1)

        lrtlist_camX = utils.geom.convert_boxlist_to_lrtlist(boxlist) # 1 x N x 9
        tidlist = torch.arange(N).reshape(1, N)

        # store for later:
        self.lrtlists_camX[self.s0,:N] = lrtlist_camX.squeeze(0)
        self.scorelists[self.s0,:N] = scorelist.squeeze(0)
        self.masklists[self.s0,:N] = masklist.squeeze(0)

        # visibility
        vis_halfmemX = self.vox_util.convert_xyz_to_visibility(xyz_camX.squeeze(0), self.Z2, self.Y2, self.X2).detach().cpu()
        self.vis_halfmemX_list.append(vis_halfmemX.detach().cpu())

        if N > 0: # if objects were detected here
            # objectness
            clist_camX = utils.geom.get_clist_from_lrtlist(lrtlist_camX)
            lenlist, _ = utils.geom.split_lrtlist(lrtlist_camX)
            # these are B x N x 3
            sizelist = (torch.max(lenlist, dim=2)[0]).clamp(min=0.5)
            mask = self.vox_util.xyz2circles(clist_camX, sizelist*4.0, self.Z2, self.Y2, self.X2, soft=True, already_mem=False)
            mask = mask * (scorelist > 0).float().reshape(1, N, 1, 1, 1)
            center_halfmemX = torch.max(mask, dim=1, keepdim=True)[0]
            center_halfmemX = (center_halfmemX > 1e-4).float()

            boxlist2d = utils.geom.get_boxlist2d_from_lrtlist(pix_T_cam, lrtlist_camX, self.H, self.W, pad=10, clamp=False)
            mask_ = masklist.reshape(N, 1, self.H, self.W)
            rgb_ = rgb_camX.repeat(N, 1, 1, 1)
            box2d_ = boxlist2d[0] # N x 4

            CH, CW = 128, 128
            crops_ = utils.geom.crop_and_resize(rgb_.float(), box2d_.float(), CH, CW)
            feats_ = self.compressnet(crops_)
            featlist = feats_.unsqueeze(0)
        else:
            center_halfmemX = torch.zeros_like(vis_halfmemX)
            featlist = torch.zeros((1, N, 128), dtype=torch.float32).cuda()

        self.center_halfmemX_list.append(center_halfmemX.detach().cpu())

        # store for later
        self.featlists[self.s0, :N] = featlist.squeeze(0)

        lrtlist_camX = lrtlist_camX.squeeze(0)
        scorelist = scorelist.squeeze(0)
        featlist = featlist.squeeze(0)

        # for each obj in obj_list, see how well it matches the dets
        print('we have %d dets to match against' % (N))

        if len(self.obj_list) == 1:
            tid = 0
            obj = self.obj_list[0]
            max_dot = 0.0
            max_ind = 0
            for n0 in range(N):
                lrt0 = lrtlist_camX[n0].cpu()
                score0 = scorelist[n0].cpu()
                feat0 = featlist[n0].cpu()
                if score0 > 0: # this det is still unclaimed
                    if obj.scorelist[self.s0] == 0.0: # this obj is still unclaimed at this frame
                        f_dot = obj.dot_me(feat0, mode=assoc_mode, lrt0=lrt0)
                    else:
                        f_dot = 0.0
                    if f_dot > max_dot:
                        max_dot = f_dot
                        max_ind = n0
                        print('(obj %d matches det %d with new max_dot of %.3f)' % (tid, n0, max_dot))
            if max_dot > self.dot_match_thresh:
                print('matched obj %d to det %d on frame %d (max_dot %.3f)' % (tid, max_ind, self.s0, max_dot))
                obj.update_with_match(self.s0, lrtlist_camX[max_ind].cpu(), max_dot, featlist[max_ind].cpu())
                scorelist[max_ind] = 0.0 # set this score to 0 so we don't assign it to another obj
        elif len(self.obj_list) > 1:
            featlist_obj = torch.stack([obj.get_feat(mode=assoc_mode) for obj in self.obj_list], dim=0)

            dot_matrix = torch.matmul(featlist_obj.cpu(), featlist.permute(1,0).cpu())
            print('dot_matrix', dot_matrix.shape)

            row_ind, col_ind = scipy.optimize.linear_sum_assignment(1-dot_matrix)

            for (row,col) in zip(row_ind, col_ind):
                dot = dot_matrix[row,col]
                obj = self.obj_list[row]
                lrt0 =  lrtlist_camX[col]
                feat0 =  featlist[col]
                if dot > self.dot_match_thresh:
                    obj.update_with_match(self.s0, lrt0.cpu(), dot, feat0.cpu())
                    # scorelist[max_ind] = 0.0 # set this score to 0 so we don't assign it to another obj
                    scorelist[col] = 0
                    print('matched obj %d to det %d on frame %d (max_dot %.3f)' % (row, col, self.s0, dot))

        # now let's walk through the unassigned dets
        for n0 in range(N):
            lrt0 = lrtlist_camX[n0].cpu()
            score0 = scorelist[n0].cpu()
            feat0 = featlist[n0].cpu()
            if score0 > 0: # check score in case it has been claimed
                print('initializing a new object on frame %d' % (self.s0))
                tid = len(self.obj_list)
                new_obj = utils.obj.Obj_util(self.s0, lrt0, score0, feat0, self.S, tid, self.Z1, self.Y1, self.X1, H=100)
                self.obj_list.append(new_obj)

        # now update plausibilities
        for tid, obj in enumerate(self.obj_list):
            print('updating forecasts and vis for obj %d' % tid)

            # on every frame, create a custom val_halfmemX,
            # where each center mask is multiplied with the appropriate value of f_dot
            # it is necessary here to loop over s again, maybe only up to s0

            def get_dets_for_frame(lrtlists, scorelists, featlists, si):
                lrtlist0 = lrtlists[si].reshape(-1, 19)
                scorelist0 = scorelists[si].reshape(-1)
                featlist0 = featlists[si].reshape(-1, 128)
                ind0 = scorelist0 > 0.9
                lrtlist0 = lrtlist0[ind0]
                scorelist0 = scorelist0[ind0]
                featlist0 = featlist0[ind0]
                N0 = lrtlist0.shape[0]
                return lrtlist0, scorelist0, featlist0, N0

            val_halfmemX_list = []

            for si in range(self.s0+1): # up to and including current frame

                lrtlist0, scorelist0, featlist0, N0 = get_dets_for_frame(self.lrtlists_camX, self.scorelists, self.featlists, si)

                aboveground = torch.max(vis_halfmemX, dim=4)[0].reshape(1, 1, self.Z2, self.Y2, 1).repeat(1, 1, 1, 1, self.X2)

                if N0 > 0:
                    dotlist0 = torch.zeros_like(scorelist0)
                    for n0 in range(N0):
                        lrt0 = lrtlist0[n0].cpu()
                        score0 = scorelist0[n0].cpu()
                        feat0 = featlist0[n0].cpu()
                        f_dot = obj.dot_me(feat0, mode=assoc_mode, lrt0=lrt0)
                        dotlist0[n0] = f_dot
                    clist = utils.geom.get_clist_from_lrtlist(lrtlist0.unsqueeze(0)) # 1 x N0 x 3
                    lenlist, _ = utils.geom.split_lrtlist(lrtlist0.unsqueeze(0)) # 1 x N0 x 3
                    sizelist = (torch.max(lenlist, dim=2)[0]).clamp(min=0.1)
                    mask = self.vox_util.xyz2circles(clist, sizelist*4.0, self.Z2, self.Y2, self.X2, soft=True, already_mem=False).detach().cpu()
                    mask = (mask > 1e-4).float()
                    any_center_halfmemX = torch.max(mask, dim=1, keepdim=True)[0]

                    mask = mask * ((scorelist0.cpu() > 0).float() * dotlist0.cpu()).reshape(1, N0, 1, 1, 1)
                    obj_center_halfmemX = torch.max(mask, dim=1, keepdim=True)[0]

                    # dilate occl, for safety
                    occl_halfmemX = (1.0 - vis_halfmemX).cuda()
                    occl_halfmemX = F.conv3d(occl_halfmemX, self.weights3d, padding=1).clamp(0, 1)
                    occl_halfmemX = F.conv3d(occl_halfmemX, self.weights3d, padding=1).clamp(0, 1)
                    occl_halfmemX = F.conv3d(occl_halfmemX, self.weights3d, padding=1).clamp(0, 1)
                    occl_halfmemX = occl_halfmemX.cpu()

                    val_halfmemX = (obj_center_halfmemX + (1.0 - any_center_halfmemX) * (1.0 - vis_halfmemX)).clamp(0,1)
                    # 1 x 1 x Z2 x Y2 x X2
                else:
                    val_halfmemX = (1.0 - vis_halfmemX).clamp(0,1)
                    val_halfmemX = F.conv3d(val_halfmemX.cuda(), self.weights3d, padding=1).clamp(0, 1).cpu()
                    val_halfmemX = F.conv3d(val_halfmemX.cuda(), self.weights3d, padding=1).clamp(0, 1).cpu()
                    val_halfmemX = F.conv3d(val_halfmemX.cuda(), self.weights3d, padding=1).clamp(0, 1).cpu()
                val_halfmemX = val_halfmemX*aboveground
                val_halfmemX_list.append(val_halfmemX)
            # end loop over si
                
            obj.update_forecasts(self.s0, torch.stack(val_halfmemX_list, dim=1), self.traj_lib, self.vox_util)

        if len(self.obj_list) and self.s0 > 0:
            all_plaus = torch.stack([obj.plauslist[:self.s0+1] for obj in self.obj_list], dim=0) # ? x s0

            # make sure it is strictly decreasing
            for si in range(1,self.s0+1):
                for o in range(len(self.obj_list)):
                    all_plaus[o,si] = torch.min(all_plaus[o,si-1], all_plaus[o,si])

            plaus_here = all_plaus[:,-1] # ?
            plaus_prev = all_plaus[:,-2] # ?

            plaus_delta = plaus_prev - plaus_here # ?
            plaus_delta_max = torch.max(plaus_delta) # 1

            if plaus_delta_max > 0.0:
                weird_obj = torch.argmax(plaus_delta)
                xyz = self.obj_list[weird_obj].forecastlist[self.s0] # H x 3
                H = xyz.shape[0]

                xy = utils.geom.apply_pix_T_cam(pix_T_cam, xyz.unsqueeze(0).cuda()).squeeze(0) # H x 2
                heatmap = utils.improc.xy2mask_single(xy, self.H, self.W).cpu() 
                heatmap /= (heatmap.sum() + 1e-8)
                heatmap *= plaus_delta_max
                print(heatmap.sum(), plaus_delta_max)
                self.voe_heatmap_list[self.s0] = heatmap.reshape(self.H, self.W).numpy()
                self.voe_xy_list[self.s0] = torch.mean(xy, dim=0).cpu().numpy()

            self.voe_value_list[self.s0] = plaus_delta_max.cpu().numpy()

    def get_per_step_output(self):
        if len(self.obj_list):
            super_plauslist = []
            for tid, obj in enumerate(self.obj_list):
                super_plauslist.append(obj.plauslist)
            super_plauslist = torch.stack(super_plauslist, dim=1).cuda()
            per_frame_plaus = torch.min(super_plauslist[:(self.s0+1)]).item()

            choice = "plausible" if per_frame_plaus > 0.5 else "implausible"

            confidence = np.abs(per_frame_plaus-0.5)*2

            voe_xy = [{}]
            if self.voe_xy_list[self.s0].sum() != 0:
                voe_xy = [{'x':float(self.voe_xy_list[self.s0][0]), 'y': float(self.voe_xy_list[self.s0][1])}]

            print(self.voe_heatmap_list[self.s0].astype(float))
            voe_heatmap = self.voe_heatmap_list[self.s0].astype(float)
            
            return choice, confidence, voe_xy, voe_heatmap
        else:
            heatmap = np.zeros((400,600)).astype(float)
            return "plausible", 1, [{}], heatmap

    def get_final_plausibility(self):
        if len(self.obj_list):
            super_lrtlist = []
            super_tidlist = []
            super_scorelist = []
            super_plauslist = []
            for tid, obj in enumerate(self.obj_list):
                super_lrtlist.append(obj.lrtlist)
                super_scorelist.append(obj.scorelist)
                super_plauslist.append(obj.plauslist)
                super_tidlist.append(obj.tidlist)
            super_lrtlist = torch.stack(super_lrtlist, dim=1).cuda()
            super_scorelist = torch.stack(super_scorelist, dim=1).cuda()
            super_plauslist = torch.stack(super_plauslist, dim=1).cuda()
            super_tidlist = torch.stack(super_tidlist, dim=1).cuda()

            final_plausibility = torch.min(super_plauslist).item()

            choice = "plausible" if final_plausibility > 0.5 else "implausible"
            confidence = np.abs(final_plausibility-0.5)*2

            return choice, confidence
        return "plausible", 1.0

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
