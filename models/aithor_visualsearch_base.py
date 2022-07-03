from ai2thor.controller import Controller
import numpy as np
import os
from arguments import args
from argparse import Namespace
from tensorboardX import SummaryWriter
from ai2thor_docker.ai2thor_docker.x_server import startx
import ipdb
st = ipdb.set_trace
import utils.aithor
import torch
import pickle
from task_base.aithor_base import Base
from backend import saverloader
import matplotlib.pyplot as plt
from PIL import Image
# from util import box_ops
import math
from ai2thor.util.metrics import compute_single_spl, get_shortest_path_to_object_type
import random
import scipy
import cv2
import copy
from task_base.tidy_task import TIDEE_TASK
from tidee.object_tracker import ObjectTrack
from tidee.navigation import Navigation
from task_base.animation_util import Animation

class Ai2Thor_Base(Base):

    def __init__(self):   

        super(Ai2Thor_Base, self).__init__()

    def save_data(self):
    
        
        # inds = list(np.flip(np.arange(len(self.mapnames_test))))
        inds = list(np.arange(len(self.mapnames_test)))
        # print(inds)
        # for m in range(len(self.mapnames_test)):
        for m in inds:
            mapname = self.mapnames_test[m]
            print("MAPNAME=", mapname)
            # self.controller.reset(scene=mapname)
            root = os.path.join(args.test_data_directory, mapname) 
            if not os.path.exists(root):
                os.mkdir(root)
            for n in range(self.num_save_traj):
                pickle_fname = f'{n}.p'
                fname = os.path.join(root, pickle_fname)
                if os.path.exists(fname):
                    continue

                while True:
                    self.controller.reset(scene=mapname)
                    bounds = utils.aithor.get_scene_bounds(self.controller)
                    obs_dict = self.run_agent(bounds)
                    if obs_dict is not None:
                        break
                # with open(fname, 'rb') as f:
                #     obs_dict = pickle.load(f)
                print("saving", fname)
                with open(fname, 'wb') as f:
                    pickle.dump(obs_dict, f, protocol=4)
                print("done")

    def run_eval_object_goal_nav(self, mapname=None, z=[0.05, 2.0]):
        '''
        Object goal naviagtion where agent must navigate to each object class that exists in the scene
        '''

        objects = self.controller.last_event.metadata['objects']
        object_classes = sorted(list(set(list([objects[i]['objectType'] for i in range(len(objects))]))))

        print("Number of object classes:", len(object_classes))
        print(object_classes)

        # np.random.seed(41)
        # inds_ = np.random.choice(np.arange(len(object_classes)), size=3, replace=False)
        # object_classes = [object_classes[i] for i in list(inds_)] #list(np.random.choice(np.array(object_classes), 2))
        # print(object_classes)
        # return [], [], []

        # SPLs = []
        time_steps = []
        successes = []
        object_types = []
        shortest_dists = []
        max_steps = args.max_steps_object_goal_nav
        num_loops= 0
        for obj_search in object_classes:
            # obj_search = 'Spoon'
            print("MODE IS", self.mode)
            print(f"Object to search for: {obj_search}")
            if obj_search not in self.include_classes:
                print("Skipping ", obj_search)
                num_loops += 1
                continue

            # use tidy task environment for this
            self.tidee_task = TIDEE_TASK(
                self.controller, 
                'test', 
                max_episode_steps=args.max_steps_object_goal_nav
                )
            self.tidee_task.controller.reset(scene=mapname)
            self.tidee_task.next_ep_called = True # hack to start episode
            self.tidee_task.done_called = False
            self.tidee_task.mapname_current = mapname

            navigation = Navigation(
                # controller=controller, 
                keep_head_down=args.keep_head_down, 
                keep_head_straight=args.keep_head_straight, 
                search_pitch_explore=args.search_pitch_explore, 
                pix_T_camX=self.pix_T_camX,
                task=self.tidee_task,
                )

            object_tracker = ObjectTrack(
                    self.name_to_id, 
                    self.id_to_name, 
                    self.include_classes, 
                    self.W, self.H, 
                    pix_T_camX=self.pix_T_camX, 
                    ddetr=self.ddetr, 
                    controller=None, 
                    use_gt_objecttrack=False,
                    do_masks=True,
                    use_solq=True,
                    id_to_mapped_id=self.id_to_mapped_id,
                    on_aws = False, 
                    navigator=navigation,
                    )

            print("Agent exploring environment...")
            # initalize navigation map
            obs_dict = navigation.explore_env(
                object_tracker=object_tracker, 
                vis=None, 
                return_obs_dict=True,
                max_fail=30, 
                )

            # start episode given explored map - compare random vs vsn network searching
            object_tracker.objects_track_dict = {}
            self.tidee_task.step_count = 0

            if args.create_movie:
                if not os.path.exists(f'{args.movie_dir}'):
                    os.mkdir(f'{args.movie_dir}')
                print("LOGGING THIS ITERATION")
                vis = Animation(self.W, self.H, navigation=navigation, name_to_id=self.name_to_id)
                print('Height:', self.H, 'Width:', self.W)
            else:
                vis = None

            if self.mode == "vsn_search": 

                rgb = torch.from_numpy(obs_dict['rgb']).cuda().float()
                xyz = torch.from_numpy(obs_dict['xyz']).cuda().float()
                origin_T_camX0 = torch.from_numpy(obs_dict['origin_T_camX0']).cuda().float()
                camX0_T_camX = torch.from_numpy(obs_dict['camX0_T_camX']).cuda().float()
                # camX0_candidate = 0 #np.random.choice(obs_dict['camX0_candidates'])

                rgb_batch = [rgb]
                xyz_batch = [xyz]
                camX0_T_camX_batch = [camX0_T_camX]
                obj_info_all = {}
                obj_info_all[0] = {}
                obj_info_all[0]['obj_id'] = self.name_to_id[obj_search]
                    
                # prepare supervision
                scene_centroid = {'x': 0., 'y': 0., 'z': 0.}
                    
                # prepare supervision
                targets, vox_util = self.model.prepare_supervision(
                    obj_info_all,
                    camX0_T_camX,
                    xyz,
                    scene_centroid,
                    self.Z,self.Y,self.X,
                    inference=True, 
                    )            
                targets_batch = [targets]
                vox_util_batch = [vox_util]
                
                print("running VSN network")
                with torch.no_grad():
                    forward_dict = self.model(
                        rgb_batch, xyz_batch, camX0_T_camX_batch, 
                        self.Z, self.Y, self.X, 
                        vox_util_batch, targets_batch,
                        mode='test',
                        # summ_writer=self.summ_writer,
                        do_inference=True,
                        do_loss=False,
                        objects_track_dict_batch=[object_tracker.objects_track_dict],
                        )
                inference_dict = self.model.inference_alt(
                    forward_dict['feat_memX0'],
                    forward_dict['feat_pos_logits'],
                    targets_batch,
                    self.Z,self.Y,self.X,
                    vox_util_batch,
                    camX0_T_camX_batch,
                    summ_writer=None
                    )
                print("Done.")
                
                feat_mem_logits = inference_dict['feat_mem_logits']
                xyz_origin_poss = inference_dict['xyz_origin_poss']
                xyz_origin_select = inference_dict['xyz_origin_select']
                farthest_pts = inference_dict['farthest_pts']
                thresh_mem = inference_dict['thresh_mem']   

                add_ = (((navigation.explorer.mapper.resolution * navigation.explorer.mapper.map_sz) - (args.voxel_max - args.voxel_min))/2) / navigation.explorer.mapper.resolution #((args.voxel_max - args.voxel_min)/self.Z) #navigation.explorer.mapper.resolution
                # farthest_pts += (add_/2)
                offset_ = navigation.explorer.mapper.resolution/((args.voxel_max - args.voxel_min)/self.Z)
                mult_Z = (navigation.explorer.mapper.map_sz*offset_)/self.Z*2 #* navigation.explorer.mapper.resolution/((args.voxel_max - args.voxel_min)/self.Z) #navigation.explorer.mapper.map_sz/self.Z #navigation.explorer.mapper.resolution/((args.voxel_max - args.voxel_min)/self.Z) * navigation.explorer.mapper.map_sz/ (self.Z * 13/8)
                # mult_X = (navigation.explorer.mapper.map_sz*offset_)/self.Z #* navigation.explorer.mapper.resolution/((args.voxel_max - args.voxel_min)/self.Z) #navigation.explorer.mapper.map_sz/self.Z #navigation.explorer.mapper.resolution/((args.voxel_max - args.voxel_min)/self.X) * navigation.explorer.mapper.map_sz/ (self.X * 13/8)
                mapper_inds = np.array([farthest_pts[:,0]*mult_Z+add_, farthest_pts[:,1]*mult_Z+add_]).T.astype(np.int32)

                if False:
                    from utils.wctb import ThorPositionTo2DFrameTranslator
                    mapname = mapnames[ind_v]
                    self.controller.reset(scene=mapname)
                    self.controller.step({"action": "ToggleMapView"})
                    cam_position = self.controller.last_event.metadata["cameraPosition"]
                    cam_orth_size = self.controller.last_event.metadata["cameraOrthSize"]
                    pos_translator = ThorPositionTo2DFrameTranslator(
                        self.controller.last_event.frame.shape, (cam_position["x"], cam_position["y"], cam_position["z"]), cam_orth_size
                    )
                    overhead_map = self.controller.last_event.frame
                    overhead_map = torch.from_numpy(overhead_map.copy()).cuda().permute(2,0,1).unsqueeze(0)
                    name = f'{mode}_inference/overheadmap_{b_classes[v]}_(right=gt)_{v}'
                    self.summ_writer.summ_rgb(name, overhead_map)

                if args.create_movie: # some plotting
                    def transparent_cmap(cmap, N=255):
                        "Copy colormap and set alpha values"

                        mycmap = cmap
                        mycmap._init()
                        mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
                        return mycmap

                    if not os.path.exists(f'{args.movie_dir}/vsn_search/'):
                        os.mkdir(f'{args.movie_dir}/vsn_search/')

                    #Use base cmap to create transparent
                    mycmap = transparent_cmap(plt.cm.Reds)

                    plt.figure(5); plt.clf()
                    plt.imshow(feat_mem_logits[0].cpu().numpy(), cmap=mycmap)
                    plt.savefig(f'{args.movie_dir}/vsn_search/{mapname}_{obj_search}.png')
                    plt.figure(4); plt.clf()
                    plt.imshow(thresh_mem, cmap=mycmap)
                    plt.savefig(f'{args.movie_dir}/vsn_search/{mapname}_{obj_search}_thresh_mem_nopts.png')
                    plt.figure(2); plt.clf()
                    plt.imshow(thresh_mem, cmap=mycmap)
                    plt.scatter(farthest_pts[:,1], farthest_pts[:,0])
                    plt.savefig(f'{args.movie_dir}/vsn_search/{mapname}_{obj_search}_thresh_mem.png')
                    plt.figure(3)
                    plt.clf()
                    plt.imshow(feat_mem_logits[0].cpu().numpy(), cmap=mycmap)
                    plt.scatter(farthest_pts[:,1], farthest_pts[:,0])
                    # plt.colorbar()
                    plt.savefig(f'{args.movie_dir}/vsn_search/{mapname}_{obj_search}_farthest.png')

                    # from utils.wctb import ThorPositionTo2DFrameTranslator
                    # self.controller.step({"action": "ToggleMapView"})
                    # cam_position = self.controller.last_event.metadata["cameraPosition"]
                    # cam_orth_size = self.controller.last_event.metadata["cameraOrthSize"]
                    # pos_translator = ThorPositionTo2DFrameTranslator(
                    #     self.controller.last_event.frame.shape, (cam_position["x"], cam_position["y"], cam_position["z"]), cam_orth_size
                    # )
                    # overhead_map = self.controller.last_event.frame
                    # self.controller.step({"action": "ToggleMapView"})
                    # plt.figure(5)
                    # plt.clf()
                    # plt.imshow(overhead_map)
                    # plt.savefig('images/test.png')
                    # st()

                    

                    from utils.wctb import ThorPositionTo2DFrameTranslator
                    self.controller.step({"action": "ToggleMapView"})
                    cam_position = self.controller.last_event.metadata["cameraPosition"]
                    cam_orth_size = self.controller.last_event.metadata["cameraOrthSize"]
                    pos_translator = ThorPositionTo2DFrameTranslator(
                        self.controller.last_event.frame.shape, (cam_position["x"], cam_position["y"], cam_position["z"]), cam_orth_size
                    )
                    overhead_map = self.controller.last_event.frame
                    self.controller.step({"action": "ToggleMapView"})
                    plt.figure(6)
                    plt.clf()
                    plt.imshow(overhead_map)
                    plt.savefig(f'{args.movie_dir}/vsn_search/{mapname}.png')

                    w, h = list(overhead_map.shape[:2])
                    heatmap = feat_mem_logits[0].cpu().numpy()
                    resized_up = cv2.resize(heatmap, (w,h), interpolation= cv2.INTER_LINEAR)
                    y, x = np.mgrid[0:h, 0:w]
                    plt.figure(5)
                    plt.clf()
                    fig, ax = plt.subplots(1, 1)
                    # ax.imshow(overhead_map)
                    cb = ax.contourf(x, y, resized_up, 15, cmap=mycmap)
                    plt.colorbar(cb)
                    plt.savefig(f'{args.movie_dir}/vsn_search/{mapname}_{obj_search}_contour.png')
                    # st()
                    plt.figure(5)
                    plt.clf()
                    fig, ax = plt.subplots(1, 1)
                    # ax.imshow(overhead_map)
                    cb = ax.contourf(x, y, resized_up, 15, cmap=mycmap)
                    ax.scatter(farthest_pts[:,1]*(w/heatmap.shape[0]), farthest_pts[:,0]*(h/heatmap.shape[1]))
                    plt.colorbar(cb)
                    plt.savefig(f'{args.movie_dir}/vsn_search/{mapname}_{obj_search}_contour_with_locs.png')  
                    # st()   

            agent_positions = []
            in_view = {}
            steps = 0
            map_poss = []

            for p in range(self.num_search_locs):
                if self.mode == "vsn_search":
                    # xyz_origin_select_p = xyz_origin_select[p].cpu().numpy()

                    # obj_center_camX0_ = {'x':xyz_origin_select_p[0], 'y':-xyz_origin_select_p[1], 'z':xyz_origin_select_p[2]}
                    # map_pos = navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)

                    map_pos = list(mapper_inds[p])
                    ind_i, ind_j  = navigation.get_clostest_reachable_map_pos(map_pos)
                    # ind_i, ind_j = map_pos
                    # ind_i, ind_j  = list(mapper_inds[p])
                elif self.mode == "random":
                    ind_i, ind_j = navigation.get_reachable_map_locations(sample=True)
                else:
                    assert(False)

                map_poss.append([ind_i, ind_j])

                navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.dist_thresh, search_mode=True)

                out = navigation.navigate_to_point_goal(
                    vis=vis, 
                    text=f"Searching for {obj_search}", 
                    object_tracker=object_tracker,
                    search_object=obj_search,
                    )

                # if len(out)==0:
                #     out = navigation.search_local_region(
                #         vis=vis, 
                #         text=f"Searching for {obj_search}", 
                #         object_tracker=object_tracker,
                #         search_object=obj_search,
                #         )

                if len(out)>0:
                    in_view['centroid'] = out['centroids'][0]
                    in_view['object_name'] = obj_search
                    break

            if args.create_movie and self.mode == "vsn_search":
                plt.figure(10); plt.clf()
                m_vis = np.invert(navigation.explorer.mapper.get_traversible_map(
                    navigation.explorer.selem, 1,loc_on_map_traversible=True))
                plt.imshow(m_vis, origin='lower', vmin=0, vmax=1,
                         cmap='Greys')
                for map_pos in map_poss:
                    plt.plot(map_pos[1], map_pos[0], color='blue', marker='o',linewidth=10, markersize=12)
                plt.savefig(f'{args.movie_dir}/vsn_search/{mapname}_{obj_search}_navigation_map.png')

            if len(in_view)>0:
                print("DETECTED OBJECT!")
                obj_center = in_view['centroid']
                obj_name = in_view['object_name']

                obj_center_camX0_ = {'x':obj_center[0], 'y':-obj_center[1], 'z':obj_center[2]}
                map_pos = navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)

                ind_i, ind_j  = navigation.get_clostest_reachable_map_pos(map_pos) # get closest navigable point to object in overhead map

                navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.dist_thresh) # set point goal in map
                navigation.navigate_to_point_goal(vis=vis, text=f"Navigate to {obj_name}", object_tracker=object_tracker)
                
                navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.dist_thresh)
                navigation.orient_camera_to_point(obj_center_camX0_, vis=vis, text=f"Orient to {obj_name}", object_tracker=object_tracker)

            success = False
            for obj in self.controller.last_event.metadata['objects']:
                if obj["objectType"]==obj_search:
                    if obj["visible"]:
                        success = True
            
            if success:
                successful_path = 1
                steps = self.tidee_task.step_count
            else:
                successful_path = 0
                steps = args.max_steps_object_goal_nav
            
            '''
            SPL doesn't quite work because there may be multiple objects of the same class
            '''
            # SPL = compute_single_spl(agent_positions, shortest_path, 1)
            # SPLs.append(SPL)

            time_steps.append(steps)
            successes.append(successful_path)
            object_types.append(obj_search)

            print("TIME STEPS:", steps)
            print("successful_path:", successful_path)
            print(time_steps)
            print(successes)
            
            num_loops += 1
            print(f"Finished {num_loops}/{len(object_classes)}")

            if vis is not None:
                for _ in range(10):
                    vis.add_frame(self.tidee_task.controller.last_event.frame, text='FINAL')
                # render movie of agent
                vis.render_movie(args.movie_dir, 0, tag=f'success={success}_{mapname}_{args.object_navigation_policy_name}_{args.tag}_object_nav_{obj_search}')

        return time_steps, successes, object_types #, shortest_dists