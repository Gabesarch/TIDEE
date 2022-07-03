import ipdb
st = ipdb.set_trace
from utils.wctb import Utils, Relations
from utils.wctb import ThorPositionTo2DFrameTranslator
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import utils.geom
import utils.aithor
import utils.samp
from nets.visual_search_network import VSN
from backend import saverloader
from arguments import args
import sys
from PIL import Image
import math

class Placement():

    def __init__(
        self, 
        W,H,
        pix_T_camX,
        include_classes, 
        name_to_id, 
        id_to_name, 
        class_to_save_to_id, 
        ddetr=None,
        task=None,
        special_classes=[]
        ): 

        self.W, self.H, = W,H
        self.pix_T_camX = pix_T_camX

        self.include_classes = include_classes
        self.name_to_id = name_to_id
        self.id_to_name = id_to_name

        self.task = task
        
        if args.do_vsn_search:
            self.X, self.Y, self.Z = args.X, args.Y, args.Z
            self.fov = args.fov
            num_classes = len(self.include_classes) - 1 - len(special_classes) # remove no object class + special classes

            self.vsn = VSN(
                num_classes, 
                args.include_rgb, 
                self.fov,self.Z,self.Y,self.X,
                class_to_save_to_id=class_to_save_to_id,
                do_masked_pos_loss=args.do_masked_pos_loss, 
                )
            self.vsn.cuda().eval()

            _ =saverloader.load_from_path(args.vsn_checkpoint, self.vsn, None, strict=True, lr_scheduler=None)

            XMIN = -4.0 # right (neg is left)
            XMAX = 4.0 # right
            YMIN = -4.0 # down (neg is up)
            YMAX = 4.0 # down
            ZMIN = -4.0 # forward
            ZMAX = 4.0 # forward
            self.bounds = torch.tensor([XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX]).cuda()

            self.num_search_locs = args.num_search_locs_object
            self.dist_thresh = args.dist_thresh

            # self.general_receptacles_classes = ['DiningTable', 'TVStand', 'CoffeeTable', 'SideTable', 'CounterTop', 'Desk', 'Dresser', 'Bed', 'Sink']

            self.general_receptacles_classes = ['CounterTop', 'DiningTable', 'CoffeeTable', 'SideTable', 'Sink', 'TowelHolder',
                'Desk','Bed',  'TVStand', 'Sofa', 'ArmChair', 'SinkBasin', 
                'Drawer', 'Chair', 'Shelf', 'Dresser', 'Fridge', 'Ottoman',   'DogBed', 'ShelvingUnit', 'Cabinet',
                'StoveBurner', 'Microwave', 'CoffeeMachine', 'GarbageCan', 
                'Toaster', 'LaundryHamper', 'Stool',  'Bathtub', 'Footstool', 'BathtubBasin', 'Safe']

            self.ddetr = ddetr

    def search_for_general_receptacle(self, navigation, vis=None, object_tracker=None):
        found_obj = False
        in_view = {}

        general_receptacles_ids = [self.name_to_id[general_receptacles_classes_i] for general_receptacles_classes_i in self.general_receptacles_classes]

        centroids, labels = object_tracker.get_centroids_and_labels(
            return_ids=False, object_cat=search_object
            )
        if len(centroids)>0:
            out = {'centroids':centroids, 'labels':labels}
        else:
            # try to search for one
            out = navigation.search_random_locs_for_object(
                vis=vis, 
                text=f"Searching for {self.general_receptacles_classes}", 
                object_tracker=object_tracker,
                search_object=self.general_receptacles_classes,
            )

        if len(out)>0:
            in_view['centroid'] = out['centroids'][0]
            in_view['object_name'] = out['labels'][0]
        
        return in_view

    def vsn_search_object(self, navigation, obj_oop_meta, obs_dict, vis=None, object_tracker=None):
        found_obj = False

        obj_search = obj_oop_meta['object_name']

        general_receptacles_ids = [self.name_to_id[general_receptacles_classes_i] for general_receptacles_classes_i in self.general_receptacles_classes]

        rgb = torch.from_numpy(obs_dict['rgb']).cuda().float()
        xyz = torch.from_numpy(obs_dict['xyz']).cuda().float()
        origin_T_camX0 = torch.from_numpy(obs_dict['origin_T_camX0']).cuda().float()
        camX0_T_camX = torch.from_numpy(obs_dict['camX0_T_camX']).cuda().float()
        camX0_candidate = np.random.choice(obs_dict['camX0_candidates'])

        rgb_batch = [rgb]
        xyz_batch = [xyz]
        camX0_T_camX_batch = [camX0_T_camX]
        obj_info_all = {}
        obj_info_all[0] = {}
        obj_info_all[0]['obj_id'] = self.name_to_id[obj_search]
            
        # prepare supervision
        scene_centroid = {'x': 0., 'y': 0., 'z': 0.}
            
        # prepare supervision
        targets, vox_util = self.vsn.prepare_supervision(
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
            forward_dict = self.vsn(
                rgb_batch, xyz_batch, camX0_T_camX_batch, 
                self.Z, self.Y, self.X, 
                vox_util_batch, targets_batch,
                mode='test',
                # summ_writer=self.summ_writer,
                do_inference=True,
                do_loss=False,
                objects_track_dict_batch=[object_tracker.objects_track_dict],
                )
        inference_dict = self.vsn.inference_alt(
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

        for p in range(args.num_search_locs_object):
            # xyz_origin_select_p = xyz_origin_select[p].cpu().numpy()

            # obj_center_camX0_ = {'x':xyz_origin_select_p[0], 'y':-xyz_origin_select_p[1], 'z':xyz_origin_select_p[2]}
            # map_pos = navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)

            map_pos = list(mapper_inds[p])

            ind_i, ind_j  = navigation.get_clostest_reachable_map_pos(map_pos) # get closest navigable point to object in overhead map

            navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.dist_thresh) # set point goal in map
            
            navigation.navigate_to_point_goal(
                vis=vis, 
                text=f"Navigate to VSN goal for {obj_search}", 
                object_tracker=object_tracker,
                # search_object=obj_search,
                )

            out = navigation.search_local_region(
                vis=vis, 
                text=f"Searching for {obj_search}", 
                object_tracker=object_tracker,
                search_object=obj_search,
                )

            if len(out)>0:
                if vis is not None:
                    rgb, depth = navigation.get_obs(head_tilt=navigation.explorer.head_tilt)
                    for _ in range(5):
                        # vis.add_frame(controller.last_event.frame, text=f"Detection; score={det_dict['pred_scores']}", box=det_dict['box'])
                        vis.add_frame(rgb, text=f"Detected receptacle")
                in_view['centroid'] = out['centroids'][0]
                in_view['object_name'] = obj_search
                print(f"OBJECT {obj_search} DETECTED")
                break

        return in_view, objs_to_search

    def search_for_related_obj(
        self, 
        navigation, 
        obs_dict, 
        objs_to_search, 
        vis=None, 
        object_tracker=None
        ):

        in_view = {}

        rgb = torch.from_numpy(obs_dict['rgb']).cuda().float()
        xyz = torch.from_numpy(obs_dict['xyz']).cuda().float()
        origin_T_camX0 = torch.from_numpy(obs_dict['origin_T_camX0']).cuda().float()
        camX0_T_camX = torch.from_numpy(obs_dict['camX0_T_camX']).cuda().float()
        camX0_candidate = np.random.choice(obs_dict['camX0_candidates'])

        rgb_batch = [rgb]
        xyz_batch = [xyz]
        camX0_T_camX_batch = [camX0_T_camX]
        obj_info_all = {}

        if 'Floor' in objs_to_search:
            objs_to_search.remove('Floor')

        if len(objs_to_search)>=args.num_related_objects:
            num_related_objs = args.num_related_objects
        else:
            num_related_objs = len(objs_to_search)
        
        
        for o_i in range(num_related_objs):
            obj_search = objs_to_search.pop(0)
            obj_info_all[o_i] = {}
            obj_info_all[o_i]['obj_id'] = self.name_to_id[obj_search]
            
            print("Searching for ", obj_search)

            # see if already found this object
            if object_tracker is not None:
                # obj_track_search = object_tracker.objects_track_dict[obj_search]
                # locs = obj_track_search['locs']
                centroids, labels = object_tracker.get_centroids_and_labels(return_ids=False, object_cat=obj_search)
                if len(centroids)>0:
                    print("HAVE THIS OBJECT IN MEMORY! Retrieving info from memory...")
                    # scores = np.array(obj_track_search['scores'])
                    # argmax_score = np.argmax(scores)
                    centroid = centroids[0] # first one is highest scoring
                    in_view['centroid'] = centroid
                    in_view['object_name'] = obj_search
                    found_obj = True
                    if vis is not None and args.visualize_vsn:
                        vis.add_found_in_memory(obj_search, centroid)
                    # if vis is not None:
                    #     for i in range(6):
                    #         vis.add_frame(controller.last_event.frame, text=f"Found related object in memory")
                    break
                else:
                    print(f"{obj_search} missing from memory...\n Running visual search network...")

            scene_centroid = {'x': 0., 'y': 0., 'z': 0.}
            
            # prepare supervision
            targets, vox_util = self.vsn.prepare_supervision(
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
                forward_dict = self.vsn(
                    rgb_batch, xyz_batch, camX0_T_camX_batch, 
                    self.Z, self.Y, self.X, 
                    vox_util_batch, targets_batch,
                    mode='test',
                    # summ_writer=self.summ_writer,
                    do_inference=True,
                    do_loss=False,
                    objects_track_dict_batch=[object_tracker.objects_track_dict],
                    )
            inference_dict = self.vsn.inference_alt(
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
            # xyz_origin_poss = xyz_origin_poss.reshape(xyz_origin_poss.shape[0], self.Z, self.X, 3)
            
            # plt.figure(1); plt.clf()
            # plt.imshow(thresh_mem)
            # plt.scatter(farthest_pts[:,1], farthest_pts[:,0])
            # plt.savefig('images/test.png')
            # plt.figure(2); plt.clf()
            # plt.imshow(feat_mem_logits[0].cpu().numpy())
            # plt.scatter(farthest_pts[:,1], farthest_pts[:,0])
            # plt.savefig('images/test2.png')
            # st()

            if vis is not None and args.visualize_vsn:
                feat_mem_sigmoid = feat_mem_logits[0].cpu().numpy()
                scores_fp = []
                for fp in range(len(farthest_pts)):
                    fp_ = farthest_pts[fp].astype(int)
                    scores_fp.append(feat_mem_sigmoid[fp_[0], fp_[1]])
                scores_fp = np.array(scores_fp)
                scores_argsort = np.flip(np.argsort(scores_fp))
                xyz_origin_select = xyz_origin_select[scores_argsort.copy(), :]
                feat_mem_logits_vis = feat_mem_sigmoid #feat_mem_logits[0].cpu().numpy()
                vis.add_active_search_visual(farthest_pts, scores_argsort, feat_mem_logits_vis, thresh_mem, obj_search)

            # st()
            in_view = {}
            # navigate to each of the points
            for p in range(args.num_search_locs_object):
                xyz_origin_select_p = xyz_origin_select[p].cpu().numpy()

                obj_center_camX0_ = {'x':xyz_origin_select_p[0], 'y':-xyz_origin_select_p[1], 'z':xyz_origin_select_p[2]}
                map_pos = navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)

                ind_i, ind_j  = navigation.get_clostest_reachable_map_pos(map_pos) # get closest navigable point to object in overhead map

                navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.dist_thresh) # set point goal in map
                
                # object_tracker.score_threshold = args.confidence_threshold_searching
                out = navigation.navigate_to_point_goal(
                    vis=vis, 
                    text=f"Searching for {obj_search}", 
                    object_tracker=object_tracker,
                    search_object=obj_search,
                    )

                if len(out)==0:
                    out = navigation.search_local_region(
                        vis=vis, 
                        text=f"Searching for {obj_search}", 
                        object_tracker=object_tracker,
                        search_object=obj_search,
                        )

                if len(out)>0:
                    if vis is not None:
                        rgb, depth = navigation.get_obs(head_tilt=navigation.explorer.head_tilt)
                        for _ in range(5):
                            vis.add_frame(rgb, text=f"Detected receptacle")
                    
                    in_view['centroid'] = out['centroids'][0]
                    in_view['object_name'] = obj_search
                    print(f"OBJECT {obj_search} DETECTED")
                    break

            if len(in_view)>0:
                break
        return in_view, objs_to_search

    def place_object_on_related(
        self, 
        navigation, 
        obj_det, 
        found_oop_dict, 
        vis=None, 
        object_tracker=None, 
        objs_to_search=None
        ):
        '''
        Places object on related
        '''

        offsets = [
                [0, 0], 
                [0, 20], 
                [20, 0], 
                [0, -20], 
                [-20, 0], 
                [20, 20], 
                [20, -20], 
                [-20, 20], 
                [-20, -20]
                ]

        if len(obj_det)==0:
            print("Related object not found!")
            # print("Dropping object..")
            success = False
        else:
            
            obj_center = obj_det['centroid']
            obj_name = obj_det['object_name']

            print(f"Attempting to place {found_oop_dict['object_name']} on {obj_name}")

            obj_center_camX0_ = {'x':obj_center[0], 'y':-obj_center[1], 'z':obj_center[2]}
            map_pos = navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)

            ind_i, ind_j  = navigation.get_clostest_reachable_map_pos(map_pos) # get closest navigable point to object in overhead map

            navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.dist_thresh) # set point goal in map
            navigation.navigate_to_point_goal(vis=vis, text=f"Navigate to {obj_name}", object_tracker=object_tracker)
            
            navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.dist_thresh)
            navigation.orient_camera_to_point(obj_center_camX0_, vis=vis, text=f"Orient to {obj_name}", object_tracker=object_tracker)

            # get 2D reprojection
            camX0_T_camX = navigation.explorer.get_camX0_T_camX()
            obj_center_camX0_reproj = {'x':obj_center[0], 'y':obj_center[1], 'z':obj_center[2]}
            camX_T_camX0 = utils.geom.safe_inverse_single(camX0_T_camX)
            rgb = self.task.get_observations()["rgb"]
            point_2D, method_text = object_tracker.get_2D_point(
                camX_T_origin=camX_T_camX0, 
                obj_center_camX0_=obj_center_camX0_reproj, 
                object_category=obj_name, 
                rgb=rgb, 
                score_threshold=args.score_threshold_interaction
                )

            success = navigation.interact_object_xy(
                "PutObject", 
                point_2D, 
                vis=vis, 
                offsets=offsets,
                num_tries_place=3,
                )

            if success:
                print("Successfully placed!")

        if not success:
            print("Will try to place on a general receptacle class...")
            # first search for remaining retrieved receptacles, then general ones
            general_classes_search = self.general_receptacles_classes.copy()
            for cl in objs_to_search:
                if cl in general_classes_search:
                    general_classes_search.remove(cl)
            classes_to_search = objs_to_search + general_classes_search
            tries = 0
            for general_class in classes_to_search:

                centroids, labels = object_tracker.get_centroids_and_labels(return_ids=False, object_cat=general_class)
                if len(centroids)>0:
                    print(f"Attempting to place {found_oop_dict['object_name']} on {general_class}.")
                    centroid = centroids[0] # first one is highest scoring
                    obj_center = centroid
                    obj_name = general_class
                else:
                    continue

                obj_center_camX0_ = {'x':obj_center[0], 'y':-obj_center[1], 'z':obj_center[2]}
                map_pos = navigation.get_map_pos_from_aithor_pos(obj_center_camX0_)

                ind_i, ind_j  = navigation.get_clostest_reachable_map_pos(map_pos) # get closest navigable point to object in overhead map

                navigation.set_point_goal(ind_i, ind_j, dist_thresh=self.dist_thresh) # set point goal in map
                navigation.navigate_to_point_goal(vis=vis, text=f"Navigate to {obj_name}", object_tracker=object_tracker)
                
                navigation.set_point_goal(int(map_pos[0]), int(map_pos[1]), dist_thresh=self.dist_thresh)
                navigation.orient_camera_to_point(obj_center_camX0_, vis=vis, text=f"Orient to {obj_name}", object_tracker=object_tracker)

                # get 2D reprojection
                camX0_T_camX = navigation.explorer.get_camX0_T_camX()
                obj_center_camX0_reproj = {'x':obj_center[0], 'y':obj_center[1], 'z':obj_center[2]}
                camX_T_camX0 = utils.geom.safe_inverse_single(camX0_T_camX)
                rgb = self.task.get_observations()["rgb"]
                point_2D, method_text = object_tracker.get_2D_point(
                    camX_T_origin=camX_T_camX0, 
                    obj_center_camX0_=obj_center_camX0_reproj, 
                    object_category=obj_name, 
                    rgb=rgb, 
                    score_threshold=args.score_threshold_interaction
                    )

                success = navigation.interact_object_xy("PutObject", point_2D, vis=vis, offsets=offsets)

                if success:
                    break

                tries += 1

                if tries >= args.num_related_objects:
                    break

        return success