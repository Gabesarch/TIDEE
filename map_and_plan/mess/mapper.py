import logging
import numpy as np
import skimage
import skimage.morphology
from map_and_plan.mess import depth_utils as du
from map_and_plan.mess import rotation_utils as ru


import matplotlib.pyplot as plt
import tkinter
import matplotlib
import matplotlib.gridspec as gridspec
# matplotlib.use('TkAgg')

import ipdb
st = ipdb.set_trace


class Mapper():
    def __init__(self, C, sc, origin, map_size, resolution, max_depth=164, z_bins=[0.05,3], max_obj=50,loc_on_map_selem = skimage.morphology.disk(2), bounds=None):
        # Internal coordinate frame is X, Y into the scene, Z up.
        self.sc = sc
        self.C = C
        self.resolution = resolution
        self.max_depth = max_depth
        
        self.z_bins = z_bins
        map_sz = int(np.ceil((map_size*100)//(resolution*100)))
        self.map_sz = map_sz
        print("MAP SIZE:", map_sz)
        self.map = np.zeros((map_sz, map_sz, len(self.z_bins)+1), dtype=np.float32)
        self.semantic_map = np.zeros((map_sz, map_sz, max_obj), dtype=np.float32)
        self.loc_on_map = np.zeros((map_sz, map_sz), dtype=np.float32)
        
        self.origin_xz = np.array([origin['x'], origin['z']])
        self.origin_map = np.array([(self.map.shape[0]-1)/2, (self.map.shape[0]-1)/2], np.float32)
        #self.origin_map = self._optimize_set_map_origin(self.origin_xz, self.resolution)
        self.objects = {}
        self.loc_on_map_selem = loc_on_map_selem
        self.added_obstacles = np.ones((map_sz, map_sz), dtype=bool)

        self.num_boxes = 0

        self.step = 0

        self.bounds = None #bounds
    
    def _optimize_set_map_origin(self, origin_xz, resolution):
        return (origin_xz + 15)/ resolution

    def transform_to_current_frame(self, XYZ):
        R = ru.get_r_matrix([0.,0.,1.], angle=self.current_rotation)
        XYZ = np.matmul(XYZ.reshape(-1,3), R.T).reshape(XYZ.shape)
        XYZ[:,:,0] = XYZ[:,:,0] + self.current_position[0] - self.origin_xz[0] + self.origin_map[0]*self.resolution
        XYZ[:,:,1] = XYZ[:,:,1] + self.current_position[1] - self.origin_xz[1] + self.origin_map[1]*self.resolution
        return XYZ

    def update_position_on_map(self, position, rotation):
        self.current_position = np.array([position['x'], position['z']], np.float32)
        self.current_rotation = -np.deg2rad(rotation)
        x, y = self.get_position_on_map()

        # import matplotlib.pyplot as plt
        # import tkinter
        # import matplotlib
        # import matplotlib.gridspec as gridspec
        # # matplotlib.use('TkAgg')
        # import ipdb
        # st = ipdb.set_trace
        #st()
        
        self.loc_on_map[int(y), int(x)] = 1
        # TODO(saurabhg): Mark location on map

    def add_observation(self, position, rotation, elevation, depth, add_obs=True):

        

        d = depth*1.
        d[d > self.max_depth] = 0
        d[d < 0.02] = np.NaN
        d = d / self.sc
        self.update_position_on_map(position, rotation)
        if not add_obs:
            return
        XYZ1 = du.get_point_cloud_from_z(d, self.C);
        # print(np.nanmin(XYZ1[:,:,2]), position['y'], np.nanmin(XYZ1[:,:,2])/position['y'])
        XYZ2 = du.make_geocentric(XYZ1*1, position['y'], elevation)
        XYZ3 = self.transform_to_current_frame(XYZ2)
        counts, is_valids, inds = du.bin_points(XYZ3, self.map.shape[0], self.z_bins, self.resolution)
        # counts2, is_valids2, inds2 = du.bin_points3D(XYZ3, self.map.shape[0], self.z_bins, self.resolution)
        self.map += counts
        # return is_valids2, inds2

        
        # # print(self.map.shape)
        # plt.figure(1); plt.clf()
        # plt.imshow(self.map)
        # plt.savefig(f'images/{self.step}.png')
        # self.step += 1
        
    def get_occupancy_vars(self, position, rotation, elevation, depth, global_downscaling):
        # this gets inds of each pixel in the depth image for a 3D occupancy 
        d = depth*1.
        d[d > self.max_depth] = 0
        d[d < 0.02] = np.NaN
        d = d / self.sc
        self.update_position_on_map(position, rotation)
        XYZ1 = du.get_point_cloud_from_z(d, self.C);
        # print(np.nanmin(XYZ1[:,:,2]), position['y'], np.nanmin(XYZ1[:,:,2])/position['y'])
        XYZ2 = du.make_geocentric(XYZ1*1, position['y'], elevation)
        XYZ3 = self.transform_to_current_frame(XYZ2)
        counts2, is_valids2, inds2 = du.bin_points3D(XYZ3, self.map.shape[0]//global_downscaling, self.z_bins, self.resolution*global_downscaling)
        return counts2, is_valids2, inds2


    # def update_map(self, depth, current_pose):
    #     with np.errstate(invalid="ignore"):
    #         depth[depth > self.vision_range * self.resolution] = np.NaN
    #     point_cloud = du.get_point_cloud_from_z(depth, self.camera_matrix, \
    #                                             scale=self.du_scale)

    #     agent_view = du.transform_camera_view(point_cloud,
    #                                           self.agent_height,
    #                                           self.agent_view_angle)

    #     shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]
    #     agent_view_centered = du.transform_pose(agent_view, shift_loc)

    #     agent_view_flat = du.bin_points(
    #         agent_view_centered,
    #         self.vision_range,
    #         self.z_bins,
    #         self.resolution)

    #     agent_view_cropped = agent_view_flat[:, :, 1]

    #     agent_view_cropped = agent_view_cropped / self.obs_threshold
    #     agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
    #     agent_view_cropped[agent_view_cropped < 0.5] = 0.0

    #     agent_view_explored = agent_view_flat.sum(2)
    #     agent_view_explored[agent_view_explored > 0] = 1.0

    #     geocentric_pc = du.transform_pose(agent_view, current_pose)

    #     geocentric_flat = du.bin_points(
    #         geocentric_pc,
    #         self.map.shape[0],
    #         self.z_bins,
    #         self.resolution)

    #     self.map = self.map + geocentric_flat

    #     map_gt = self.map[:, :, 1] / self.obs_threshold
    #     map_gt[map_gt >= 0.5] = 1.0
    #     map_gt[map_gt < 0.5] = 0.0

    #     explored_gt = self.map.sum(2)
    #     explored_gt[explored_gt > 1] = 1.0

    #     return agent_view_cropped, map_gt, agent_view_explored, explored_gt

    def _get_mask(self, obj, object_masks):
        rgb = np.array([obj.color[x] for x in 'rgb'])
        mask = object_masks == rgb
        mask = np.all(mask, 2)
        return mask

    def add_trophy(self, XYZ3, trophy_mask):
        if "trophy" not in self.objects.keys():
            self.objects['trophy'] = {'channel_id': 1, 'area': 0}
            logging.error("Spotted trophy for the first time!")
        map_channel = self.objects['trophy']['channel_id']
        mask = trophy_mask.copy()
        mask_nan = np.zeros(trophy_mask.shape)
        mask_nan[:] = np.NaN
        mask_nan[mask] = 0
        xyz3 = XYZ3 + mask_nan[:,:,np.newaxis]
        counts, is_valids = du.bin_points(xyz3, self.map.shape[0], [], self.resolution)
        counts[0,0] = 0
        self.semantic_map[:,:,map_channel] += counts[:,:,0]
        area = np.sum(self.semantic_map[:,:,map_channel] > 0)
        if np.log10(area+1) - np.log10(self.objects['trophy']['area']+1) > 1:
            logging.info('Object: {0}: {1}'.format("trophy", area))
        self.objects['trophy'][area] = area
        self.objects['trophy']['in_view'] = True
        self.objects['trophy']['view_mask'] = trophy_mask

    def add_box_closed(self, XYZ3, box_masks):
        for uuid in self.objects:
            if uuid == 'trophy':
                continue
            self.objects[uuid]['in_view'] = False
            self.objects[uuid]['view_mask'] = None
        for idx, box_mask in enumerate(box_masks):
            mask = box_mask.copy()
            mask_nan = np.zeros(mask.shape)
            mask_nan[:] = np.NaN
            mask_nan[mask] = 0
            xyz3 = XYZ3 + mask_nan[:,:,np.newaxis]
            counts, is_valids = du.bin_points(xyz3, self.map.shape[0], [], self.resolution)
            counts[0,0] = 0
            cur_count_map = counts[:,:,0]

            max_overlap_uuid = None
            max_overlap_area = 10
            for uuid in self.objects:
                if "box" not in uuid:
                    continue
                map_channel = self.objects[uuid]['channel_id']
                box_count_map = self.semantic_map[:,:,map_channel]

                overlap_idx = np.logical_and(box_count_map > 0, cur_count_map > 0)
                if overlap_idx.sum() == 0:
                    continue

                overlap_area = np.sum(np.minimum(cur_count_map[overlap_idx], box_count_map[overlap_idx]))
                if overlap_area > max_overlap_area:
                    max_overlap_area = overlap_area
                    max_overlap_uuid = uuid

            if max_overlap_uuid is None:
                # instantiate new box
                #uuid = "box_{}".format(self.num_boxes + 2)
                uuid = self.box_closed_uuid_s[idx]
                self.objects[uuid] = {'channel_id': self.num_boxes + 2, 'area': 0}
                logging.error(f'Spotted object: {uuid}')
                map_channel = self.objects[uuid]['channel_id']
                self.objects[uuid]['in_view'] = True
                self.objects[uuid]['view_mask'] = box_mask
                self.semantic_map[:,:, map_channel] += counts[:,:,0]
                self.num_boxes += 1
            else:
                # merge with previous box
                self.objects[max_overlap_uuid]['in_view'] = True
                self.objects[max_overlap_uuid]['view_mask'] = box_mask
                map_channel = self.objects[max_overlap_uuid]['channel_id']
                self.semantic_map[:,:,map_channel] += cur_count_map

    '''
    def add_objects(self, XYZ3, object_masks, objects):
        for idx, obj in enumerate(objects):
            if obj not in self.objects.keys():
                self.objects[obj] = {'channel_id': len(self.objects) + 1, 'area': 0}
                logging.error(f'Spotted object: {obj}')
            map_channel = self.objects[obj]['channel_id']
            # mask = self._get_mask(obj, object_masks)
            mask = object_masks[idx]
            mask_nan = np.zeros(mask.shape)
            mask_nan[:] = np.NaN
            mask_nan[mask] = 0
            xyz3 = XYZ3 + mask_nan[:,:,np.newaxis]
            counts, is_valids = du.bin_points(xyz3, self.map.shape[0], [], self.resolution)
            counts[0,0] = 0
            self.semantic_map[:,:,map_channel] += counts[:,:,0]
            area = np.sum(self.semantic_map[:,:,map_channel] > 0)
            if np.log10(area+1) - np.log10(self.objects[obj]['area']+1) > 1:
                logging.info(f'Object: {obj}: {area}') 
            self.objects[obj]['area'] = area
    '''

    def get_position_on_map(self):
        map_position = self.current_position - self.origin_xz + self.origin_map*self.resolution
        map_position = map_position / self.resolution
        return map_position

    def convert_xz_to_map_pos(self, xz):
        map_position = xz - self.origin_xz + self.origin_map*self.resolution
        map_position = map_position / self.resolution
        return map_position

    def get_position_on_map_from_aithor_position(self, position_origin):
        xz = np.array([position_origin['z'], position_origin['x']], np.float32)
        return self.convert_xz_to_map_pos(xz)
    
    def get_rotation_on_map(self):
        map_rotation = self.current_rotation
        return map_rotation

    def add_obstacle_in_front_of_agent(self, selem, size_obstacle=10, pad_width=7):
        '''
        salem: dilation structure normally used to dilate the map for path planning
        '''
        # self.loc_on_map_selem
        # loc_on_map = self.loc_on_map.copy()
        # erosion_size = int(np.floor(selem.shape[0]/2))
        size_obstacle = self.loc_on_map_selem.shape[0] #- erosion_size
        # print("size_obstacle", size_obstacle)
        loc_on_map_salem_size = int(np.floor(self.loc_on_map_selem.shape[0]/2))
        # print("loc_on_map_salem_size", loc_on_map_salem_size)
        # loc_on_map_salem_size = int(size_obstacle/2)
        x, y = self.get_position_on_map()
        # print(self.current_rotation)
        if -np.deg2rad(0)==self.current_rotation:
            # plt.figure()
            # plt.imshow(self.get_traversible_map(skimage.morphology.disk(5), 1, True))
            # plt.plot(x, y, 'o')
            # plt.savefig('images/test.png')
            

            ys = [int(y+loc_on_map_salem_size+1), int(y+loc_on_map_salem_size+size_obstacle)]
            y_begin = min(ys)
            y_end = max(ys)
            xs = [int(x-np.floor(size_obstacle/2))-pad_width, int(x+np.floor(size_obstacle/2))+pad_width]
            x_begin = min(xs)
            x_end = max(xs)
        elif -np.deg2rad(90)==self.current_rotation:
            xs = [int(x+loc_on_map_salem_size+1), int(x+loc_on_map_salem_size+size_obstacle)]
            x_begin = min(xs)
            x_end = max(xs)
            ys = [int(y-np.floor(size_obstacle/2))-pad_width, int(y+np.floor(size_obstacle/2))+pad_width]
            y_begin = min(ys)
            y_end = max(ys)
        elif -np.deg2rad(180)==self.current_rotation:
            ys = [int(y-loc_on_map_salem_size-1), int(y-loc_on_map_salem_size-size_obstacle)]
            y_begin = min(ys)
            y_end = max(ys)
            xs = [int(x-np.floor(size_obstacle/2))-pad_width, int(x+np.floor(size_obstacle/2))+pad_width]
            x_begin = min(xs)
            x_end = max(xs)
        elif -np.deg2rad(270)==self.current_rotation:
            xs = [int(x-loc_on_map_salem_size-1), int(x-loc_on_map_salem_size-size_obstacle)]
            x_begin = min(xs)
            x_end = max(xs)
            ys = [int(y-np.floor(size_obstacle/2))-pad_width, int(y+np.floor(size_obstacle/2))+pad_width]
            y_begin = min(ys)
            y_end = max(ys)
        else:
            return 
            st()
            assert(False)

        # st()
        # plt.figure()
        # plt.imshow(np.sum(self.map[:,:,1:], 2)>= 100)
        # plt.plot(x, y, 'o')
        # plt.savefig('images/test.png')
        # traversible_locs = skimage.morphology.binary_dilation(self.loc_on_map, self.loc_on_map_selem) == True 



        # if np.sum(self.added_obstacles[y_begin:y_end, x_begin:x_end])<(size_obstacle*size_obstacle)/4:
        #     dilate_obstacles = True
        # else:
        #     dilate_obstacles = False
        # print("Y len", y_end-y_begin, "X len", x_end-x_begin)
        self.added_obstacles[y_begin:y_end, x_begin:x_end] = False 
        # if dilate_obstacles:
        #     # if there already is an obstacle there, then make the obstacle bigger
        #     print("(mapper) OBSTACLE EXISTS.. Making it bigger.")
        #     self.added_obstacles = skimage.morphology.binary_erosion(self.added_obstacles)



        # self.map[y_begin:y_end, x_begin:x_end,1:] += 100 
        # plt.figure()
        # plt.imshow(np.sum(self.map[:,:,1:], 2)>= 100)
        # plt.plot(x, y, 'o')

        # obstacle = np.sum(self.map[:,:,1:], 2) >= 100
        # traversible = skimage.morphology.binary_dilation(obstacle, selem) != True
        # traversible = np.logical_and(self.added_obstacles, traversible)
        # traversible_locs = skimage.morphology.binary_dilation(self.loc_on_map, self.loc_on_map_selem) == True 
        # traversible = np.logical_or(traversible_locs, traversible)
        # plt.figure()
        # plt.imshow(traversible)
        # # plt.savefig('images/test3.png')
        
        # state_xy = self.get_position_on_map()
        # state_theta = self.get_rotation_on_map()
        # arrow_len = 2.0/self.resolution
        # plt.arrow(state_xy[0], state_xy[1], 
        #             arrow_len*np.cos(state_theta+np.pi/2),
        #             arrow_len*np.sin(state_theta+np.pi/2), 
        #                 color='b', head_width=20)
        # plt.savefig('images/test2.png')
        # st()

    # def dilate_obstacles_around_agent(self, selem, map_percent=0.1):
    #     '''
    #     dilate region around agent
    #     '''
    #     x, y = self.get_position_on_map()
    #     obstacle = np.sum(self.map[:,:,1:], 2) >= 100
    #     # self.map_sz
    #     radius = int(0.05*self.map_sz)
    #     x_start = int(x - radius)
    #     x_end = int(x + radius)
    #     y_start = int(y - radius)
    #     y_end = int(y + radius)
    #     obstacle_ = obstacle[y_start:y_end, x_start:x_end]
    #     plt.figure()
    #     plt.imshow(obstacle_)
    #     plt.savefig('images/test.png')
    #     obstacle_ = skimage.morphology.binary_dilation(obstacle_, selem)
    #     plt.figure()
    #     plt.imshow(obstacle_)
    #     plt.savefig('images/test2.png')
    #     plt.figure()
    #     # obstacle[y_start:y_end, x_start:x_end] = True
    #     plt.imshow(obstacle)
    #     plt.plot(x,y,'o')
    #     plt.savefig('images/test3.png')
    #     st()
    #     obstacle[y_start:y_end, x_start:x_end] = obstacle_
    #     xs, ys = np.where(obstacle)
    #     self.map[xs,ys,1:] += 100

    def get_traversible_map(self, selem, point_count, loc_on_map_traversible):
        # obstacle = np.sum(self.map[:,:,1:], 2) >= 100
        obstacle = np.sum(self.map[:,:,1:-1], 2) >= 100
        # plt.figure()
        # plt.imshow(obstacle)
        # plt.savefig('images/test2.png')
        # if np.sum(self.map[:,:,-1])>0:
        #     st()
        traversible = skimage.morphology.binary_dilation(obstacle, selem) != True

        # also add in obstacles
        traversible = traversible = np.logical_and(self.added_obstacles, traversible)
        # plt.figure()
        # plt.imshow(traversible)
        # plt.savefig('images/test3.png')
        if loc_on_map_traversible:
            # struct = array([[ True,  True,  True],
            #                 [ True,  True,  True],
            #                 [ True,  True,  True]], dtype=bool)
            traversible_locs = skimage.morphology.binary_dilation(self.loc_on_map, self.loc_on_map_selem) == True 
            traversible = np.logical_or(traversible_locs, traversible)

        if self.bounds is not None:
            # limit to scene boundaries
            bounds_x = [self.bounds[0], self.bounds[1]]
            bounds_z = [self.bounds[2], self.bounds[3]]
            len_x_map = int((max(bounds_x) - min(bounds_x))/self.resolution)
            len_z_map = int((max(bounds_z) - min(bounds_z))/self.resolution)
            half_x_map = len_x_map//2
            half_z_map = len_z_map//2
            x_range = [int(self.origin_map[0]-half_x_map), int(self.origin_map[0]+half_x_map)]
            z_range = [int(self.origin_map[1]-half_z_map), int(self.origin_map[1]+half_z_map)]
            # xz_min = np.array([min(bounds_x), min(bounds_z)], np.float32)
            # xz_min_map = self.convert_xz_to_map_pos(xz_min).astype(np.uint8)
            # xz_max = np.array([max(bounds_x), max(bounds_z)], np.float32)
            # xz_max_map = self.convert_xz_to_map_pos(xz_max).astype(np.uint8)
            # offset = int(self.origin_map[0]) + int(self.origin_map[0]/2)
            # zs = [xz_min_map[0], xz_max_map[0]]
            # # offset = int(self.origin_map[0]/2)
            # # xs = [xz_min_map[0]+offset, xz_max_map[0]+offset] 
            # xs = [xz_min_map[1], xz_max_map[1]] 
            # st()
            traversible[:z_range[0], :] = False
            traversible[z_range[1]:, :] = False
            traversible[:,:x_range[0]] = False
            traversible[:,x_range[1]:] = False
        # plt.figure()
        # plt.imshow(traversible)
        # plt.savefig('images/test4.png')
        # if not traversible.any():
        #     st()
        return traversible
    
    def get_explored_map(self, selem, point_count):
        traversible = skimage.morphology.binary_dilation(self.loc_on_map, selem) == True 
        # traversible = self.get_traversible_map(selem, point_count, loc_on_map_traversible=True)
        explored = np.sum(self.map, 2) >= point_count
        explored = np.logical_or(explored, traversible)
        if self.bounds is not None:
            # limit to scene boundaries
            bounds_x = [self.bounds[0], self.bounds[1]]
            bounds_z = [self.bounds[2], self.bounds[3]]
            len_x_map = int((max(bounds_x) - min(bounds_x))/self.resolution)
            len_z_map = int((max(bounds_z) - min(bounds_z))/self.resolution)
            half_x_map = len_x_map//2
            half_z_map = len_z_map//2
            x_range = [int(self.origin_map[0]-half_x_map), int(self.origin_map[0]+half_x_map)]
            z_range = [int(self.origin_map[1]-half_z_map), int(self.origin_map[1]+half_z_map)]
            # xz_min = np.array([min(bounds_x), min(bounds_z)], np.float32)
            # xz_min_map = self.convert_xz_to_map_pos(xz_min).astype(np.uint8)
            # xz_max = np.array([max(bounds_x), max(bounds_z)], np.float32)
            # xz_max_map = self.convert_xz_to_map_pos(xz_max).astype(np.uint8)
            # offset = int(self.origin_map[0]) + int(self.origin_map[0]/2)
            # zs = [xz_min_map[0], xz_max_map[0]]
            # # offset = int(self.origin_map[0]/2)
            # # xs = [xz_min_map[0]+offset, xz_max_map[0]+offset] 
            # xs = [xz_min_map[1], xz_max_map[1]] 
            # st()
            explored[:z_range[0], :] = True
            explored[z_range[1]:, :] = True
            explored[:,:x_range[0]] = True
            explored[:,x_range[1]:] = True
        return explored
    
    def process_pickup(self, uuid):
        # Upon execution of a successful pickup action, clear out the map at
        # the current location, so that traversibility can be updated.
        import pdb; pdb.set_trace()

    def get_object_on_map(self, uuid):
        map_channel = 0
        if uuid in self.objects.keys():
            map_channel = self.objects[uuid]['channel_id']
        object_on_map = self.semantic_map[:,:,map_channel]
        return object_on_map > np.median(object_on_map[object_on_map > 0])
