import logging
import os
import imageio
import numpy as np
import machine_common_sense as mcs
import ipdb
import pathlib
import sys
import random
import json
import matplotlib.pyplot as plt
import tkinter
import matplotlib
import matplotlib.gridspec as gridspec
matplotlib.use('TkAgg')
st = ipdb.set_trace
from detectron2.structures import BoxMode
import cv2

def find_unity_executable():
    return str(next(pathlib.Path('../../MCS').glob('MCS-AI2-THOR*.x86_64'), None))

class DataCollector():
    def __init__(self,unity_exe_path, 
                      n_episodes, 
                      materials_dict,
                      distraction_salient_dict,
                      detection_save_path,
                      rgbd_save_path):
        self.do_visualize = False
        self.do_dataset = True

        self.unity_exe_path = unity_exe_path
        self.n_episodes = n_episodes
        self.materials_dict = materials_dict
        self.distraction_salient_dict = distraction_salient_dict

        # data saving
        self.detection_save_path = detection_save_path
        self.rgbd_save_path = rgbd_save_path
        self.sample_counter = 0

        # controller init
        self.controller = mcs.create_controller(self.unity_exe_path, 
                                                debug=False,
                                                enable_noise=False,
                                                depth_maps=True,
                                                object_masks=True,
                                                history_enabled=False)

        # object hyperparams initialization
        self.random_scale_ratio = 0.9
        self.spawn_trophy = 0.7 # 0.7
        self.inside_box_ratio = 0.3 # 0.3
        self.box_opened_ratio = 0.3 # 0.3
        self.num_trophy = 1
        self.min_num_boxes = 2
        self.max_num_boxes = 5
        self.min_distractions = 0
        self.max_distractions = 8
        self.angle_bins = 12
        self.angle_step = 360 / self.angle_bins

        # opeable box lists initialization
        self.box_list = ["chest_1", "suitcase_1", "chest_2"]
        self.box_material_dict = {
            "chest_1": ["metal", "plastic", "wood"], # sturdy box
            "chest_2": ["metal", "plastic", "wood"], # treasure chest
            "suitcase_1": ["metal", "plastic"], # suitcase
        }

        self.scale_minmax_dict = {"chest_1": [0.4, 0.75], 
                                  "chest_2": [0.7, 2], 
                                  "suitcase_1": [0.7, 2]}

        self.box_rotation_dict = {"chest_1":180,
                                  "chest_2": 90,
                                  "suitcase_1":180}

        # distraction object lists initialization
        self.distraction_list = list(self.distraction_salient_dict.keys())

        # ceiling/wall/floor material lists initialization
        self.wall_material_list = self.materials_dict['wall'] + self.materials_dict['ceramic'] + self.materials_dict['fabric'] + self.materials_dict['wood']
        self.ceiling_material_list = self.materials_dict['wall'] + self.materials_dict['ceramic'] + self.materials_dict['fabric'] + self.materials_dict['wood']
        self.floor_material_list = self.materials_dict['ceramic'] + self.materials_dict['fabric'] + self.materials_dict['wood']

    def random_scene_config(self, eps):
        '''
        Random initialize the scene with trophies, openable boxes
        Inputs:
            None
        Outputs:
            scene_config: A configuration dict, similar to trophy_example.json
                          But it does not have the "performer start" field, which will be set outside
            inside_box: Boolean variable, whether the trophy lies inside the box
        '''
        inside_box = False
        scene_config = {}
        scene_config['name'] = "detector_training_scene_{}".format(eps)
        scene_config['version'] = 2

        # Random select ceiling, wall, and floor materials
        scene_config['ceilingMaterial'] = random.choice(self.ceiling_material_list)
        scene_config['floorMaterial'] = random.choice(self.floor_material_list)
        scene_config['wallMaterial'] = random.choice(self.wall_material_list)

        scene_config['objects'] = []
        # Randomly add trophies
        for i in range(self.num_trophy):
            trophy = {}
            # since just one trophy now
            trophy['id'] = "trophy_{}".format(i)
            trophy['type'] = "trophy"
            trophy['shows'] = []
            shows = {}
            shows['stepBegin'] = 0
            # Should we change trophy scale?
            # TODO: We should also add cases where it's inside a container?
            if np.random.rand() < self.inside_box_ratio:
                inside_box = True
                trophy_x = np.random.uniform(-4.5,4.5)
                trophy_y = np.random.uniform(0.05, 0.12)
                trophy_z = np.random.uniform(-4.5,4.5)
                shows['position'] = {'x': trophy_x, 'y':trophy_y, 'z': trophy_z}
                trophy_roty = float(np.random.randint(360))
                shows['rotation'] = {'x': random.choice([0,90]), 'y': trophy_roty, 'z':0}
                trophy['shows'].append(shows)
                if np.random.rand() < self.spawn_trophy:
                    scene_config['objects'].append(trophy)
                # Add box
                box = {}
                box['id'] = "box_{}".format(i)
                box['type'] = random.choice(self.box_list)
                box['salientMaterials'] = [random.choice(self.box_material_dict[box['type']])]
                box['materials'] = [random.choice(self.materials_dict[box['salientMaterials'][0]])]
                box['openable'] = True
                box['opened'] = True
                box['id'] = box['id'] + '_opened'
                box['shows'] = []
                shows = {}
                shows['stepBegin'] = 0
                rot_angle = (trophy_roty - 90) % 360
                if np.random.rand() < self.random_scale_ratio:
                    if np.random.rand() < 0.7:
                        # same scale for all dimensions
                        scale = np.random.uniform(self.scale_minmax_dict[box['type']][0], self.scale_minmax_dict[box['type']][1])
                        shows['scale'] = {'x': scale, 'y': scale, 'z': scale}
                    else:
                        # different scale for each dimension
                        shows['scale'] = {'x': np.random.uniform(self.scale_minmax_dict[box['type']][0], self.scale_minmax_dict[box['type']][1]), 
                                          'y': np.random.uniform(self.scale_minmax_dict[box['type']][0], self.scale_minmax_dict[box['type']][1]), 
                                          'z': np.random.uniform(self.scale_minmax_dict[box['type']][0], self.scale_minmax_dict[box['type']][1])}
                box_x = trophy_x + 0.2 * np.cos(rot_angle/180*np.pi)
                box_z = trophy_z - 0.2 * np.sin(rot_angle/180*np.pi)
                box_roty = (trophy_roty + self.box_rotation_dict[box['type']]) % 360
                shows['position'] = {'x': box_x, 'y': 0, 'z': box_z}
                shows['rotation'] = {'x': 0, 'y': box_roty, 'z': 0}
                box['shows'].append(shows)
                scene_config['objects'].append(box)
            else:
                shows['position'] = {'x': np.random.uniform(-4.9,4.9), 'y':0, 'z': np.random.uniform(-4.9,4.9)}
                shows['rotation'] = {'x': random.choice([0,90]), 'y':float(np.random.randint(360)), 'z':0}
                trophy['shows'].append(shows)
                if np.random.rand() < self.spawn_trophy:
                    scene_config['objects'].append(trophy)

        # randomly add containers
        num_boxes = np.random.randint(self.min_num_boxes, self.max_num_boxes)
        for i in range(num_boxes):
            box = {}
            box['id'] = "box_{}".format(i+1)
            box['type'] = random.choice(self.box_list)
            box['salientMaterials'] = [random.choice(self.box_material_dict[box['type']])]
            box['materials'] = [random.choice(self.materials_dict[box['salientMaterials'][0]])]
            
            if np.random.rand() < self.box_opened_ratio:
                box['openable'] = True
                box['opened'] = True
                box['id'] = box['id'] + '_opened'
            
            box['shows'] = []
            shows = {}
            shows['stepBegin'] = 0
            shows['position'] = {'x': np.random.uniform(-4.9,4.9), 'y':0, 'z': np.random.uniform(-4.9,4.9)}
            shows['rotation'] = {'x': 0, 'y':float(np.random.randint(360)), 'z':0}
            if np.random.rand() < self.random_scale_ratio:
                if np.random.rand() < 0.7:
                    # same scale for all dimensions
                    scale = np.random.uniform(self.scale_minmax_dict[box['type']][0]-0.4, self.scale_minmax_dict[box['type']][1])
                    shows['scale'] = {'x': scale, 'y': scale, 'z': scale}
                else:
                    # different scale for each dimension
                    shows['scale'] = {'x': np.random.uniform(self.scale_minmax_dict[box['type']][0]-0.4, self.scale_minmax_dict[box['type']][1]), 
                                      'y': np.random.uniform(self.scale_minmax_dict[box['type']][0]-0.4, self.scale_minmax_dict[box['type']][1]), 
                                      'z': np.random.uniform(self.scale_minmax_dict[box['type']][0]-0.4, self.scale_minmax_dict[box['type']][1])}
            box['shows'].append(shows)
            scene_config['objects'].append(box)
        
        # randomly add distractions
        num_distractions = np.random.randint(self.min_distractions, self.max_distractions)
        for i in range(num_distractions):
            obj = {}
            obj['id'] = "obj_{}".format(i)
            obj['type'] = random.choice(self.distraction_list)
            if type(self.distraction_salient_dict[obj['type']]) is list:
                obj['salientMaterials'] = [random.choice(self.distraction_salient_dict[obj['type']])]
                obj['materials'] = [random.choice(self.materials_dict[obj['salientMaterials'][0]])]
            obj['shows'] = []
            shows = {}
            shows['stepBegin'] = 0
            shows['position'] = {'x': np.random.uniform(-4.9,4.9), 'y':0, 'z': np.random.uniform(-4.9,4.9)}
            shows['rotation'] = {'x': 0, 'y':float(np.random.randint(360)), 'z':0}
            if np.random.rand() < self.random_scale_ratio:
                shows['scale'] = {'x': np.random.uniform(0.2,1.2), 'y': np.random.uniform(0.2,2.2), 'z': np.random.uniform(0.2,2.2)}
            obj['shows'].append(shows)
            scene_config['objects'].append(obj)

        return scene_config, inside_box

    def get_bbox_and_seg(self, mask):
        inds = np.where(mask == 1)

        # bbox
        ymin, ymax, xmin, xmax = inds[0].min(), inds[0].max(), inds[1].min(), inds[1].max()
        bbox = [xmin, ymin, xmax, ymax]

        # seg contour
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        seg = [contours[0].reshape(-1).tolist()]

        return bbox, seg


    def run(self):
        '''
        The runner function that runs the data collection process
        Inputs:
            None
        Outputs:
            None. Files are saved
        '''
        plt.figure(figsize=(12,5))
        gs1 = gridspec.GridSpec(5,12)
        gs1.update(wspace=0.05, hspace=0.05)
        self.sample_counter = 57934
        self.trophy_counter = 31866
        self.box_opened_counter = 30785
        self.box_closed_counter = 30789

        for eps in range(4946, self.n_episodes):
            print("Episode {0} / {1}, sample = {2}, trophy = {3}, box_opened = {4}, box_closed = {5}".format(eps, self.n_episodes, self.sample_counter, self.trophy_counter, self.box_opened_counter, self.box_closed_counter))
            scene_config, inside_box = self.random_scene_config(eps)
            for bin_id in range(self.angle_bins):
                # randomly select angle from bin
                angle_low = bin_id * self.angle_step
                angle_high = (bin_id+1) * self.angle_step
                angle = float(np.random.randint(angle_low, angle_high))
                tan_a = np.tan(angle/180*np.pi)
                if np.abs(tan_a) < 1e-10:
                    tan_a += 1e-5

                # get trophy position
                tx = scene_config['objects'][0]['shows'][0]['position']['x']
                tz = scene_config['objects'][0]['shows'][0]['position']['z']

                # get max radius given this angle
                r_min = 1
                r1 = np.sqrt(((5 - tx)/tan_a)**2 + (5-tx)**2)
                r2 = np.sqrt(((-5 - tx)/tan_a)**2 + (-5-tx)**2)
                r3 = np.sqrt((5-tz)**2 + ((5-tz)*tan_a)**2)
                r4 = np.sqrt((-5-tz)**2 + ((-5-tz)*tan_a)**2)
                r_max = min(r1, r2, r3, r4)
                if inside_box:
                    r_max = min(r_max, 1)
                    r_min = min(r_max, 0.4)
                if np.abs(r_min - r_max) < 1e-8:
                    r_min = r_max / 2

                # get dist from trophy
                dist = np.random.uniform(r_min, r_max)
                agent_x = tx + dist * np.sin(angle/180*np.pi)
                agent_z = tz + dist * np.cos(angle/180*np.pi)

                # random offset to rotation so trophy is not always trivially centered
                angle = (180+angle+np.random.randint(-5,6))%360

                # Set agent starting point
                scene_config['performerStart'] = {}
                scene_config['performerStart']['position'] = {'x': agent_x, 'z': agent_z}
                head_tilt = 0
                if inside_box:
                    heat_tilt = np.random.randint(0,7) * 10
                else:
                    head_tilt = np.random.randint(0,5) * 10
                scene_config['performerStart']['rotation'] = {'x': head_tilt, 'y': angle}

                #scene_config, status = mcs.load_config_json_file("../../trophy_example_simple.json")

                output = self.controller.start_scene(scene_config)
                rgb = np.array(output.image_list[0])
                depth = np.array(output.depth_map_list[0])
                rgb_normed = rgb.astype(np.float32) / 255.0
                depth_normed = depth.reshape(400,600,1).astype(np.float32) / 15.0
                rgbd = np.concatenate([rgb_normed, depth_normed], axis=2)
                semantic_mask = np.array(output.object_mask_list[0])
                objects = output.object_list
                trophy_mask = np.zeros(semantic_mask.shape[:2])
                all_box_mask = np.zeros(semantic_mask.shape[:2])
                box_masks = []
                box_opened = []
                for obj in objects:
                    if 'trophy' in obj.uuid:
                        color = np.array([obj.color['r'], obj.color['g'], obj.color['b']]).reshape(1,1,3)
                        trophy_mask += (semantic_mask == color).sum(2)==3
                    elif 'box' in obj.uuid:
                        color = np.array([obj.color['r'], obj.color['g'], obj.color['b']]).reshape(1,1,3)
                        box_mask = (semantic_mask == color).sum(2)==3
                        box_masks.append(box_mask)
                        if 'opened' in obj.uuid:
                            box_opened.append(True)
                        else:
                            box_opened.append(False)
                        all_box_mask += box_mask

                # Visualize
                if self.do_visualize:
                    ax1 = plt.subplot(gs1[bin_id])
                    ax1.imshow(rgb)
                    ax2 = plt.subplot(gs1[12+bin_id])
                    ax2.imshow(depth)
                    ax3 = plt.subplot(gs1[24+bin_id])
                    ax3.imshow(semantic_mask)
                    ax4 = plt.subplot(gs1[36+bin_id])
                    ax4.imshow(trophy_mask)
                    ax5 = plt.subplot(gs1[48+bin_id])
                    ax5.imshow(all_box_mask)

                if self.do_dataset:
                    # save rgb image, depth
                    '''
                    rgb_name = os.path.join(self.rgb_save_path, "{}.png".format(self.trophy_sample_counter))
                    depth_name = os.path.join(self.depth_save_path, "{}.npy".format(self.trophy_sample_counter))
                    imageio.imwrite(rgb_name, rgb)
                    np.save(depth_name, depth)
                    '''
                    # save data for trophy
                    objs = []
                    if trophy_mask.sum() > 0:
                        bbox, seg = self.get_bbox_and_seg(trophy_mask)
                        if len(seg[0]) >= 6:
                            trophy = {
                                "bbox": bbox,
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "segmentation": seg,
                                "category_id": 0,
                                "is_crowd": 0
                            }
                            objs.append(trophy)
                            self.trophy_counter += 1
                    '''
                    data_filename = os.path.join(self.trophy_save_path, "e{0}_v{1}.npz".format(eps, bin_id))
                    rgbd_filename = os.path.join(self.rgbd_save_path, "e{0}_v{1}.npy".format(eps, bin_id))
                    np.save(rgbd_filename, rgbd)
                    np.savez(data_filename,
                        rgbd_filename = rgbd_filename,
                        image_id = self.trophy_sample_counter,
                        height = 400,
                        width = 600,
                        annotations = objs)
                    self.trophy_sample_counter += 1
                    '''

                    # save data for boxes
                    for box_id, box_mask in enumerate(box_masks):
                        if box_mask.sum() == 0:
                            continue
                        bbox, seg = self.get_bbox_and_seg(box_mask)
                        if len(seg[0]) < 6:
                            continue
                        box_cur = {
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "segmentation": seg,
                            "category_id": 1 if not box_opened[box_id] else 2,
                            "is_crowd": 0
                        }
                        if box_opened[box_id]:
                            self.box_opened_counter += 1
                        else:
                            self.box_closed_counter += 1
                        objs.append(box_cur)

                    '''
                    for obj in objs:
                        print(obj['category_id'])
                    print("")
                    '''

                    if len(objs) > 0:
                        data_filename = os.path.join(self.detection_save_path, "e{0}_v{1}.npz".format(eps, bin_id))
                        rgbd_filename = os.path.join(self.rgbd_save_path, "e{0}_v{1}.npy".format(eps, bin_id))
                        np.save(rgbd_filename, rgbd)
                        np.savez(data_filename,
                            rgbd_filename = rgbd_filename,
                            image_id = self.sample_counter,
                            height = 400,
                            width = 600,
                            annotations = objs)
                        self.sample_counter += 1

                    self.controller.end_scene(None)

            if self.do_visualize:
                plt.savefig("episode_collage.png")
                plt.show()


def main():
    unity_exe_path = find_unity_executable()
    detection_save_path = "../data/detection"
    rgbd_save_path = "../data/rgbd"

    if unity_exe_path == 'None':
        print("Unity executable not found in /mcs", file=sys.stderr)
        exit(1)

    # Salient material --> material dict
    materials = ["block_blank", "block_stuff", "cardboard", "ceramic", "fabric", "metal", "plastic", "rubber", "sofa_1", "sofa_chair_1", "sofa_2", "wood", "wall"]
    materials_dict = {}
    for material in materials:
        with open("../txts/{}.txt".format(material), 'rb') as f:
            lines = f.readlines()
            lines = [line.decode("utf-8").rstrip('\n').rstrip('"').lstrip('"') for line in lines]
        materials_dict[material] = lines

    # Disctraction object --> material dict
    with open('../txts/distractions.json') as f:
        distraction_salient_dict = json.load(f)

    # collect data
    data_collector = DataCollector(
        unity_exe_path=unity_exe_path,
        n_episodes=10000,
        materials_dict=materials_dict,
        distraction_salient_dict=distraction_salient_dict,
        detection_save_path=detection_save_path,
        rgbd_save_path=rgbd_save_path)

    data_collector.run()

if __name__ == "__main__":
    main()