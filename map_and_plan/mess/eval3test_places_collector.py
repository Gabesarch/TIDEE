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
import glob
import matplotlib.gridspec as gridspec
matplotlib.use('TkAgg')
st = ipdb.set_trace
from detectron2.structures import BoxMode
import cv2
import urllib.request, json

def find_unity_executable():
    return str(next(pathlib.Path('../../MCS').glob('MCS-AI2-THOR*.x86_64'), None))

class DataCollector():
    def __init__(self,
                 unity_exe_path,
                 json_read_path, 
                 rgbd_save_path,
                 detection_save_path,
                 n_episodes
                 ):
        self.do_dataset = True
        self.unity_exe_path = unity_exe_path
        self.json_read_path = json_read_path
        self.rgbd_save_path = rgbd_save_path
        self.detection_save_path = detection_save_path
        self.n_episodes = n_episodes

        # Get jsons
        self.jsons = glob.glob("{0}/{1}".format(self.json_read_path, "*.json"))
        #random.shuffle(self.spatio_temporal_jsons)
        print("Number of files: {}".format(len(self.jsons)))

        # controller init
        self.controller = mcs.create_controller(self.unity_exe_path, 
                                                config_file_path='../level1.config')

    def get_bbox_and_seg(self, mask):
        inds = np.where(mask == 1)

        # bbox
        ymin, ymax, xmin, xmax = inds[0].min(), inds[0].max(), inds[1].min(), inds[1].max()
        bbox = [xmin, ymin, xmax, ymax]

        # seg contour
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        seg = [contours[0].reshape(-1).tolist()]

        return bbox, seg

    def save_sample(self, output, rgbd_name, detection_name):
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
            color = np.array([obj.color['r'], obj.color['g'], obj.color['b']]).reshape(1,1,3)
            mask = (semantic_mask == color).sum(2) == 3
            if mask.sum() < 20:
                # ignore small or empty masks
                continue
            if 'trophy' in obj.uuid:
                trophy_mask += mask
            elif 'box' in obj.uuid:
                box_masks.append(mask)
                if 'opened' in obj.uuid:
                    box_opened.append(True)
                else:
                    box_opened.append(False)
                all_box_mask += mask

        # Detectron2-style obj annotations
        objs = []
        if trophy_mask.sum() > 0:
            bbox, seg = self.get_bbox_and_seg(trophy_mask)
            trophy = {
                "bbox": bbox, 
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": seg,
                "category_id": 0,
                "is_crowd": 0
            }
            objs.append(trophy)
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
                "category_id": 1 if box_opened[box_id] else 2,
                "is_crowd": 0
            }
            objs.append(box_cur)

        np.save(rgbd_name, rgbd)
        np.savez(detection_name,
            rgbd_filename = rgbd_name,
            image_id = self.sample_counter,
            height = 400,
            width = 600,
            annotations = objs)
        self.sample_counter += 1

        return

    def run(self):
        self.sample_counter = 0
        for i in range(0,self.n_episodes):
            config_json_file_path = self.jsons[i]
            scene_config, status = mcs.load_scene_json_file(config_json_file_path)
            
            name = config_json_file_path.split('/')[-1].split('.')[0]
            
            eval3_address = "https://evaluation-images.s3.amazonaws.com/eval-3-scenes"
            link_address = os.path.join(eval3_address, "{}_debug.json".format(name))
            with urllib.request.urlopen(link_address) as url:
                scene_config = json.loads(url.read().decode())
                
            new_obj_list = []
            for obj in scene_config['objects']:
                if obj['isTarget']:
                    obj['id'] = obj['id'] + '_trophy'
                elif obj['isContainer']:
                    obj['id'] = obj['id'] + '_box'
                new_obj_list.append(obj)
            scene_config['objects'] = new_obj_list

            output = self.controller.start_scene(scene_config)
            
            v = 0
            while v < 12:
                # (1) Look down and rotate, and collect
                output = self.controller.step('LookDown')
                for _ in range(3):
                    output = self.controller.step('RotateLeft')
                rgbd_name = os.path.join(self.rgbd_save_path, "t{0}_v{1}.npy".format(i, v))
                detection_name = os.path.join(self.detection_save_path, "t{0}_v{1}.npz".format(i, v))
                self.save_sample(output, rgbd_name, detection_name)
                v += 1

                # (2) Look up and rotate, and collect
                output = self.controller.step('LookUp')
                for _ in range(3):
                    output = self.controller.step('RotateLeft')
                rgbd_name = os.path.join(self.rgbd_save_path, "t{0}_v{1}.npy".format(i, v))
                detection_name = os.path.join(self.detection_save_path, "t{0}_v{1}.npz".format(i, v))
                self.save_sample(output, rgbd_name, detection_name)
                v += 1


def main():
    unity_exe_path = find_unity_executable()
    rgbd_save_path = "/home/sirdome/katefgroup/andy/mess_final/data/rgbd_test"
    detection_save_path = "/home/sirdome/katefgroup/andy/mess_final/data/detection_test"
    json_read_path = "/home/sirdome/katefgroup/andy/eval3/eval_3_interactive_container"

    if unity_exe_path == 'None':
        print("Unity executable not found in /mcs", file=sys.stderr)
        exit(1)

    # collect data
    data_collector = DataCollector(
        unity_exe_path=unity_exe_path,
        json_read_path=json_read_path,
        rgbd_save_path=rgbd_save_path,
        detection_save_path=detection_save_path,
        n_episodes = 100)

    data_collector.run()

if __name__ == "__main__":
    main()

