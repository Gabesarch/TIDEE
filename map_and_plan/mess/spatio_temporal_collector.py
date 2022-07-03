import logging
import os
import imageio
import alphashape
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

def find_unity_executable():
    return str(next(pathlib.Path('../../MCS').glob('MCS-AI2-THOR*.x86_64'), None))

class DataCollector():
    def __init__(self,
                 unity_exe_path,
                 json_read_path, 
                 data_save_path,
                 n_episodes
                 ):
        self.do_dataset = True
        self.unity_exe_path = unity_exe_path
        self.json_read_path = json_read_path
        self.data_save_path = data_save_path
        self.n_episodes = n_episodes

        # Get jsons
        self.spatio_temporal_jsons = glob.glob("{0}/{1}".format(self.json_read_path, "*SpatioTemporal*"))[2:]
        print("Numbers training files: {}".format(len(self.spatio_temporal_jsons)))

        # controller init
        self.controller = mcs.create_controller(self.unity_exe_path, 
                                                debug=False,
                                                enable_noise=False,
                                                depth_maps=True,
                                                object_masks=True)

    def run(self):
        for i in range(self.n_episodes):
            config_json_file_path = self.spatio_temporal_jsons[i]
            scene_config, status = mcs.load_config_json_file(config_json_file_path)

            rgb_seq = []
            depth_seq = []
            sem_seq = []
            objs_seq = []

            output = self.controller.start_scene(scene_config)
            camera_info = {}
            camera_info['camera_aspect_ratio'] = output.camera_aspect_ratio
            camera_info['camera_clipping_planes'] = output.camera_clipping_planes
            camera_info['camera_fov'] = output.camera_field_of_view
            camera_info['camera_height'] = output.camera_height
            while output is not None:
                rgb = np.array(output.image_list[0])
                depth = np.array(output.depth_map_list[0])
                semantic_mask = np.array(output.object_mask_list[0])
                objects = output.object_list
                objs = []
                for obj in objects:
                    obj_dict = {}
                    obj_dict['sem_color'] =np.array([obj.color['r'],obj.color['g'],obj.color['b']])
                    obj_dict['uuid'] = obj.uuid
                    objs.append(obj_dict)

                rgb_seq.append(rgb)
                depth_seq.append(depth)
                sem_seq.append(semantic_mask)
                objs_seq.append(objs)

                '''
                plt.figure()
                ax1 = plt.subplot(1,3,1)
                ax1.imshow(rgb)
                ax2 = plt.subplot(1,3,2)
                ax2.imshow(depth)
                ax3 = plt.subplot(1,3,3)
                ax3.imshow(semantic_mask)
                plt.show()
                '''

                output = self.controller.step("Pass")

            rgb_seq = np.stack(rgb_seq, axis=0)
            depth_seq = np.stack(depth_seq, axis=0)
            sem_seq = np.stack(sem_seq, axis=0)
            if self.do_dataset:
                save_name = "{0}/sample_{1}.npz".format(self.data_save_path, i)
                np.savez(save_name,
                    camera_info = camera_info, # dictionary of camera information
                    json_dict = scene_config, # dictionary of the loaded json file
                    rgb_seq = rgb_seq, # (S,H,W,3)
                    depth_seq = depth_seq, # (S,H,W)
                    sem_seq = sem_seq, # (S,H,W,3)
                    objs_seq = objs_seq # [[obj_dict1, obj_dict2,...], [], ...], obj_dict: "sem_color", "uuid"
                    )


def main():
    unity_exe_path = find_unity_executable()
    data_save_path = "/home/nel/andy/mess/data/spatio_temporal"
    json_read_path = "/home/nel/andy/evaluation3Training"

    if unity_exe_path == 'None':
        print("Unity executable not found in /mcs", file=sys.stderr)
        exit(1)

    # collect data
    data_collector = DataCollector(
        unity_exe_path=unity_exe_path,
        json_read_path=json_read_path,
        data_save_path=data_save_path,
        n_episodes = 10)

    data_collector.run()

if __name__ == "__main__":
    main()
