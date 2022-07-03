import torch, torchvision
from torchvision.utils import save_image
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os
import ntpath
import numpy as np
import cv2
import random
import itertools
import urllib
import json
import PIL.Image as Image

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.data import DatasetMapper
import glob
import detectron2.data.transforms as T
import copy
from detectron2.data import detection_utils
from LossEvalHook import LossEvalHook

import ipdb
st = ipdb.set_trace

test_processed_dir = '/home/sirdome/katefgroup/andy/mess_final/data/detection'
test_files = glob.glob(os.path.join(test_processed_dir, '*.npz'))

def test_dataset_function():
    
    dataset_dicts = []
    print("Loading test dataset...")

    for file in test_files:
        meta = np.load(file, allow_pickle=True)
        record = {}
        record["rgbd_filename"] = meta['rgbd_filename']
        record["image_id"] = int(meta['image_id'])
        record["height"] = int(meta['height'])
        record["width"] = int(meta['width'])
        record["file_name"] = "lala"
        anno_orig = meta['annotations'].tolist()
        anno_kept = []
        for anno in anno_orig:
            if len(anno['segmentation'][0]) < 6:
                continue
            '''
            if anno['category_id'] == 2:
                anno['category_id'] = 1
            elif anno['category_id'] == 1:
                anno['category_id'] = 2
            '''
            anno_kept.append(anno)
        record["annotations"] = anno_kept

        dataset_dicts.append(record)

    print("Data loaded!")
    
    return dataset_dicts 

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    rgbd_filename = dataset_dict.pop("rgbd_filename")
    rgbd_filename = '/home/sirdome/katefgroup/andy/mess_final/data/rgbd/' + str(rgbd_filename).split('/')[-1]
    rgbd = np.load(rgbd_filename)
    rgb_image = (rgbd[:,:,:3] * 255).astype(np.uint8)

    transform_list = []
    
    # Back to RGBD
    rgb_image, transforms = T.apply_transform_gens(transform_list, rgb_image)
    dataset_dict["image"] = torch.as_tensor(rgb_image.transpose(2,0,1).astype("float32"))

    annos = [
        detection_utils.transform_instance_annotations(obj, transforms, rgb_image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]

    instances = detection_utils.annotations_to_instances(annos, rgb_image.shape[:2])
    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

    return dataset_dict
    
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=custom_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                custom_mapper
            )
        ))
        return hooks

# setup trophy detector
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "/home/sirdome/katefgroup/andy/mess_final/checkpoints/model_freeze.pth"
cfg.MODEL.DEVICE='cuda'
cfg.DATASETS.TEST = ("mcs_eval3_places_test",) 
cfg.TEST.EVAL_PERIOD = 1
thing_classes = ['trophy', 'box_opened', 'box_closed']
d = "train"
DatasetCatalog.register("mcs_eval3_places_test", lambda d=d: test_dataset_function())
MetadataCatalog.get("mcs_eval3_places_test").thing_classes = thing_classes

'''
evaluator = COCOEvaluator("mcs_eval3_places_test", cfg, True, output_dir="./output/")
test_loader = build_detection_test_loader(cfg, "mcs_eval3_places_test", mapper=custom_mapper)
inference_on_dataset(detector.model, test_loader, evaluator)
'''
cfg.OUTPUT_DIR = './logs_detector_rgb'

cfg.DATASETS.TRAIN = ("mcs_eval3_places_test",) # add train set name
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.SOLVER.IMS_PER_BATCH = 12
cfg.SOLVER.BASE_LR = 0.00
cfg.SOLVER.MAX_ITER = (
    1
)


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

