import os
from models.aithor_tidee_base import Ai2Thor_Base
from task_base.messup import mess_up, mess_up_from_loaded
from tidee.identify_OOP import IDOOP
from tidee.navigation import Navigation
from task_base.animation_util import Animation
from tidee.memory import Memory
from tidee.placement import Placement
from tidee.object_tracker import ObjectTrack
from task_base.tidy_task import TIDEE_TASK
import ipdb
st = ipdb.set_trace
import torch
import random
import numpy as np
from arguments import args
import traceback
import pickle
import matplotlib.pyplot as plt
import utils.aithor
import sys
from backend import saverloader
from utils.utils import save_dict_as_json, load_json
import os
import wandb
import traceback

'''
Class for running the full tidy task with all modules
'''

class Ai2Thor(Ai2Thor_Base):
    def __init__(self):   

        super(Ai2Thor, self).__init__()

        self.skip_if_exists = True

        ns = {"train":args.n_train_messup, "val":args.n_val_messup, "test":args.n_test_messup}
        self.ns_split = list(np.arange(ns[args.eval_split]))
        self.n = ns[args.eval_split]

        # initialize detector
        args.data_mode = "solq"
        from nets.solq import DDETR
        if args.do_predict_oop:
            num_classes = len(self.include_classes) 
            self.label_to_id = {False:0, True:1} # not out of place = False
            self.id_to_label = {0:'IP', 1:'OOP'}
            num_classes2 = 2 # 0 oop + 1 not oop + 2 no object
            self.score_boxes_name = 'pred1'
            self.score_labels_name = 'pred1'
            self.score_labels_name2 = 'pred2' # which prediction head has the OOP label?
        else:
            num_classes = len(self.include_classes) 
            num_classes2 = None
            self.score_boxes_name = 'pred1' # only one prediction head so same for both
            self.score_labels_name = 'pred1'
        self.ddetr = DDETR(num_classes, load_pretrained=False, num_classes2=num_classes2)
        self.ddetr.cuda()
        print("loading checkpoint for solq...")
        print(f"loading {args.SOLQ_oop_checkpoint}")
        ddetr_path = args.SOLQ_oop_checkpoint 
        _ = saverloader.load_from_path(ddetr_path, self.ddetr, None, strict=(not args.load_strict_false), lr_scheduler=None)
        print("loaded!")
        self.ddetr.eval()

        self.tidee_task = TIDEE_TASK(self.controller, args.eval_split)

        self.metrics_file = os.path.join(args.metrics_dir, f'tidy_task_metrics_{args.eval_split}.txt')

        # self.main()

    def main(self):

        # initialize wandb
        if args.set_name=="test00":
            wandb.init(mode="disabled")
        else:
            wandb.init(project="tidy", name=args.set_name, group=args.group, config=args, dir=args.wandb_directory)
        
        metrics = {}
        if os.path.exists(self.metrics_file) and args.skip_if_exists:
            metrics = load_json(self.metrics_file)
        for i_task in range(self.tidee_task.num_episodes_total):

            print(f"Starting task: {i_task} / {self.tidee_task.num_episodes_total}")

            self.vis = None

            self.tidee_task.set_next_episode_indices()
            episode_name = self.tidee_task.get_episode_name()

            if self.skip_if_exists and (episode_name in metrics.keys()):
                print(f"File already in metrics... skipping...")
                continue

            self.tidee_task.initialize_scene()

            if args.episode_in_try_except:
                try:
                    self.run_tidee(i_task)
                except KeyboardInterrupt:
                    sys.exit(0)
                except Exception as e:
                    tbl = wandb.Table(columns=["Error", "Traceback"])
                    tbl.add_data(str(e), str(traceback.format_exc()))
                    wandb.log({f"Errors/{self.tag}": tbl})
                    print(e)
                    print(traceback.format_exc())
            else:
                self.run_tidee(i_task)

            metrics = self.run_eval(metrics, episode_name, i_task)


    def run_tidee(self, i_task):

        episode_name = self.tidee_task.get_episode_name()
            
        print(f"Starting episode: {episode_name}")
                    
        memory = Memory(
            self.include_classes, 
            self.name_to_id, 
            self.id_to_name, 
            self.W, self.H,
            ddetr=self.ddetr,
            task=self.tidee_task
            )
        placement = Placement(
            self.W, self.H, 
            self.pix_T_camX, 
            self.include_classes, 
            self.name_to_id, 
            self.id_to_name, 
            self.class_to_save_to_id, 
            task=self.tidee_task, 
            special_classes=self.special_classes
            )
        id_oop = IDOOP(
            self.pix_T_camX, 
            self.W, self.H, 
            self.include_classes, 
            self.id_to_name, 
            self.name_to_id,
            ddetr=self.ddetr, 
            task=self.tidee_task, 
            id_to_mapped_id=self.id_to_mapped_id
            )

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

        if args.create_movie and i_task%args.log_every==0:
            print("LOGGING THIS ITERATION")
            vis = Animation(self.W, self.H, navigation=navigation, name_to_id=self.name_to_id)
            print('Height:', self.H, 'Width:', self.W)
        else:
            vis = None
        self.vis = vis

        print("Agent exploring environment...")
        # exploration & mapping 
        obs_dict = navigation.explore_env(
            object_tracker=object_tracker, 
            vis=vis, 
            return_obs_dict=True,
            max_fail=30, 
            )

        if args.do_visual_memex:
            memory.process_scene_graph(obs_dict, episode_name.split('_')[0])

        # picked_up = []
        for search_i in range(args.num_pipeline_attempts):

            if self.tidee_task.is_done():
                break

            print(f"Identifying out of place object #{search_i}")
                
            found_oop_dict = id_oop.exhuastive_oop_search(navigation, vis=vis, object_tracker=object_tracker)
            
            # pick up oop object if found one
            found_one = found_oop_dict['found_oop']
            if found_one:
                print("Found out of place object!")
                success = id_oop.pick_up_oop_obj(navigation, found_oop_dict, vis=vis, object_tracker=object_tracker)

                if not success:
                    # if vis is not None:
                    #     vis.render_movie(args.movie_dir, 0, tag=episode_name)
                    print("Pickup failed... continuing OOP search...")
                    continue

                print(f"Episode: {episode_name}")
                
                if args.do_most_common_memory:
                    print("GETTING MOST COMMON IN MEMORY...")
                    search_classes = memory.get_most_common_objs_in_memory(found_oop_dict, obs_dict, vis=vis)
                    if len(search_classes[0])==0:
                        obj_det = placement.search_for_general_receptacle(navigation, vis=vis, object_tracker=object_tracker)
                        objs_to_search = []
                    else:
                        obj_det, objs_to_search = placement.search_for_related_obj(navigation, obs_dict, search_classes[0], vis=vis, object_tracker=object_tracker)
                    success = placement.place_object_on_related(navigation, obj_det, found_oop_dict, vis=vis, object_tracker=object_tracker, objs_to_search=objs_to_search)
                elif args.do_visual_memex:
                    print("Running RGCN VISUAL retrieval...")
                    inference_dict = memory.run_visual_memex(found_oop_dict, obs_dict, vis=vis)
                    obj_det, objs_to_search = placement.search_for_related_obj(navigation, obs_dict, inference_dict['top_k_classes'][0], vis=vis, object_tracker=object_tracker)
                    success = placement.place_object_on_related(navigation, obj_det, found_oop_dict, vis=vis, object_tracker=object_tracker, objs_to_search=objs_to_search)
                elif args.do_random_receptacle_search:
                    obj_det = placement.search_for_general_receptacle(navigation, vis=vis, object_tracker=object_tracker)
                    success = placement.place_object_on_related(navigation, obj_det, found_oop_dict, vis=vis, object_tracker=object_tracker, objs_to_search=objs_to_search)
                elif args.use_vsn_for_place_proposal:
                    obj_det = placement.vsn_search_object(navigation, found_oop_dict, obs_dict, vis=vis, object_tracker=object_tracker) 
                    success = placement.place_object_on_related(navigation, obj_det, found_oop_dict, vis=vis, object_tracker=object_tracker, objs_to_search=objs_to_search)
                else:
                    assert(False)
            else:
                pass

            # if vis is not None:
            #     vis.render_movie(args.movie_dir, 0, tag=episode_name)

        self.tidee_task.step("Done")



    def run_eval(self, metrics, episode_name, i_task, save=True):
        # save out evaluation images
        self.tidee_task.render_episode_images()

        metrics_instance = self.tidee_task.get_metrics()
        print(metrics_instance)
        metrics[episode_name] = metrics_instance

        if save:
            
            aggregrated_metrics = self.tidee_task.aggregate_metrics(metrics)

            print('\n\n---------- File 1 ---------------')
            to_log = []  
            to_log.append('-'*40 + '-'*40)
            # list_of_files = files #list(metrics.keys())
            # to_log.append(f'Files: {str(list_of_files)}')
            to_log.append(f'Split: {args.eval_split}')
            to_log.append(f'Number of files: {len(list(metrics.keys()))}')
            for f_n in aggregrated_metrics.keys(): #keys_include:
                to_log.append(f'{f_n}: {aggregrated_metrics[f_n]}') 
            to_log.append('-'*40 + '-'*40)

            os.makedirs(args.metrics_dir, exist_ok=True)
            path = os.path.join(args.metrics_dir, f'tidy_task_summary_{args.eval_split}.txt')
            with open(path, "w") as fobj:
                for x in to_log:
                    fobj.write(x + "\n")

            save_dict_as_json(metrics, self.metrics_file)

            aggregrated_metrics["num episodes"] = i_task
            tbl = wandb.Table(columns=list(aggregrated_metrics.keys()))
            tbl.add_data(*list(aggregrated_metrics.values()))
            wandb.log({f"Metrics_summary/Summary": tbl, 'step':i_task})                

            cols = ["file"]+list(metrics_instance.keys())
            # cols.remove('pred_actions')
            tbl = wandb.Table(columns=cols)
            for f_k in metrics.keys():
                to_add_tbl = [f_k]
                for k in list(metrics[f_k].keys()):
                    if k=="pred_actions":
                        continue
                    to_add_tbl.append(metrics[f_k][k])
                # list_values = [f_k] + list(metrics[f_k].values())
                tbl.add_data(*to_add_tbl)
            wandb.log({f"Metrics_summary/Metrics": tbl, 'step':i_task})

        if self.vis is not None:
            # render movie of agent
            self.vis.render_movie(args.movie_dir, 0, tag=episode_name)

        return metrics
                
            


                


if __name__ == '__main__':
    Ai2Thor()
        
    


