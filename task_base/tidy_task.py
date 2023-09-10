
import numpy as np
import os
# import hyperparams as hyp
from arguments import args
import ipdb
st = ipdb.set_trace
from .messup import mess_up_from_loaded
import utils.aithor
import pickle
# from allenact_plugins.ithor_plugin.ithor_util import include_object_data
import matplotlib.pyplot as plt
import json
import copy

class TIDEE_TASK():

    def __init__(
        self, 
        controller,
        split,
        init_agent_rotation_random=False, 
        diplay_every=10,
        max_episode_steps=None,
        ):  
        '''
        controller: Ai2Thor controller 
        split: train, val, test
        init_agent_rotation_random: Randomize rotation angle at start of episode
        diplay_every: how often to output logging
        '''

        self.init_agent_rotation_random = init_agent_rotation_random
        self.controller = controller
        self.step_count = 0
        self.api_fails = 0
        self.split = split
        mapnames_split_dict = self.get_scene_split()
        self.mapnames_split = mapnames_split_dict[split]
        self.W, self.H = args.W, args.H
        if max_episode_steps is None:
            self.max_episode_steps = args.max_traj_steps
        else:
            self.max_episode_steps = max_episode_steps
        self.max_api_fails = args.max_api_fails

        ns = {"train":args.n_train_messup, "val":args.n_val_messup, "test":args.n_test_messup}
        self.ns_split = list(np.arange(ns[split]))

        self.num_episodes_total = len(self.mapnames_split) * len(self.ns_split)

        self.next_ep_called = False

        self.diplay_every = diplay_every # display every ten actions

        self.init_episodes()

        with open('./data/object_receptacle_counts.json', 'r') as fp:
            self.object_receptacle_counts = json.load(fp)



    def init_episodes(self):

        self.mapnames_iter = iter(self.mapnames_split)
        self.ns_iter = iter(self.ns_split)

        self.mapname_current = next(self.mapnames_iter)

        self.n = -1

        # self.start_next_episode()

    def get_episode_name(self):
        return f"{self.mapname_current}_{self.n}"

    def is_done(self):
        return (self.step_count>=self.max_episode_steps or self.done_called or self.api_fails>=self.max_api_fails)

    def step(self, action, obj_relative_coord=None): #object_category=None):

        if not self.next_ep_called:
            print("ERROR: Must call start_next_episode() OR set_next_episode_indices() then initialize_scene() after init before calling actions")
            assert(False)

        if self.is_done():
            if action!="Done":
                print("Warning: action called when episode is done!")
            return

        if obj_relative_coord is not None:
            assert(len(obj_relative_coord)==2) # must only have x,y
            assert(not (max(obj_relative_coord)>1.0 or min(obj_relative_coord)<0.0)) # relative coords
            x = obj_relative_coord[0]
            y = obj_relative_coord[1]
        else:
            assert(not (action in ["PickupObject", "PutObject"])) # need object coordinates

        if self.step_count%self.diplay_every==0:
            print(f"[Episode {self.get_episode_name()}] step {self.step_count}: {action}")

        if action=="MoveAhead":
            self.move_ahead()
        elif action=="MoveBack":
            self.move_back()
        elif action=="MoveRight":
            self.move_right()
        elif action=="MoveLeft":
            self.move_left()
        elif action=="RotateLeft":
            self.rotate_left()
        elif action=="RotateRight":
            self.rotate_right()
        elif action=="Stand":
            self.stand()
        elif action=="Crouch":
            self.crouch()
        elif action=="LookUp":
            self.look_up()
        elif action=="LookDown":
            self.look_down()
        elif action=="PickupObject":
            self.pickup_object(x,y)
        elif action=="PutObject":
            self.put_object(x,y)
        elif action=="DropObject":
            self.pickup_object()
        elif action=="OpenObject":
            self.open_object(x,y)
        elif action=="CloseObject":
            self.close_object(x,y)
        elif action=="Done":
            self.done()
        else:
            print(action)
            assert(False) # what action is this?
        self.step_count += 1
        success = self.action_success()

        if not success:
            self.api_fails += 1

        return success

    def action_success(self):
        if self.is_done():
            return False
        return self.controller.last_event.metadata["lastActionSuccess"]

    def get_observations(self):
        obs = {}
        obs["rgb"] =  self.controller.last_event.frame
        obs["depth"] = self.controller.last_event.depth_frame
        return obs

    def reset_episode_mapname(self, mapname):
        self.controller.reset(scene=mapname)

        event = self.controller.step(action="GetReachablePositions")
        nav_pts = event.metadata["actionReturn"]
        nav_pts = np.array([list(d.values()) for d in nav_pts])

        if len(nav_pts)<5: # something wrong happened here, reset
            self.controller.reset(scene=mapname)
            mess_up_from_loaded(self.controller, object_messup_meta)

        event = self.controller.step(action="GetReachablePositions")
        nav_pts = event.metadata["actionReturn"]
        nav_pts = np.array([list(d.values()) for d in nav_pts])

        self.move_agent_to_scene_center(nav_pts)

        self.next_ep_called = True
        self.step_count = 0

    def skip_next_episode(self):

        self.next_ep_called = False
        self.step_count = 0
        self.api_fails = 0
        
        n = next(self.ns_iter, None)

        if n is None:
            self.mapname_current = next(self.mapnames_iter, None)

            if self.mapname_current is None:
                self.finish()
                return 

            self.ns_iter = iter(self.ns_split)

            n = next(self.ns_iter)

        self.n = n

        print(f"Skipping episode {self.get_episode_name()}")

        self.done_called = False
        self.picked_up = []

    def set_next_episode_indices(self):

        self.next_ep_called = False
        self.step_count = 0
        self.api_fails = 0
        
        n = next(self.ns_iter, None)

        if n is None:
            self.mapname_current = next(self.mapnames_iter, None)

            if self.mapname_current is None:
                self.finish()
                return 

            self.ns_iter = iter(self.ns_split)

            n = next(self.ns_iter)

        self.n = n

        self.done_called = False
        self.picked_up = []

    def initialize_scene(self):

        self.next_ep_called = True

        print(f"Starting episode {self.get_episode_name()}")

        self.controller.reset(scene=self.mapname_current)

        objects_original = self.controller.last_event.metadata['objects']
        self.objects_original = objects_original

        # mess up scene
        messup_fname = os.path.join(args.mess_up_dir, self.mapname_current, f'{self.n}.p')
        with open(messup_fname, 'rb') as f:
            load_dict = pickle.load(f)
        object_dict = load_dict['object_dict'] 
        oop_IDs = load_dict['oop_IDs'] 
        self.oop_IDs = oop_IDs
        object_messup_meta = load_dict['objects_messup'] 

        if args.save_object_images:
            self.image_dict_original = utils.aithor.get_images_of_objects(self.controller, objects_original, oop_IDs, self.H,self.W)
        else:
            self.image_dict_original = None

        mess_up_from_loaded(self.controller, object_messup_meta)

        objects_messup = self.controller.last_event.metadata['objects']
        self.objects_messup = objects_messup

        if args.save_object_images:
            self.image_dict_messup = utils.aithor.get_images_of_objects(self.controller, objects_messup, oop_IDs, self.H, self.W)
        else:
            self.image_dict_messup = None

        event = self.controller.step(action="GetReachablePositions")
        nav_pts = event.metadata["actionReturn"]
        nav_pts = np.array([list(d.values()) for d in nav_pts])

        if len(nav_pts)<5: # something wrong happened here, reset
            self.controller.reset(scene=self.mapname_current)
            mess_up_from_loaded(self.controller, object_messup_meta)

        event = self.controller.step(action="GetReachablePositions")
        nav_pts = event.metadata["actionReturn"]
        nav_pts = np.array([list(d.values()) for d in nav_pts])

        self.move_agent_to_scene_center(nav_pts)

    def start_next_episode(self):
        '''
        Equivalent to set_next_episode_indices() then initialize_scene() in a single step
        '''

        self.next_ep_called = True
        self.step_count = 0
        self.api_fails = 0
        
        n = next(self.ns_iter, None)

        if n is None:
            self.mapname_current = next(self.mapnames_iter, None)

            if self.mapname_current is None:
                self.finish()
                return 

            self.ns_iter = iter(self.ns_split)

            n = next(self.ns_iter)

        self.n = n

        self.done_called = False
        self.picked_up = []

        print(f"Starting episode {self.get_episode_name()}")

        self.controller.reset(scene=self.mapname_current)

        objects_original = self.controller.last_event.metadata['objects']
        self.objects_original = objects_original

        # mess up scene
        messup_fname = os.path.join(args.mess_up_dir, self.mapname_current, f'{n}.p')
        with open(messup_fname, 'rb') as f:
            load_dict = pickle.load(f)
        object_dict = load_dict['object_dict'] 
        oop_IDs = load_dict['oop_IDs'] 
        self.oop_IDs = oop_IDs
        object_messup_meta = load_dict['objects_messup'] 

        if args.save_object_images:
            self.image_dict_original = utils.aithor.get_images_of_objects(self.controller, objects_original, oop_IDs, self.H,self.W)
        else:
            self.image_dict_original = None

        mess_up_from_loaded(self.controller, object_messup_meta)

        objects_messup = self.controller.last_event.metadata['objects']
        self.objects_messup = objects_messup

        if args.save_object_images:
            self.image_dict_messup = utils.aithor.get_images_of_objects(self.controller, objects_messup, oop_IDs, self.H, self.W)
        else:
            self.image_dict_messup = None

        event = self.controller.step(action="GetReachablePositions")
        nav_pts = event.metadata["actionReturn"]
        nav_pts = np.array([list(d.values()) for d in nav_pts])

        if len(nav_pts)<5: # something wrong happened here, reset
            self.controller.reset(scene=self.mapname_current)
            mess_up_from_loaded(self.controller, object_messup_meta)

        event = self.controller.step(action="GetReachablePositions")
        nav_pts = event.metadata["actionReturn"]
        nav_pts = np.array([list(d.values()) for d in nav_pts])

        self.move_agent_to_scene_center(nav_pts)
        
    def render_episode_images(self):
        '''
        This render images of all objects that were picked up by agent: (1) original objects, (2) messup objects, (3) relocated (picked up) objects by the agent. Used for mechanical turk evaluation.
        '''
        if not args.save_object_images:
            print("ERROR: Render images was called without turning on --save_object_images. Please enable this flag to allow rendering.")
            print("Skipping rendering...")
            return
        if not self.is_done():
            print("WARNING: render_episode_images() called before episode finished.")

        objects_cleanup = self.controller.last_event.metadata['objects']

        self.image_dict_cleanup = utils.aithor.get_images_of_objects(self.controller, objects_cleanup, self.oop_IDs, self.H,self.W)

        root_folder = os.path.join(args.image_dir, 'cleanup')
        root_folder = os.path.join(root_folder, self.mapname_current)
        root_folder = os.path.join(root_folder, str(self.n))
        if not os.path.exists(root_folder):
            os.makedirs(root_folder, exist_ok = True)
        for key in list(self.image_dict_cleanup.keys()):
            if key not in self.picked_up:
                continue
            rgb = self.image_dict_cleanup[key]['rgb']
            receptacle = self.image_dict_cleanup[key]['receptacle']
            plt.figure(1, figsize=(14, 8)); plt.clf()
            plt.imshow(rgb)
            plt.xticks([])
            plt.yticks([])
            plt.gca().axis('off')
            name = key.split('_')[0]
            # plt.title(name)
            plt.savefig(os.path.join(root_folder, f'{key}-{receptacle}.png'), bbox_inches='tight')


        root_folder = os.path.join(args.image_dir, 'original')
        root_folder = os.path.join(root_folder, self.mapname_current)
        root_folder = os.path.join(root_folder, str(self.n))
        if not os.path.exists(root_folder):
            os.makedirs(root_folder, exist_ok = True)
        for key in list(self.image_dict_original.keys()):
            if key not in self.picked_up:
                continue
            rgb = self.image_dict_original[key]['rgb']
            receptacle = self.image_dict_original[key]['receptacle']
            plt.figure(1, figsize=(14, 8)); plt.clf()
            plt.imshow(rgb)
            plt.xticks([])
            plt.yticks([])
            plt.gca().axis('off')
            name = key.split('_')[0]
            # plt.title(name)
            plt.savefig(os.path.join(root_folder, f'{key}-{receptacle}.png'), bbox_inches='tight')


        root_folder = os.path.join(args.image_dir, 'messup')
        root_folder = os.path.join(root_folder, self.mapname_current)
        root_folder = os.path.join(root_folder, str(self.n))
        if not os.path.exists(root_folder):
            os.makedirs(root_folder, exist_ok = True)
            # os.mkdir(root_folder)
        for key in list(self.image_dict_messup.keys()):
            if key not in self.picked_up:
                continue
            rgb = self.image_dict_messup[key]['rgb']
            receptacle = self.image_dict_messup[key]['receptacle']
            plt.figure(1, figsize=(14, 8)); plt.clf()
            plt.imshow(rgb)
            plt.xticks([])
            plt.yticks([])
            plt.gca().axis('off')
            name = key.split('_')[0]
            # plt.title(name)
            plt.savefig(os.path.join(root_folder, f'{key}-{receptacle}.png'), bbox_inches='tight')
        

    def finish(self):
        print("Finished all episodes!")
        self.controller.stop()

    def move_agent_to_scene_center(self, nav_pts):
        '''
        Move agent close to scene center. 
        '''
        # initialize agent and get object info
        # event = self.controller.step(action="GetReachablePositions")
        # nav_pts = event.metadata["actionReturn"]
        # nav_pts = np.array([list(d.values()) for d in nav_pts])

        # move agent close to scene center
        scene_center = np.expand_dims(np.array(list(self.controller.last_event.metadata['sceneBounds']['center'].values())), axis=0)
        argmin_dist_2_scene_center = np.argmin(np.sqrt(np.sum((nav_pts - scene_center)**2, axis=1)))
        pos_start = nav_pts[argmin_dist_2_scene_center]

        # initialize rotation randomly
        if self.init_agent_rotation_random:
            rots = [0., 90., 180., 270.]
            rand_rot_ind = np.random.randint(len(rots))
            rot = rots[rand_rot_ind]
        else:
            rot = 0.

        self.controller.step(
            'TeleportFull', 
            x=pos_start[0], y=pos_start[1], z=pos_start[2], 
            rotation=dict(x=0.0, y=rot, z=0.0), 
            horizon=0.0, standing=True
            )

    @property
    def held_object(self):
        """Return the data corresponding to the object held by the agent (if
        any)."""
        metadata = self.controller.last_event.metadata

        if len(metadata["inventoryObjects"]) == 0:
            return None

        assert len(metadata["inventoryObjects"]) <= 1

        held_obj_id = metadata["inventoryObjects"][0]["objectId"]
        return next(o for o in metadata["objects"] if o["objectId"] == held_obj_id)

    def pickup_object(self, x: float, y: float) -> bool:
        """Pick up the object corresponding to x/y.

        The action will not be successful if the object at x/y is not
        pickupable.

        # Parameters
        x : (float, min=0.0, max=1.0) horizontal percentage from the last frame
           that the target object is located.
        y : (float, min=0.0, max=1.0) vertical percentage from the last frame
           that the target object is located.

        # Returns
        `True` if the action was successful, otherwise `False`.
        """
        # if len(self.controller.last_event.metadata["inventoryObjects"]) != 0:
        #     return False

        self.controller.step(
            action="PickupObject",
            # objectId=object_id_select_,
            x=x,
            y=y,
            forceAction=True,
        )
        if self.controller.last_event.metadata['lastActionSuccess']:
            self.picked_up.append(self.held_object["name"])
            return True
        return False

    def open_object(self, x: float, y: float) -> bool:
        """Opens the object corresponding to x/y.

        # Parameters
        x : (float, min=0.0, max=1.0) horizontal percentage from the last frame
           that the target object is located.
        y : (float, min=0.0, max=1.0) vertical percentage from the last frame
           that the target object is located.

        # Returns
        `True` if the action was successful, otherwise `False`.
        """

        self.controller.step(
            action="OpenObject",
            x=x,
            y=y,
            forceAction=True,
        )
        if self.controller.last_event.metadata['lastActionSuccess']:
            return True
        return False

    def close_object(self, x: float, y: float) -> bool:
        """Closes the object corresponding to x/y.

        # Parameters
        x : (float, min=0.0, max=1.0) horizontal percentage from the last frame
           that the target object is located.
        y : (float, min=0.0, max=1.0) vertical percentage from the last frame
           that the target object is located.

        # Returns
        `True` if the action was successful, otherwise `False`.
        """

        self.controller.step(
            action="CloseObject",
            x=x,
            y=y,
            forceAction=True,
        )
        if self.controller.last_event.metadata['lastActionSuccess']:
            return True
        return False

    def put_object(self, x: float, y: float) -> bool:
        """Pick up the object corresponding to x/y.

        The action will not be successful if the object at x/y is not
        pickupable.

        # Parameters
        x : (float, min=0.0, max=1.0) horizontal percentage from the last frame
           that the target object is located.
        y : (float, min=0.0, max=1.0) vertical percentage from the last frame
           that the target object is located.

        # Returns
        `True` if the action was successful, otherwise `False`.
        """
        # if self.held_object is None:
        #     self.task.controller.last_event.metadata["lastActionSuccess"] = False
        #     return False
        while True:
            # get held object out of the way
            self.controller.step(
                action="MoveHeldObjectUp",
                moveMagnitude=0.05,
                forceVisible=False
            )
            # sum_ += 0.05
            if not self.controller.last_event.metadata["lastActionSuccess"]:
                break
        self.controller.step(
            action="PutObject",
            # objectId=object_id_select_,
            x=x,
            y=y,
            forceAction=True,
        )
        if self.controller.last_event.metadata['lastActionSuccess']:
            return True
        if 'CLOSED' in self.controller.last_event.metadata["errorMessage"]: 
            # open if closed then try again
            while True:
                self.controller.step(
                    action="MoveHeldObjectUp",
                    moveMagnitude=0.05,
                    forceVisible=False
                )
                if not self.controller.last_event.metadata["lastActionSuccess"]:
                    break
            query = self.controller.step(
                action="GetObjectInFrame",
                x=x,
                y=y,
                checkVisible=False
            )
            object_id = query.metadata["actionReturn"]
            self.controller.step(
                action="OpenObject",
                objectId=object_id,
                openness=1,
                forceAction=True
            )
            self.controller.step(
                action="PutObject",
                objectId=object_id,
                forceAction=True,
            )
            if self.controller.last_event.metadata['lastActionSuccess']:
                return True
        return False

    def drop_object(self):
        """Drops held object
        """
        # if self.held_object is None:
        #     return False
        self.controller.step(
            "DropHandObjectAhead",
            forceAction=True,
            autoSimulation=False,
            randomMagnitude=0.0,
            actionSimulationSeconds=1.5,
        )
        if self.controller.last_event.metadata['lastActionSuccess']:
            return True
        return False

    # def pickup_object_category(self, object_category) -> bool:
    #     """Pick up the object corresponding to object catefory

    #     # Parameters
    #     object_category: object category to target

    #     # Returns
    #     `True` if the action was successful, otherwise `False`.
    #     """
    #     if len(self.last_event.metadata["inventoryObjects"]) != 0:
    #         return False
    #     objects = self.controller.last_event.metadata['objects']
    #     object_dict = {}
    #     for obj in objects:
    #         object_dict[obj['objectId']] = obj
    #     object_id_select = []
    #     for key in list(object_dict.keys()):
    #         if object_category in key and object_dict[key]['visible']:
    #             object_id_select.append(key)
    #     if len(object_id_select)==0:
    #         # no object of that category in view
    #         return False
    #     else:
    #         # try to place on all the objects in view with that category label
    #         for object_id_select_ in object_id_select:
    #             self.controller.step(
    #                 action="PickupObject",
    #                 objectId=object_id_select_,
    #                 forceAction=True,
    #             )
    #             if self.controller.last_event.metadata['lastActionSuccess']:
    #                 self.picked_up.append(object_dict[key]["name"])
    #                 return True
    #     return False

    def move_ahead(self) -> bool:
        """Move the agent ahead from its facing direction by 0.25 meters."""
        if args.movementGaussianSigma is not None:
            amount = np.random.normal(args.STEP_SIZE, scale=args.movementGaussianSigma)
        else:
            amount = args.STEP_SIZE
        self.controller.step(
                action="MoveAhead",
                moveMagnitude=amount
            )

    def move_back(self) -> bool:
        """Move the agent back from its facing direction by 0.25 meters."""
        if args.movementGaussianSigma is not None:
            amount = np.random.normal(args.STEP_SIZE, scale=args.movementGaussianSigma)
        else:
            amount = args.STEP_SIZE
        self.controller.step(
                action="MoveBack",
                moveMagnitude=args.STEP_SIZE
            )

    def move_right(self) -> bool:
        """Move the agent right from its facing direction by 0.25 meters."""
        if args.movementGaussianSigma is not None:
            amount = np.random.normal(args.STEP_SIZE, scale=args.movementGaussianSigma)
        else:
            amount = args.STEP_SIZE
        self.controller.step(
                action="MoveRight",
                moveMagnitude=amount
            )

    def move_left(self) -> bool:
        """Move the agent left from its facing direction by 0.25 meters."""
        if args.movementGaussianSigma is not None:
            amount = np.random.normal(args.STEP_SIZE, scale=args.movementGaussianSigma)
        else:
            amount = args.STEP_SIZE
        self.controller.step(
                action="MoveLeft",
                moveMagnitude=amount
            )

    def rotate_left(self) -> bool:
        """Rotate the agent left from its facing direction."""
        if args.rotateGaussianSigma is not None:
            amount = np.random.normal(args.DT, scale=args.rotateGaussianSigma)
        else:
            amount = args.DT
        self.controller.step(
            action="RotateLeft",
            degrees=amount
        )

    def rotate_right(self) -> bool:
        """Rotate the agent left from its facing direction."""
        if args.rotateGaussianSigma is not None:
            amount = np.random.normal(args.DT, scale=args.rotateGaussianSigma)
        else:
            amount = args.DT
        self.controller.step(
            action="RotateRight",
            degrees=amount
        )

    def stand(self) -> bool:
        """Stand the agent from the crouching position."""
        self.controller.step(
            action="Stand",
        )

    def crouch(self) -> bool:
        """Crouch the agent from the standing position."""
        self.controller.step(
            action="Crouch",
        )

    def look_up(self) -> bool:
        """Turn the agent's head and camera up by 30 degrees."""
        if args.rotateGaussianSigma is not None:
            amount = np.random.normal(args.HORIZON_DT, scale=args.rotateGaussianSigma)
            amount = np.round(amount, 1)
        else:
            amount = args.HORIZON_DT
        self.controller.step(
            action="LookUp",
            degrees=amount
        )

    def look_down(self) -> bool:
        """Turn the agent's head and camera down by 30 degrees."""
        if args.rotateGaussianSigma is not None:
            amount = np.random.normal(args.HORIZON_DT, scale=args.rotateGaussianSigma)
            amount = np.round(amount, 1)
        else:
            amount = args.HORIZON_DT
        self.controller.step(
            action="LookDown",
            degrees=amount
        )

    def done(self) -> bool:
        """Agent's signal that it's completed its current rearrangement phase.

        Note that we do not automatically switch from the walkthrough
        phase to the unshuffling phase, and vice-versa, that is up to
        the user. This allows users to call .poses after the agent calls
        done, and have it correspond to the current episode.
        """
        self.controller.step(
            action="Done",
        )
        self.done_called = True

    # def place_object(self, object_category):
    #     '''
    #     Places object on object category if in view. If fail, then tries to place on general receptacle. 
    #     '''

    #     object_dict = {}
    #     for obj in objects:
    #         object_dict[obj['objectId']] = obj
    #     object_id_select = []
    #     for key in list(object_dict.keys()):
    #         if object_category in key and object_dict[key]['visible']:
    #             object_id_select.append(key)

    #     if len(object_id_select)==0:
    #         # no object of that category in view
    #         pass
    #     else:
    #         # try to place on all the objects in view with that category label
    #         for object_id_select_ in object_id_select:
    #             # # move held object up otherwise placement can sometimes fail
    #             # while True:
    #             #     self.controller.step(
    #             #         action="MoveHeldObjectUp",
    #             #         moveMagnitude=0.05,
    #             #         forceVisible=False
    #             #     )
    #             #     # sum_ += 0.05
    #             #     if not self.controller.last_event.metadata["lastActionSuccess"]:
    #             #         break
    #             self.controller.step(
    #                 action="PutObject",
    #                 objectId=object_id_select_,
    #                 forceAction=True,
    #             )
    #             if self.controller.last_event.metadata['lastActionSuccess']:
    #                 return True
    #             elif 'CLOSED' in self.controller.last_event.metadata["errorMessage"]: # open if closed
    #                 self.controller.step(
    #                     action="OpenObject",
    #                     objectId=object_id_select_,
    #                     openness=1,
    #                     forceAction=True
    #                 )
    #                 self.controller.step(
    #                     action="PutObject",
    #                     objectId=object_id_select_,
    #                     forceAction=True,
    #                 )
    #                 if self.controller.last_event.metadata['lastActionSuccess']:
    #                     return True

    #     # We couldn't teleport the object to the target location, let's try placing it
    #     # in a visible receptacle.
    #     possible_receptacles = [
    #         o for o in self.controller.last_event.metadata["objects"] if o["visible"] and o["receptacle"]
    #     ]
    #     possible_receptacles = sorted(
    #         possible_receptacles, key=lambda o: (o["distance"], o["objectId"])
    #     )
    #     for possible_receptacle in possible_receptacles:
    #         self.controller.step(
    #             action="PlaceHeldObject",
    #             objectId=possible_receptacle["objectId"],
    #             # **self.physics_step_kwargs,
    #         )
    #         if self.controller.last_event.metadata["lastActionSuccess"]:
    #             return False

    #     # # We failed to place the object into a receptacle, let's just drop it.
    #     # if not self.controller.last_event.metadata["lastActionSuccess"]:
    #     #     self.controller.step(
    #     #         "DropHandObjectAhead",
    #     #         forceAction=True,
    #     #         autoSimulation=False,
    #     #         randomMagnitude=0.0,
    #     #         **{**self.physics_step_kwargs, "actionSimulationSeconds": 1.5},
    #     #     )

    #     return False


    def get_scene_split(self):
        num_train_houses = args.num_train_houses
        num_test_houses = args.num_test_houses
        num_val_houses = args.num_val_houses

        ############## Get training houses ##############
        # training house is 16-20
        a_h = np.arange(1, 1+num_train_houses)
        b_h = np.arange(201, 201+num_train_houses)
        c_h = np.arange(301, 301+num_train_houses)
        d_h = np.arange(401, 401+num_train_houses)
        abcd = np.hstack((a_h,b_h,c_h,d_h))

        mapnames = []
        # housesets = []
        for i in range(a_h.shape[0]):
            houseset = []
            if True: #args.do_kitchen:
                mapname = 'FloorPlan' + str(a_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if True: #args.do_living_room:
                mapname = 'FloorPlan' + str(b_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if True: #args.do_bedroom:
                mapname = 'FloorPlan' + str(c_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if True: #args.do_bathroom:
                mapname = 'FloorPlan' + str(d_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            # housesets.append(houseset)

        mapnames_train = mapnames
        # self.housesets_train = housesets

        ############## Get validation houses ##############
        # validation houses
        a_h = np.arange(1+num_train_houses, 1+num_train_houses+num_val_houses)
        b_h = np.arange(201+num_train_houses, 201+num_train_houses+num_val_houses)
        c_h = np.arange(301+num_train_houses, 301+num_train_houses+num_val_houses)
        d_h = np.arange(401+num_train_houses, 401+num_train_houses+num_val_houses)
        abcd = np.hstack((a_h,b_h,c_h,d_h))

        mapnames = []
        # housesets = []
        for i in range(a_h.shape[0]):
            houseset = []
            if True: #args.do_kitchen:
                mapname = 'FloorPlan' + str(a_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if True: #args.do_living_room:
                mapname = 'FloorPlan' + str(b_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if True: #args.do_bedroom:
                mapname = 'FloorPlan' + str(c_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if True: #args.do_bathroom:
                mapname = 'FloorPlan' + str(d_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            # housesets.append(houseset)

        mapnames_val = mapnames
        # self.housesets_val = housesets

        ############## Get testing houses ##############
        # get rest of the houses in orders
        a_h = np.arange(1+num_train_houses+num_val_houses, 1+num_train_houses+num_test_houses+num_val_houses)
        b_h = np.arange(201+num_train_houses+num_val_houses, 201+num_train_houses+num_test_houses+num_val_houses)
        c_h = np.arange(301+num_train_houses+num_val_houses, 301+num_train_houses+num_test_houses+num_val_houses)
        d_h = np.arange(401+num_train_houses+num_val_houses, 401+num_train_houses+num_test_houses+num_val_houses)
        abcd = np.hstack((a_h,b_h,c_h,d_h))

        mapnames = []
        # housesets = []
        for i in range(a_h.shape[0]):
            # houseset = []
            if True: #args.do_kitchen:
                mapname = 'FloorPlan' + str(a_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if True: #args.do_living_room:
                mapname = 'FloorPlan' + str(b_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if True: #args.do_bedroom:
                mapname = 'FloorPlan' + str(c_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if True: #args.do_bathroom:
                mapname = 'FloorPlan' + str(d_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            # housesets.append(houseset)

        mapnames_test = mapnames
        # self.housesets_test = housesets

        mapnames_split = {"train":mapnames_train, "val":mapnames_val, "test":mapnames_test}

        return mapnames_split

    def get_metrics(self):
        self.objects_current = self.controller.last_event.metadata['objects']
        metrics = self.evaluate_cleanup(self.objects_original, self.objects_current, self.objects_messup)
        return metrics

    # Below are some alternate evaluation methods that could be used but do not capture common sense placements well

    def evaluate_cleanup(self, objs_original, objs_cleanup, objs_messup):
        '''
        Measure euclidean distance from original location
        '''

        objects_current_scenes = set([obj["objectType"] for obj in objs_original] + ["Floor"])

        object_placement_total_cur_scene = {}
        object_receptacle_probabilities = copy.deepcopy(self.object_receptacle_counts)
        for k in self.object_receptacle_counts.keys():
            receptacle_counts = self.object_receptacle_counts[k]
            total = 0
            for r in receptacle_counts.keys():
                if r in objects_current_scenes:
                    total += receptacle_counts[r]
            object_placement_total_cur_scene[k] = total
            for r in receptacle_counts.keys():
                object_receptacle_probabilities[k][r] = max(object_receptacle_probabilities[k][r],0)
                object_receptacle_probabilities[k][r] /= max(1,total)

        objs_messup_dict = {obj["name"]:obj for obj in objs_messup}
        objs_original_dict = {obj["name"]:obj for obj in objs_original}

        correctly_moved = 0
        incorrectly_moved = 0
        for obj_id in self.picked_up:
            if obj_id in self.oop_IDs:
                correctly_moved += 1
            else:
                incorrectly_moved += 1

        num_moved = 0
        num_dirty = 0
        energies_dirty = {"energy_oop":0, "energy_allpickedandoop":0, "energy_all":0}
        energies_current = {"energy_oop":0, "energy_allpickedandoop":0, "energy_all":0}
        energies_original = {"energy_oop":0, "energy_allpickedandoop":0, "energy_all":0}
        energies_best = {"energy_oop":0, "energy_allpickedandoop":0, "energy_all":0}
        for obj_current in objs_cleanup:
            if not obj_current["pickupable"]:
                continue

            obj_name = obj_current["name"]
            obj_messup = objs_messup_dict[obj_name]
            obj_original = objs_original_dict[obj_name]

            mess_o_type = obj_messup["objectType"]
            mess_r_type = "Floor" if obj_messup["parentReceptacles"] is None else obj_messup["parentReceptacles"][0].split('|')[0]
            clean_o_type = obj_current["objectType"]
            clean_r_type = "Floor" if obj_current["parentReceptacles"] is None else obj_current["parentReceptacles"][0].split('|')[0]
            original_o_type = obj_original["objectType"]
            original_r_type = "Floor" if obj_original["parentReceptacles"] is None else obj_original["parentReceptacles"][0].split('|')[0]

            # energy of oop objects only
            if obj_name in self.oop_IDs:
                energies_dirty["energy_oop"] += object_receptacle_probabilities[mess_o_type][mess_r_type]
                energies_current["energy_oop"] += object_receptacle_probabilities[clean_o_type][clean_r_type]
                energies_original["energy_oop"] += object_receptacle_probabilities[original_o_type][original_r_type]
                energies_best["energy_oop"] += max(list(object_receptacle_probabilities[clean_o_type].values()))
            
            # energy of any picked up object + oop object
            if obj_name in (self.picked_up + self.oop_IDs):
                energies_dirty["energy_allpickedandoop"] += object_receptacle_probabilities[mess_o_type][mess_r_type]
                energies_current["energy_allpickedandoop"] += object_receptacle_probabilities[clean_o_type][clean_r_type]
                energies_original["energy_allpickedandoop"] += object_receptacle_probabilities[original_o_type][original_r_type]
                energies_best["energy_allpickedandoop"] += max(list(object_receptacle_probabilities[clean_o_type].values()))
            
            # energy of all objects
            energies_dirty["energy_all"] += object_receptacle_probabilities[mess_o_type][mess_r_type]
            energies_current["energy_all"] += object_receptacle_probabilities[clean_o_type][clean_r_type]
            energies_original["energy_all"] += object_receptacle_probabilities[original_o_type][original_r_type]
            energies_best["energy_all"] += max(list(object_receptacle_probabilities[clean_o_type].values()))
        
        metrics = {}
        for k in energies_current.keys():
            # take min to restrict metric 0-1
            metrics[k] = np.clip((1 - energies_current[k]/energies_original[k]) / (1 - energies_dirty[k]/energies_original[k]), a_min=0, a_max=1)

        metrics["correctly_moved"] = correctly_moved
        metrics["incorrectly_moved"] = incorrectly_moved
        metrics["missed_moved"] = len(self.oop_IDs) - correctly_moved
        metrics["total_moved"] = correctly_moved+incorrectly_moved
        metrics["steps"] = self.step_count
        metrics["errors"] = self.api_fails
        
        return metrics

    def aggregate_metrics(self, metrics):
        keys_include = ['energy_oop', 'energy_allpickedandoop', 'energy_all', 'correctly_moved', 'incorrectly_moved', 'missed_moved', 'steps', 'errors']

        metrics_avg = {}
        for f_n in keys_include:
            metrics_avg[f_n] = 0
        count = 0
        for k in metrics.keys():
            for f_n in keys_include:
                metrics_avg[f_n] += metrics[k][f_n]
            count += 1
        for f_n in keys_include:
            metrics_avg[f_n] /=  count 
        metrics_avg['num episodes'] = len(metrics.keys())

        return metrics_avg


