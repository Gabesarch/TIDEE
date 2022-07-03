
import numpy as np
import os
# import hyperparams as hyp
from arguments import args
import ipdb
st = ipdb.set_trace
from task_base.messup import mess_up_from_loaded
import utils.aithor
import pickle
from allenact_plugins.ithor_plugin.ithor_util import include_object_data
import matplotlib.pyplot as plt

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
        self.split = split
        mapnames_split_dict = self.get_scene_split()
        self.mapnames_split = mapnames_split_dict[split]
        self.W, self.H = args.W, args.H
        if max_episode_steps is None:
            self.max_episode_steps = args.max_episode_steps
        else:
            self.max_episode_steps = max_episode_steps

        ns = {"train":args.n_train_messup, "val":args.n_val_messup, "test":args.n_test_messup}
        self.ns_split = list(np.arange(ns[split]))

        self.num_episodes_total = len(self.mapnames_split) * len(self.ns_split)

        self.next_ep_called = False

        self.diplay_every = diplay_every # display every ten actions

        self.init_episodes()

    def init_episodes(self):

        self.mapnames_iter = iter(self.mapnames_split)
        self.ns_iter = iter(self.ns_split)

        self.mapname_current = next(self.mapnames_iter)

        self.n = -1

        # self.start_next_episode()

    def get_episode_name(self):
        return f"{self.mapname_current}_{self.n}"

    def is_done(self):
        return (self.step_count>=self.max_episode_steps or self.done_called)

    def step(self, action, obj_relative_coord=None): #object_category=None):

        if not self.next_ep_called:
            print("ERROR: Must call start_next_episode() after init before calling actions")
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
        elif action=="Done":
            self.done()
        else:
            assert(False) # what action is this?
        self.step_count += 1

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

        self.next_ep_called = True
        self.step_count = 0
        
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

    def start_next_episode(self):

        self.next_ep_called = True
        self.step_count = 0
        
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

        # if hyp.save_messup or not hyp.load_messup:
        #     print("Messing up environment...")
        #     # mess up objects

        #     num_objects = hyp.num_objects_messup # number of objects to have the agent mess up in the room
        #     oop_objects_gt, oop_IDs = mess_up(self.controller, self.include_classes, num_objects, prob_drop=0.5, vis=vis)
        #     objects_messup = self.controller.last_event.metadata['objects']
        #     if hyp.save_messup:
        #         save_dict = {'oop_objects_gt':oop_objects_gt, 'oop_IDs':oop_IDs, 'objects_messup':objects_messup}
        #         root = os.path.join(hyp.messup_dir, mapname) 
        #         if not os.path.exists(root):
        #             os.mkdir(root)
        #         # n = 0
        #         pickle_fname = f'{n}.p'
        #         fname = os.path.join(root, pickle_fname)
        #         # if os.path.exists(fname):
        #         #     continue
        #         print("saving", fname)
        #         with open(fname, 'wb') as f:
        #             pickle.dump(save_dict, f, protocol=4)
        #         print("done")
        #         # assert(False)
        #         continue
        # elif hyp.load_messup:
        #     root = os.path.join(hyp.messup_dir, mapname) 
        #     # n = 0
        #     pickle_fname = f'{n}.p'
        #     fname = os.path.join(root, pickle_fname)
        #     print("loading", fname)
        #     with open(fname, 'rb') as f:
        #         load_dict = pickle.load(f)
        #     oop_objects_gt = load_dict['oop_objects_gt']
        #     oop_IDs = load_dict['oop_IDs']
        #     objects_messup = load_dict['objects_messup']
        #     if args.save_object_images:
        #         image_dict_original = utils.aithor.get_images_of_objects(self.controller, objects_original, oop_IDs, self.pix_T_camX, self.H,self.W)
        #     mess_up_from_loaded(self.controller, objects_messup, oop_IDs=oop_IDs)
    
        
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
        # root_images = f'./MT_images/'
        
        # save cleanup images (after episode)
        if not os.path.exists(args.image_dir):
            os.mkdir(args.image_dir)
        root_folder = os.path.join(args.image_dir, 'cleanup')
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)
        root_folder = os.path.join(root_folder, self.mapname_current)
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)
        root_folder = os.path.join(root_folder, str(self.n))
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)
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


        # save original location of objects location
        if not os.path.exists(args.image_dir):
            os.mkdir(args.image_dir)
        root_folder = os.path.join(args.image_dir, 'original')
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)
        root_folder = os.path.join(root_folder, self.mapname_current)
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)
        root_folder = os.path.join(root_folder, str(self.n))
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)
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


        # save messup locations of objects
        if not os.path.exists(args.image_dir):
            os.mkdir(args.image_dir)
        root_folder = os.path.join(args.image_dir, 'messup')
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)
        root_folder = os.path.join(root_folder, self.mapname_current)
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)
        root_folder = os.path.join(root_folder, str(self.n))
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)
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
        with include_object_data(self.controller):
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



    def place_object(self, object_category):
        '''
        Places object on object category if in view. If fail, then tries to place on general receptacle. 
        '''

        object_dict = {}
        for obj in objects:
            object_dict[obj['objectId']] = obj
        object_id_select = []
        for key in list(object_dict.keys()):
            if object_category in key and object_dict[key]['visible']:
                object_id_select.append(key)

        if len(object_id_select)==0:
            # no object of that category in view
            pass
        else:
            # try to place on all the objects in view with that category label
            for object_id_select_ in object_id_select:
                # # move held object up otherwise placement can sometimes fail
                # while True:
                #     self.controller.step(
                #         action="MoveHeldObjectUp",
                #         moveMagnitude=0.05,
                #         forceVisible=False
                #     )
                #     # sum_ += 0.05
                #     if not self.controller.last_event.metadata["lastActionSuccess"]:
                #         break
                self.controller.step(
                    action="PutObject",
                    objectId=object_id_select_,
                    forceAction=True,
                )
                if self.controller.last_event.metadata['lastActionSuccess']:
                    return True
                elif 'CLOSED' in self.controller.last_event.metadata["errorMessage"]: # open if closed
                    self.controller.step(
                        action="OpenObject",
                        objectId=object_id_select_,
                        openness=1,
                        forceAction=True
                    )
                    self.controller.step(
                        action="PutObject",
                        objectId=object_id_select_,
                        forceAction=True,
                    )
                    if self.controller.last_event.metadata['lastActionSuccess']:
                        return True

        # We couldn't teleport the object to the target location, let's try placing it
        # in a visible receptacle.
        possible_receptacles = [
            o for o in self.controller.last_event.metadata["objects"] if o["visible"] and o["receptacle"]
        ]
        possible_receptacles = sorted(
            possible_receptacles, key=lambda o: (o["distance"], o["objectId"])
        )
        for possible_receptacle in possible_receptacles:
            self.controller.step(
                action="PlaceHeldObject",
                objectId=possible_receptacle["objectId"],
                # **self.physics_step_kwargs,
            )
            if self.controller.last_event.metadata["lastActionSuccess"]:
                return False

        # # We failed to place the object into a receptacle, let's just drop it.
        # if not self.controller.last_event.metadata["lastActionSuccess"]:
        #     self.controller.step(
        #         "DropHandObjectAhead",
        #         forceAction=True,
        #         autoSimulation=False,
        #         randomMagnitude=0.0,
        #         **{**self.physics_step_kwargs, "actionSimulationSeconds": 1.5},
        #     )

        return False


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
            if args.do_kitchen:
                mapname = 'FloorPlan' + str(a_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if args.do_living_room:
                mapname = 'FloorPlan' + str(b_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if args.do_bedroom:
                mapname = 'FloorPlan' + str(c_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if args.do_bathroom:
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
            if args.do_kitchen:
                mapname = 'FloorPlan' + str(a_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if args.do_living_room:
                mapname = 'FloorPlan' + str(b_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if args.do_bedroom:
                mapname = 'FloorPlan' + str(c_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if args.do_bathroom:
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
            if args.do_kitchen:
                mapname = 'FloorPlan' + str(a_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if args.do_living_room:
                mapname = 'FloorPlan' + str(b_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if args.do_bedroom:
                mapname = 'FloorPlan' + str(c_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            if args.do_bathroom:
                mapname = 'FloorPlan' + str(d_h[i])
                mapnames.append(mapname)
                # houseset.append(mapname)
            # housesets.append(houseset)

        mapnames_test = mapnames
        # self.housesets_test = housesets

        mapnames_split = {"train":mapnames_train, "val":mapnames_val, "test":mapnames_test}

        return mapnames_split


    # Below are some alternate evaluation methods that could be used but do not capture common sense placements well

    def evaluate_cleanup(self, objs_original, objs_cleanup, objs_messup):
        '''
        Measure euclidean distance from original location
        '''
        original_dict = {}
        for obj_idx in range(len(objs_original)):
            obj = objs_original[obj_idx]
            if (obj['objectType'] not in self.include_classes) or not obj["pickupable"]:
                continue
            original_dict[obj["name"]] = {'position':np.array(list(obj['axisAlignedBoundingBox']['center'].values()))}

        total_displacement_cleanup = 0.0
        for obj_i in range(len(objs_cleanup)):
            obj_clean = objs_cleanup[obj_i]
            if (obj_clean['objectType'] not in self.include_classes) or not obj_clean["pickupable"]:
                continue
            obj_clean_ID = obj_clean["name"]
            obj_clean_pos = np.array(list(obj_clean['axisAlignedBoundingBox']['center'].values()))
            obj_original_pos = original_dict[obj_clean_ID]['position']
            euclidean = np.sqrt(np.sum((obj_original_pos - obj_clean_pos)**2))
            total_displacement_cleanup += euclidean

        total_displacement_messup = 0.0
        for obj_i in range(len(objs_messup)):
            obj_messup = objs_messup[obj_i]
            if (obj_messup['objectType'] not in self.include_classes) or not obj_messup["pickupable"]:
                continue
            obj_messup_ID = obj_messup["name"]
            obj_messup_pos = np.array(list(obj_messup['axisAlignedBoundingBox']['center'].values()))
            obj_original_pos = original_dict[obj_messup_ID]['position']
            euclidean = np.sqrt(np.sum((obj_original_pos - obj_messup_pos)**2))
            total_displacement_messup += euclidean

        displacement_measure = total_displacement_cleanup/total_displacement_messup
        
        return displacement_measure

    # # compute global precision and recall of detector (i.e. picked up items)
    # oop_IDs_check = oop_IDs.copy()
    # true_pos = 0
    # false_pos = 0
    # for p_u_id in picked_up:
    #     if p_u_id in oop_IDs_check:
    #         oop_IDs_check.remove(p_u_id)
    #         true_pos += 1
    #     else:
    #         false_pos += 1
    # base = true_pos + false_pos
    # if base==0:
    #     precision = 0
    # else:
    #     precision = true_pos / (true_pos + false_pos)
    # recall = true_pos / hyp.num_objects_messup
    # errors['precision'].append(precision)
    # errors['recall'].append(recall)

    # print("PRECISION:", precision)
    # print("RECALL:", recall)

    # displacement_measure = self.evaluate_cleanup(objects_original, objects_cleanup, objects_messup)
    # print("DISPLACEMENT MEASURE:", displacement_measure)
    # displacement_measures.append(displacement_measure)  
    # print("ERRORS", errors[episode]) 
    # errors['displacement_measures'] = displacement_measures



