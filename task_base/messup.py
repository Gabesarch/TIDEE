import numpy as np
import utils.aithor
import matplotlib.pyplot as plt
import math
from arguments import args

import ipdb
st = ipdb.set_trace

def mess_up_from_loaded(controller, objects_messup, oop_IDs=None):

    print("messing up objects from loaded file")
    objectPoses = utils.aithor.change_pose(objects_messup)
    controller.step(
        action='SetObjectPoses',
        objectPoses=objectPoses,
    )
    if not controller.last_event.metadata['lastActionSuccess']:
        print("messup failed?")
    print("Done.")

def mess_up(controller, include_classes, num_objects, prob_drop=0.5, vis=None, force_visible=True):
    '''
    Here a "mess-up" agent goes around, targets small objects, picks them up, and either drops them or throws them when standing in a random location
    inputs:
        1) num_objects: number of objects that the mess up agent should disarrange
        2) prob_drop: probability of dropping the object compared to throwing it (throwing = 1-prob_drop)
        3) figure for making animation plot
    '''

    # force visible (cant be in these to start)
    obj_rem = ['Fridge', 'Box', "Dresser", 'Pot', 'Pan', 'Bowl', 'Microwave', 'GarbageCan', 'Plate', 'Cabinet', 'Safe', 'Drawer', ]
    
    # make all objects unbreakable
    for obj_type in include_classes:
        controller.step(action='MakeObjectsOfTypeUnbreakable', objectType=obj_type)

    # if force_visible:
    #     random_seed = 42 #np.random.randint(1000)
    #     controller.step(action="InitialRandomSpawn",
    #         randomSeed=random_seed,
    #         forceVisible=True,
    #         numPlacementAttempts=10,
    #         placeStationary=False,
    #     )

    # get pickupable objects
    objects = controller.last_event.metadata['objects']
    object_dict = {}
    pickupable_objects = []
    for obj_idx in range(len(objects)):
        obj = objects[obj_idx]
        object_dict[obj["name"]] = {'meta':obj, 'out_of_place':False, 'obj_idx':obj_idx}
        if (obj['objectType'] not in include_classes) or not obj["pickupable"]:
            continue
        if force_visible:
            vis_obj = True
            if obj['parentReceptacles'] is not None:
                for obj_rec in obj['parentReceptacles']:
                    for obj_r in obj_rem:
                        if obj_r in obj_rec:
                            vis_obj = False
                            break
                        if not vis_obj:
                            break 
                    if not vis_obj:
                        break 
            if not vis_obj:
                continue 
        pickupable_objects.append(obj)

    # get navigable locations
    event = controller.step(action="GetReachablePositions")
    nav_pts = event.metadata["actionReturn"]
    nav_pts = np.array([list(d.values()) for d in nav_pts])

    yaw_rotations = np.arange(0,360,45)

    # Mess up objects!
    num_navigable = len(nav_pts)
    oop_IDs = []
    # for n in range(num_objects):
    successes = 0
    while True:

        random_select_obj = np.random.randint(len(pickupable_objects))
        obj_select = pickupable_objects.pop(random_select_obj)
        # oop_objects.append(obj_select)

        objId = obj_select["objectId"]
        print(f"Object targeted is {objId}")
        parents = obj_select["parentReceptacles"]
        print(f"parent receptacles are: {parents}")

        # messup agent picks up an object
        controller.step(
            action="PickupObject",
            objectId=objId,
            forceAction=True,
            manualInteract=False
        )

        if vis is not None:
            vis.add_frame(controller.last_event.frame, text="PickupObject", add_map=False)

        for _ in range(10):
            # mess up agent moves to a random location
            random_select_nav = np.random.randint(num_navigable)
            nav_point = nav_pts[random_select_nav]
            yaw_select = np.random.choice(yaw_rotations)
            event = controller.step('TeleportFull', position=dict(x=nav_point[0], y=nav_point[1], z=nav_point[2]), rotation=dict(x=0.0, y=yaw_select, z=0.0), horizon=0.0, standing=True)

            if vis is not None:
                vis.add_frame(controller.last_event.frame, text="Teleport", add_map=False)

            # drop or throw object
            prob = np.random.uniform(0,1)
            
            if prob <= prob_drop:
                # controller.step(
                #     action="LookUp",
                # )
                utils.aithor.move_held_obj_out_of_view(controller, "MoveHeldObjectUp") # need this or aithor will complain there is an obstruction 
                action = "DropHandObject"
                print("ACTION: ", action)
                controller.step(
                    action=action,
                    forceAction=False
                )
                # controller.step(
                #     action="LookDown",
                # )
            else:
                
                utils.aithor.move_held_obj_out_of_view(controller, "MoveHeldObjectUp") # need this or aithor will complain there is an obstruction 
                action = "ThrowObject"
                print("ACTION: ", action)
                controller.step(
                    action=action,
                    moveMagnitude=150.0,
                    forceAction=False
                )
            
            if controller.last_event.metadata['lastActionSuccess']:
                break
            else:
                print(controller.last_event.metadata["errorMessage"])

        if not controller.last_event.metadata['lastActionSuccess']: # if fail then force action
                # utils.aithor.move_held_obj_out_of_view(controller, "MoveHeldObjectUp")
                print("Forcing")
                action = "DropHandObject"
                controller.step(
                    action=action,
                    forceAction=True
                )
            # action = "DropHandObject"
            # controller.step(
            #     action=action,
            #     forceAction=True
            # )
            # print("Moving object failed. moving object back to original location.")
            # objects_cur = controller.last_event.metadata['objects']
            # objectPoses = utils.aithor.change_pose_single_obj(objects_cur, obj_select['name'], obj_select['position'], obj_select['rotation'])
            # controller.step(
            #     action='SetObjectPoses',
            #     objectPoses=objectPoses,
            # )
        # else:
        #     successes += 1
        #     # update object dict
        #     object_dict[objId]['out_of_place'] = True
        #     obj_idx = object_dict[objId]['obj_idx']
        #     object_dict[objId]['meta'] = controller.last_event.metadata['objects'][obj_idx]
        #     oop_IDs.append(objId)

        successes += 1
        
        obj_name = obj_select["name"]
        oop_IDs.append(obj_name)
        object_dict[obj_name]['out_of_place'] = True
        obj_idx = object_dict[obj_name]['obj_idx']
        object_dict[obj_name]['meta'] = controller.last_event.metadata['objects'][obj_idx]
        print(object_dict[obj_name]['meta']['name'])
        print("end state:", controller.last_event.metadata['objects'][obj_idx]['axisAlignedBoundingBox']['center'])

        if vis is not None:
            vis.add_frame(controller.last_event.frame, text=action, add_map=False)

        if successes==num_objects:
            break

        # if obj_select["isPickedUp"]:

        # # just in case
        # action = "DropHandObject"
        # controller.step(
        #     action=action,
        #     forceAction=True
        # )
        

    return object_dict, oop_IDs
    
def save_mess_up():
    from task_base.aithor_base import Base
    import os
    import pickle
    base = Base()
    include_classes = base.include_classes
    mapnames_train = base.mapnames_train
    mapnames_val = base.mapnames_val
    mapnames_test = base.mapnames_test
    num_objects = args.num_objects
    controller = base.controller

    # for mapname in mapnames_train:
        # print(mapname)
    #     root = os.path.join(args.mess_up_dir, mapname)
    #     if not os.path.exists(root):
    #         os.mkdir(root)
    #     for n in range(args.n_train_messup):
    #         controller.reset(scene=mapname)
    #         object_dict, oop_IDs = mess_up(controller, include_classes, num_objects)
    #         objects_messup = controller.last_event.metadata['objects']
    #         save_dict = {'object_dict':object_dict, 'oop_IDs':oop_IDs, 'objects_messup':objects_messup}

    #         pickle_fname = f'{n}.p'
    #         fname_ = os.path.join(root, pickle_fname)

    #         print("saving", fname_)
    #         with open(fname_, 'wb') as f:
    #             pickle.dump(save_dict, f, protocol=4)
    #         print("done.")

    for mapname in mapnames_val:
        print(mapname)
        root = os.path.join(args.mess_up_dir, mapname)
        if not os.path.exists(root):
            os.mkdir(root)
        for n in range(args.n_val_messup):
            controller.reset(scene=mapname)
            object_dict, oop_IDs = mess_up(controller, include_classes, num_objects)
            objects_messup = controller.last_event.metadata['objects']
            save_dict = {'object_dict':object_dict, 'oop_IDs':oop_IDs, 'objects_messup':objects_messup}

            pickle_fname = f'{n}.p'
            fname_ = os.path.join(root, pickle_fname)

            print("saving", fname_)
            with open(fname_, 'wb') as f:
                pickle.dump(save_dict, f, protocol=4)
            print("done.")

    for mapname in mapnames_test:
        print(mapname)
        root = os.path.join(args.mess_up_dir, mapname)
        if not os.path.exists(root):
            os.mkdir(root)
        for n in range(args.n_test_messup):
            controller.reset(scene=mapname)
            object_dict, oop_IDs = mess_up(controller, include_classes, num_objects)
            objects_messup = controller.last_event.metadata['objects']
            save_dict = {'object_dict':object_dict, 'oop_IDs':oop_IDs, 'objects_messup':objects_messup}

            pickle_fname = f'{n}.p'
            fname_ = os.path.join(root, pickle_fname)

            print("saving", fname_)
            with open(fname_, 'wb') as f:
                pickle.dump(save_dict, f, protocol=4)
            print("done.")


if __name__ == '__main__':
    import pickle
    # pickle_fname = 
    fname = './data/messup/FloorPlan1/0.p' #os.path.join(root, pickle_fname)
    print("loading", fname)
    with open(fname, 'rb') as f:
        load_dict = pickle.load(f)
    st()
    


