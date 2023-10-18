from ai2thor.controller import Controller
import numpy as np
import os
# import hyperparams as hyp
from arguments import args
from argparse import Namespace
from tensorboardX import SummaryWriter
from ai2thor_docker.ai2thor_docker.x_server import startx
import ipdb
st = ipdb.set_trace

class Base():

    def __init__(self):   

        self.W = args.W #480
        self.H = args.H #480

        self.X = args.X #128
        self.Y = args.Y #4 # height of map (floor to ceiling)
        self.Z = args.Z #128

        num_mem_houses = args.num_mem_houses
        num_train_houses = args.num_train_houses
        buffer = 0 #args.buffer
        num_test_houses = args.num_test_houses
        num_val_houses = args.num_val_houses

        ############## Get memory houses ##############
        # memory house is 1-15
        a_h = np.arange(1, 1+num_mem_houses)
        b_h = np.arange(201, 201+num_mem_houses)
        c_h = np.arange(301, 301+num_mem_houses)
        d_h = np.arange(401, 401+num_mem_houses)
        abcd = np.hstack((a_h,b_h,c_h,d_h))

        mapnames = []
        housesets = []
        for i in range(a_h.shape[0]):
            houseset = []
            if args.do_kitchen:
                mapname = 'FloorPlan' + str(a_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            if args.do_living_room:
                mapname = 'FloorPlan' + str(b_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            if args.do_bedroom:
                mapname = 'FloorPlan' + str(c_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            if args.do_bathroom:
                mapname = 'FloorPlan' + str(d_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            housesets.append(houseset)

        self.mapnames_mem = mapnames
        self.housesets_mem = housesets

        ############## Get training houses ##############
        # training house is 16-20
        a_h = np.arange(1+num_mem_houses, 1+num_mem_houses+num_train_houses)
        b_h = np.arange(201+num_mem_houses, 201+num_mem_houses+num_train_houses)
        c_h = np.arange(301+num_mem_houses, 301+num_mem_houses+num_train_houses)
        d_h = np.arange(401+num_mem_houses, 401+num_mem_houses+num_train_houses)
        abcd = np.hstack((a_h,b_h,c_h,d_h))

        mapnames = []
        housesets = []
        for i in range(a_h.shape[0]):
            houseset = []
            if args.do_kitchen:
                mapname = 'FloorPlan' + str(a_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            if args.do_living_room:
                mapname = 'FloorPlan' + str(b_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            if args.do_bedroom:
                mapname = 'FloorPlan' + str(c_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            if args.do_bathroom:
                mapname = 'FloorPlan' + str(d_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            housesets.append(houseset)

        self.mapnames_train = mapnames
        self.housesets_train = housesets

        ############## Get validation houses ##############
        # validation houses
        a_h = np.arange(1+num_mem_houses+num_train_houses, 1+num_mem_houses+num_train_houses+num_val_houses)
        b_h = np.arange(201+num_mem_houses+num_train_houses, 201+num_mem_houses+num_train_houses+num_val_houses)
        c_h = np.arange(301+num_mem_houses+num_train_houses, 301+num_mem_houses+num_train_houses+num_val_houses)
        d_h = np.arange(401+num_mem_houses+num_train_houses, 401+num_mem_houses+num_train_houses+num_val_houses)
        abcd = np.hstack((a_h,b_h,c_h,d_h))

        mapnames = []
        housesets = []
        for i in range(a_h.shape[0]):
            houseset = []
            if args.do_kitchen:
                mapname = 'FloorPlan' + str(a_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            if args.do_living_room:
                mapname = 'FloorPlan' + str(b_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            if args.do_bedroom:
                mapname = 'FloorPlan' + str(c_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            if args.do_bathroom:
                mapname = 'FloorPlan' + str(d_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            housesets.append(houseset)

        self.mapnames_val = mapnames
        self.housesets_val = housesets

        ############## Get testing houses ##############
        # get rest of the houses in orders
        a_h = np.arange(1+num_mem_houses+buffer+num_train_houses+num_val_houses, 1+num_mem_houses+num_train_houses+buffer+num_test_houses+num_val_houses)
        b_h = np.arange(201+num_mem_houses+buffer+num_train_houses+num_val_houses, 201+num_mem_houses+num_train_houses+buffer+num_test_houses+num_val_houses)
        c_h = np.arange(301+num_mem_houses+buffer+num_train_houses+num_val_houses, 301+num_mem_houses+num_train_houses+buffer+num_test_houses+num_val_houses)
        d_h = np.arange(401+num_mem_houses+buffer+num_train_houses+num_val_houses, 401+num_mem_houses+num_train_houses+buffer+num_test_houses+num_val_houses)
        abcd = np.hstack((a_h,b_h,c_h,d_h))

        mapnames = []
        housesets = []
        for i in range(a_h.shape[0]):
            houseset = []
            if args.do_kitchen:
                mapname = 'FloorPlan' + str(a_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            if args.do_living_room:
                mapname = 'FloorPlan' + str(b_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            if args.do_bedroom:
                mapname = 'FloorPlan' + str(c_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            if args.do_bathroom:
                mapname = 'FloorPlan' + str(d_h[i])
                mapnames.append(mapname)
                houseset.append(mapname)
            housesets.append(houseset)

        self.mapnames_test = mapnames
        self.housesets_test = housesets

        self.class_agnostic = False
        self.random_select = True

        self.include_classes = [
            'ShowerDoor', 'Cabinet', 'CounterTop', 'Sink', 'Towel', 'HandTowel', 'TowelHolder', 'SoapBar', 
            'ToiletPaper', 'ToiletPaperHanger', 'HandTowelHolder', 'SoapBottle', 'GarbageCan', 'Candle', 'ScrubBrush', 
            'Plunger', 'SinkBasin', 'Cloth', 'SprayBottle', 'Toilet', 'Faucet', 'ShowerHead', 'Box', 'Bed', 'Book', 
            'DeskLamp', 'BasketBall', 'Pen', 'Pillow', 'Pencil', 'CellPhone', 'KeyChain', 'Painting', 'CreditCard', 
            'AlarmClock', 'CD', 'Laptop', 'Drawer', 'SideTable', 'Chair', 'Blinds', 'Desk', 'Curtains', 'Dresser', 
            'Watch', 'Television', 'WateringCan', 'Newspaper', 'FloorLamp', 'RemoteControl', 'HousePlant', 'Statue', 
            'Ottoman', 'ArmChair', 'Sofa', 'DogBed', 'BaseballBat', 'TennisRacket', 'VacuumCleaner', 'Mug', 'ShelvingUnit', 
            'Shelf', 'StoveBurner', 'Apple', 'Lettuce', 'Bottle', 'Egg', 'Microwave', 'CoffeeMachine', 'Fork', 'Fridge', 
            'WineBottle', 'Spatula', 'Bread', 'Tomato', 'Pan', 'Cup', 'Pot', 'SaltShaker', 'Potato', 'PepperShaker', 
            'ButterKnife', 'StoveKnob', 'Toaster', 'DishSponge', 'Spoon', 'Plate', 'Knife', 'DiningTable', 'Bowl', 
            'LaundryHamper', 'Vase', 'Stool', 'CoffeeTable', 'Poster', 'Bathtub', 'TissueBox', 'Footstool', 'BathtubBasin', 
            'ShowerCurtain', 'TVStand', 'Boots', 'RoomDecor', 'PaperTowelRoll', 'Ladle', 'Kettle', 'Safe', 'GarbageBag', 'TeddyBear', 
            'TableTopDecor', 'Dumbbell', 'Desktop', 'AluminumFoil', 'Window', 'LightSwitch']
        # self.include_classes.append('no_object') # ddetr has no object class
        if args.use_solq:
            self.special_classes = ['AppleSliced', 'BreadSliced', 'EggCracked', 'LettuceSliced', 'PotatoSliced', 'TomatoSliced']
            self.include_classes += self.special_classes
        else:
            self.special_classes = []
        self.include_classes.append('no_object') # ddetr has no object class

        self.traj_classes = ['Bed','Sofa','Laptop', 'CoffeeTable', 'FloorLamp', 'TVStand', 'Pillow', 'ArmChair', 'HousePlant', 'Painting', 'Dresser', 'Desk']

        self.REARRANGE_SIM_OBJECTS, self.OBJECT_TYPES_WITH_PROPERTIES, self.PICKUPABLE_OBJECTS, self.OPENABLE_OBJECTS = get_rearrangement_categories()

        self.receptacles = ['Cabinet', 'CounterTop', 'Sink', 'TowelHolder',
            'GarbageCan', 
            'SinkBasin', 'Bed', 
            'Drawer', 'SideTable', 'Chair', 'Desk', 'Dresser',  
            'Ottoman', 'ArmChair', 'Sofa', 'DogBed', 'ShelvingUnit', 
            'Shelf', 'StoveBurner', 'Microwave', 'CoffeeMachine', 'Fridge', 
            'Toaster', 'DiningTable',  
            'LaundryHamper', 'Stool', 'CoffeeTable', 'Bathtub', 'Footstool', 'BathtubBasin', 
            'TVStand', 'Safe']

        self.z_params = navigation_calibration_params()

        self.classes_to_save = self.include_classes 
        self.class_to_save_to_id = {self.classes_to_save[i]:i for i in range(len(self.classes_to_save))}

        self.name_to_id = {}
        self.id_to_name = {}
        self.instance_counter = {}
        idx = 0
        for name in self.include_classes:
            self.name_to_id[name] = idx
            self.id_to_name[idx] = name
            self.instance_counter[name] = 0
            idx += 1

        if args.use_solq:
            print("Note: mapping sliced classes to non-sliced in SOLQ")
            self.name_to_mapped_name = {'AppleSliced':'Apple', 'BreadSliced':'Bread', 'EggCracked':'Egg', 'LettuceSliced':'Lettuce', 'PotatoSliced':'Potato', 'TomatoSliced':'Tomato'}
            self.id_to_mapped_id = {self.name_to_id[k]:self.name_to_id[v] for k, v in self.name_to_mapped_name.items()}
        else:
            self.id_to_mapped_id = None
                
        self.set_name = args.set_name #'test'
        print("set name=", self.set_name)
        self.data_path = args.data_path  #f'.'
        self.checkpoint_root = self.data_path + '/checkpoints'
        if not os.path.exists(self.checkpoint_root):
            os.mkdir(self.checkpoint_root)
        self.log_dir = '.' + '/tb' + '/' + self.set_name
        self.checkpoint_path = self.checkpoint_root + '/' + self.set_name
        
        if self.set_name != 'test00':
            if not os.path.exists(self.checkpoint_path):
                os.mkdir(self.checkpoint_path)
            else:
                print(self.checkpoint_path)
                val = input("Path exists. Delete folder? [y/n]: ")
                if val == 'y':
                    import shutil
                    shutil.rmtree(self.checkpoint_path)
                    os.mkdir(self.checkpoint_path)
                else:
                    print("ENDING")
                    assert(False)

            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
            else:
                print(self.log_dir)
                val = input("Path exists. Delete folder? [y/n]: ")
                if val == 'y':
                    for filename in os.listdir(self.log_dir):
                        file_path = os.path.join(self.log_dir, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            print('Failed to delete %s. Reason: %s' % (file_path, e))
                else:
                    print("ENDING")
                    assert(False)


        self.fov = args.fov


        actions = ['MoveAhead', 'MoveLeft', 'MoveRight', 'MoveBack', 'RotateRight', 'RotateLeft']

        self.actions = {i:actions[i] for i in range(len(actions))}

        self.action_mapping = {"Pass":0, "MoveAhead":1, "MoveLeft":2, "MoveRight":3, "MoveBack":4, "RotateRight":5, "RotateLeft":6, "LookUp":9, "LookDown":10}

        hfov = float(self.fov) * np.pi / 180.
        self.pix_T_camX = np.array([
            [(self.W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., (self.H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.pix_T_camX[0,2] = self.W/2.
        self.pix_T_camX[1,2] = self.H/2.

        self.agent_height = 1.5759992599487305

        # fix these
        self.STEP_SIZE = args.STEP_SIZE #0.25 # move step size
        self.DT = args.DT #45 # yaw rotation degrees
        self.HORIZON_DT = args.HORIZON_DT #30 # pitch rotation degrees

        self.obs = Namespace()
        self.obs.STEP_SIZE = self.STEP_SIZE
        self.obs.DT = self.DT
        self.obs.HORIZON_DT = self.HORIZON_DT

        self.visibilityDistance = args.visibilityDistance

        self.obs.camera_height = 1.5759992599487305
        self.obs.camera_aspect_ratio = [self.H, self.W]
        self.obs.camera_field_of_view = self.fov
        self.obs.head_tilt = 0.0 #event.metadata['agent']['cameraHorizon']
        self.obs.reward = 0
        self.obs.goal = Namespace(metadata={"category": 'cover'})

        self.writer = SummaryWriter(self.log_dir, max_queue=args.MAX_QUEUE, flush_secs=60)

        self.num_save_traj = 10    

        if not args.dont_use_controller:
            self.controller, self.server_port = init_controller(
                self.STEP_SIZE,
                self.W,
                self.H,
                self.fov,
                self.DT,
                args.visibilityDistance,
                )

def init_controller(
    STEP_SIZE,
    W,
    H,
    fov,
    DT,
    visibilityDistance,
    ):

    if args.start_startx:
        if not (args.mode=="rearrangement"):
            server_port = startx()
            print("SERVER PORT=", server_port)
            controller = Controller(
                    # scene=mapname, 
                    visibilityDistance=visibilityDistance,
                    gridSize=STEP_SIZE,
                    width=W,
                    height=H,
                    fieldOfView=fov,
                    renderObjectImage=True,
                    renderDepthImage=True,
                    renderInstanceSegmentation=True,
                    x_display=str(server_port),
                    snapToGrid=False,
                    rotateStepDegrees=DT,
                    )
        else:
            server_port = startx()
            args.server_port = server_port
            print("SERVER PORT=", server_port)
            return None, None
    else:
        if not (args.mode=="rearrangement"):
            # server_port = 3
            if args.do_headless_rendering:
                from ai2thor.platform import CloudRendering
                controller = Controller(
                        # scene=mapname, 
                        visibilityDistance=visibilityDistance,
                        gridSize=STEP_SIZE,
                        width=W,
                        height=H,
                        fieldOfView=fov,
                        renderObjectImage=True,
                        renderDepthImage=True,
                        renderInstanceSegmentation=True,
                        # x_display=str(server_port),
                        snapToGrid=False,
                        rotateStepDegrees=DT,
                        platform=CloudRendering,
                        )
            else:
                controller = Controller(
                        # scene=mapname, 
                        visibilityDistance=visibilityDistance,
                        gridSize=STEP_SIZE,
                        width=W,
                        height=H,
                        fieldOfView=fov,
                        renderObjectImage=True,
                        renderDepthImage=True,
                        renderInstanceSegmentation=True,
                        # x_display=str(server_port),
                        snapToGrid=False,
                        rotateStepDegrees=DT,
                        )
        else:
            return None, None

    
    return controller, server_port


def get_rearrangement_categories():

    REARRANGE_SIM_OBJECTS = [
        # A
        "AlarmClock", "AluminumFoil", "Apple", "AppleSliced", "ArmChair",
        "BaseballBat", "BasketBall", "Bathtub", "BathtubBasin", "Bed", "Blinds", "Book", "Boots", "Bottle", "Bowl", "Box",
        # B
        "Bread", "BreadSliced", "ButterKnife",
        # C
        "Cabinet", "Candle", "CD", "CellPhone", "Chair", "Cloth", "CoffeeMachine", "CoffeeTable", "CounterTop", "CreditCard",
        "Cup", "Curtains",
        # D
        "Desk", "DeskLamp", "Desktop", "DiningTable", "DishSponge", "DogBed", "Drawer", "Dresser", "Dumbbell",
        # E
        "Egg", "EggCracked",
        # F
        "Faucet", "Floor", "FloorLamp", "Footstool", "Fork", "Fridge",
        # G
        "GarbageBag", "GarbageCan",
        # H
        "HandTowel", "HandTowelHolder", "HousePlant", "Kettle", "KeyChain", "Knife",
        # L
        "Ladle", "Laptop", "LaundryHamper", "Lettuce", "LettuceSliced", "LightSwitch",
        # M
        "Microwave", "Mirror", "Mug",
        # N
        "Newspaper",
        # O
        "Ottoman",
        # P
        "Painting", "Pan", "PaperTowel", "Pen", "Pencil", "PepperShaker", "Pillow", "Plate", "Plunger", "Poster", "Pot",
        "Potato", "PotatoSliced",
        # R
        "RemoteControl", "RoomDecor",
        # S
        "Safe", "SaltShaker", "ScrubBrush", "Shelf", "ShelvingUnit", "ShowerCurtain", "ShowerDoor", "ShowerGlass",
        "ShowerHead", "SideTable", "Sink", "SinkBasin", "SoapBar", "SoapBottle", "Sofa", "Spatula", "Spoon", "SprayBottle",
        "Statue", "Stool", "StoveBurner", "StoveKnob",
        # T
        "TableTopDecor", "TargetCircle", "TeddyBear", "Television", "TennisRacket", "TissueBox", "Toaster", "Toilet",
        "ToiletPaper", "ToiletPaperHanger", "Tomato", "TomatoSliced", "Towel", "TowelHolder", "TVStand",
        # V
        "VacuumCleaner", "Vase",
        # W
        "Watch", "WateringCan", "Window", "WineBottle",
    ]
    # fmt: on


    # fmt: off
    OBJECT_TYPES_WITH_PROPERTIES = {
        "StoveBurner": {"openable": False, "receptacle": True, "pickupable": False},
        "Drawer": {"openable": True, "receptacle": True, "pickupable": False},
        "CounterTop": {"openable": False, "receptacle": True, "pickupable": False},
        "Cabinet": {"openable": True, "receptacle": True, "pickupable": False},
        "StoveKnob": {"openable": False, "receptacle": False, "pickupable": False},
        "Window": {"openable": False, "receptacle": False, "pickupable": False},
        "Sink": {"openable": False, "receptacle": True, "pickupable": False},
        "Floor": {"openable": False, "receptacle": True, "pickupable": False},
        "Book": {"openable": True, "receptacle": False, "pickupable": True},
        "Bottle": {"openable": False, "receptacle": False, "pickupable": True},
        "Knife": {"openable": False, "receptacle": False, "pickupable": True},
        "Microwave": {"openable": True, "receptacle": True, "pickupable": False},
        "Bread": {"openable": False, "receptacle": False, "pickupable": True},
        "Fork": {"openable": False, "receptacle": False, "pickupable": True},
        "Shelf": {"openable": False, "receptacle": True, "pickupable": False},
        "Potato": {"openable": False, "receptacle": False, "pickupable": True},
        "HousePlant": {"openable": False, "receptacle": False, "pickupable": False},
        "Toaster": {"openable": False, "receptacle": True, "pickupable": False},
        "SoapBottle": {"openable": False, "receptacle": False, "pickupable": True},
        "Kettle": {"openable": True, "receptacle": False, "pickupable": True},
        "Pan": {"openable": False, "receptacle": True, "pickupable": True},
        "Plate": {"openable": False, "receptacle": True, "pickupable": True},
        "Tomato": {"openable": False, "receptacle": False, "pickupable": True},
        "Vase": {"openable": False, "receptacle": False, "pickupable": True},
        "GarbageCan": {"openable": False, "receptacle": True, "pickupable": False},
        "Egg": {"openable": False, "receptacle": False, "pickupable": True},
        "CreditCard": {"openable": False, "receptacle": False, "pickupable": True},
        "WineBottle": {"openable": False, "receptacle": False, "pickupable": True},
        "Pot": {"openable": False, "receptacle": True, "pickupable": True},
        "Spatula": {"openable": False, "receptacle": False, "pickupable": True},
        "PaperTowelRoll": {"openable": False, "receptacle": False, "pickupable": True},
        "Cup": {"openable": False, "receptacle": True, "pickupable": True},
        "Fridge": {"openable": True, "receptacle": True, "pickupable": False},
        "CoffeeMachine": {"openable": False, "receptacle": True, "pickupable": False},
        "Bowl": {"openable": False, "receptacle": True, "pickupable": True},
        "SinkBasin": {"openable": False, "receptacle": True, "pickupable": False},
        "SaltShaker": {"openable": False, "receptacle": False, "pickupable": True},
        "PepperShaker": {"openable": False, "receptacle": False, "pickupable": True},
        "Lettuce": {"openable": False, "receptacle": False, "pickupable": True},
        "ButterKnife": {"openable": False, "receptacle": False, "pickupable": True},
        "Apple": {"openable": False, "receptacle": False, "pickupable": True},
        "DishSponge": {"openable": False, "receptacle": False, "pickupable": True},
        "Spoon": {"openable": False, "receptacle": False, "pickupable": True},
        "LightSwitch": {"openable": False, "receptacle": False, "pickupable": False},
        "Mug": {"openable": False, "receptacle": True, "pickupable": True},
        "ShelvingUnit": {"openable": False, "receptacle": True, "pickupable": False},
        "Statue": {"openable": False, "receptacle": False, "pickupable": True},
        "Stool": {"openable": False, "receptacle": True, "pickupable": False},
        "Faucet": {"openable": False, "receptacle": False, "pickupable": False},
        "Ladle": {"openable": False, "receptacle": False, "pickupable": True},
        "CellPhone": {"openable": False, "receptacle": False, "pickupable": True},
        "Chair": {"openable": False, "receptacle": True, "pickupable": False},
        "SideTable": {"openable": False, "receptacle": True, "pickupable": False},
        "DiningTable": {"openable": False, "receptacle": True, "pickupable": False},
        "Pen": {"openable": False, "receptacle": False, "pickupable": True},
        "SprayBottle": {"openable": False, "receptacle": False, "pickupable": True},
        "Curtains": {"openable": False, "receptacle": False, "pickupable": False},
        "Pencil": {"openable": False, "receptacle": False, "pickupable": True},
        "Blinds": {"openable": True, "receptacle": False, "pickupable": False},
        "GarbageBag": {"openable": False, "receptacle": False, "pickupable": False},
        "Safe": {"openable": True, "receptacle": True, "pickupable": False},
        "Painting": {"openable": False, "receptacle": False, "pickupable": False},
        "Box": {"openable": True, "receptacle": True, "pickupable": True},
        "Laptop": {"openable": True, "receptacle": False, "pickupable": True},
        "Television": {"openable": False, "receptacle": False, "pickupable": False},
        "TissueBox": {"openable": False, "receptacle": False, "pickupable": True},
        "KeyChain": {"openable": False, "receptacle": False, "pickupable": True},
        "FloorLamp": {"openable": False, "receptacle": False, "pickupable": False},
        "DeskLamp": {"openable": False, "receptacle": False, "pickupable": False},
        "Pillow": {"openable": False, "receptacle": False, "pickupable": True},
        "RemoteControl": {"openable": False, "receptacle": False, "pickupable": True},
        "Watch": {"openable": False, "receptacle": False, "pickupable": True},
        "Newspaper": {"openable": False, "receptacle": False, "pickupable": True},
        "ArmChair": {"openable": False, "receptacle": True, "pickupable": False},
        "CoffeeTable": {"openable": False, "receptacle": True, "pickupable": False},
        "TVStand": {"openable": False, "receptacle": True, "pickupable": False},
        "Sofa": {"openable": False, "receptacle": True, "pickupable": False},
        "WateringCan": {"openable": False, "receptacle": False, "pickupable": True},
        "Boots": {"openable": False, "receptacle": False, "pickupable": True},
        "Ottoman": {"openable": False, "receptacle": True, "pickupable": False},
        "Desk": {"openable": False, "receptacle": True, "pickupable": False},
        "Dresser": {"openable": False, "receptacle": True, "pickupable": False},
        "Mirror": {"openable": False, "receptacle": False, "pickupable": False},
        "DogBed": {"openable": False, "receptacle": True, "pickupable": False},
        "Candle": {"openable": False, "receptacle": False, "pickupable": True},
        "RoomDecor": {"openable": False, "receptacle": False, "pickupable": False},
        "Bed": {"openable": False, "receptacle": True, "pickupable": False},
        "BaseballBat": {"openable": False, "receptacle": False, "pickupable": True},
        "BasketBall": {"openable": False, "receptacle": False, "pickupable": True},
        "AlarmClock": {"openable": False, "receptacle": False, "pickupable": True},
        "CD": {"openable": False, "receptacle": False, "pickupable": True},
        "TennisRacket": {"openable": False, "receptacle": False, "pickupable": True},
        "TeddyBear": {"openable": False, "receptacle": False, "pickupable": True},
        "Poster": {"openable": False, "receptacle": False, "pickupable": False},
        "Cloth": {"openable": False, "receptacle": False, "pickupable": True},
        "Dumbbell": {"openable": False, "receptacle": False, "pickupable": True},
        "LaundryHamper": {"openable": True, "receptacle": True, "pickupable": False},
        "TableTopDecor": {"openable": False, "receptacle": False, "pickupable": True},
        "Desktop": {"openable": False, "receptacle": False, "pickupable": False},
        "Footstool": {"openable": False, "receptacle": True, "pickupable": True},
        "BathtubBasin": {"openable": False, "receptacle": True, "pickupable": False},
        "ShowerCurtain": {"openable": True, "receptacle": False, "pickupable": False},
        "ShowerHead": {"openable": False, "receptacle": False, "pickupable": False},
        "Bathtub": {"openable": False, "receptacle": True, "pickupable": False},
        "Towel": {"openable": False, "receptacle": False, "pickupable": True},
        "HandTowel": {"openable": False, "receptacle": False, "pickupable": True},
        "Plunger": {"openable": False, "receptacle": False, "pickupable": True},
        "TowelHolder": {"openable": False, "receptacle": True, "pickupable": False},
        "ToiletPaperHanger": {"openable": False, "receptacle": True, "pickupable": False},
        "SoapBar": {"openable": False, "receptacle": False, "pickupable": True},
        "ToiletPaper": {"openable": False, "receptacle": False, "pickupable": True},
        "HandTowelHolder": {"openable": False, "receptacle": True, "pickupable": False},
        "ScrubBrush": {"openable": False, "receptacle": False, "pickupable": True},
        "Toilet": {"openable": True, "receptacle": True, "pickupable": False},
        "ShowerGlass": {"openable": False, "receptacle": False, "pickupable": False},
        "ShowerDoor": {"openable": True, "receptacle": False, "pickupable": False},
        "AluminumFoil": {"openable": False, "receptacle": False, "pickupable": True},
        "VacuumCleaner": {"openable": False, "receptacle": False, "pickupable": False}
    }
    # fmt: on

    PICKUPABLE_OBJECTS = list(
        sorted(
            [
                object_type
                for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
                if properties["pickupable"]
            ]
        )
    )

    OPENABLE_OBJECTS = list(
        sorted(
            [
                object_type
                for object_type, properties in OBJECT_TYPES_WITH_PROPERTIES.items()
                if properties["openable"] and not properties["pickupable"]
            ]
        )
    )

    return REARRANGE_SIM_OBJECTS, OBJECT_TYPES_WITH_PROPERTIES, PICKUPABLE_OBJECTS, OPENABLE_OBJECTS


def navigation_calibration_params():

    z_params = {
        # 0s
        'FloorPlan1':[0.05, 2.0],
        'FloorPlan2':[0.05, 2.0],
        'FloorPlan3':[0.05, 2.0], # this one is broken?
        'FloorPlan4':[0.05, 1.7],
        'FloorPlan5':[0.05, 1.9],
        'FloorPlan6':[0.05, 1.9],
        'FloorPlan7':[0.05, 2.5],
        'FloorPlan8':[0.05, 2.5],
        'FloorPlan9':[0.05, 2.0],
        'FloorPlan10':[0.05, 2.2],
        'FloorPlan11':[0.05, 2.0],
        'FloorPlan12':[0.05, 2.1],
        'FloorPlan13':[0.05, 2.7],
        'FloorPlan14':[0.05, 2.2],
        'FloorPlan15':[0.05, 1.8],
        'FloorPlan16':[0.05, 2.2],
        'FloorPlan17':[0.05, 2.0],
        'FloorPlan18':[0.05, 2.0],
        'FloorPlan19':[0.05, 2.0],
        'FloorPlan20':[0.05, 2.3],
        'FloorPlan21':[0.05, 1.7],
        'FloorPlan22':[0.05, 2.2],
        'FloorPlan23':[0.05, 2.2],
        'FloorPlan24':[0.05, 2.0],
        'FloorPlan25':[0.05, 2.0],
        'FloorPlan26':[0.05, 2.2],
        'FloorPlan27':[0.05, 2.2],
        'FloorPlan28':[0.05, 1.9],
        'FloorPlan29':[0.05, 1.9],
        'FloorPlan30':[0.05, 2.0],
        # 200s
        'FloorPlan201':[0.05, 1.8],
        'FloorPlan202':[0.05, 2.4],
        'FloorPlan203':[0.05, 2.0],
        'FloorPlan204':[0.05, 2.4],
        'FloorPlan205':[0.05, 2.0],
        'FloorPlan206':[0.05, 2.0],
        'FloorPlan207':[0.05, 2.0],
        'FloorPlan208':[0.05, 2.4],
        'FloorPlan209':[0.05, 2.5],
        'FloorPlan210':[0.05, 2.4],
        'FloorPlan211':[0.05, 2.0],
        'FloorPlan212':[0.05, 1.7],
        'FloorPlan213':[0.05, 2.1],
        'FloorPlan214':[0.05, 2.5],
        'FloorPlan215':[0.05, 2.2],
        'FloorPlan216':[0.05, 2.1],
        'FloorPlan217':[0.05, 2.1],
        'FloorPlan218':[0.05, 2.0],
        'FloorPlan219':[0.05, 2.0],
        'FloorPlan220':[0.05, 2.0],
        'FloorPlan221':[0.05, 2.0], # broken
        'FloorPlan221':[0.05, 2.1],
        'FloorPlan222':[0.05, 2.1],
        'FloorPlan223':[0.05, 2.2],
        'FloorPlan224':[0.05, 2.2],
        'FloorPlan225':[0.05, 2.0],
        'FloorPlan226':[0.05, 2.0],
        'FloorPlan227':[0.05, 2.5],
        'FloorPlan228':[0.05, 2.5],
        'FloorPlan229':[0.05, 2.4],
        'FloorPlan230':[0.05, 2.5], # needs finetuning
        # 300s
        'FloorPlan301':[0.05, 2.0],
        'FloorPlan302':[0.05, 2.1],
        'FloorPlan303':[0.05, 2.0],
        'FloorPlan304':[0.05, 2.4],
        'FloorPlan305':[0.05, 2.0],
        'FloorPlan306':[0.05, 2.1],
        'FloorPlan307':[0.05, 2.2],
        'FloorPlan308':[0.05, 2.0],
        'FloorPlan309':[0.05, 2.0],
        'FloorPlan310':[0.05, 2.0],
        'FloorPlan311':[0.05, 2.0],
        'FloorPlan312':[0.05, 2.0],
        'FloorPlan313':[0.05, 2.0],
        'FloorPlan314':[0.05, 1.8],
        'FloorPlan315':[0.05, 1.9],
        'FloorPlan316':[0.05, 2.0],
        'FloorPlan317':[0.05, 2.2],
        'FloorPlan318':[0.05, 2.2],
        'FloorPlan319':[0.05, 2.3],
        'FloorPlan320':[0.05, 2.0],
        'FloorPlan321':[0.05, 1.9],
        'FloorPlan322':[0.05, 2.1],
        'FloorPlan323':[0.05, 2.0],
        'FloorPlan324':[0.05, 2.0],
        'FloorPlan325':[0.05, 2.0], # needs finetuning/is broken
        'FloorPlan326':[0.05, 2.2],
        'FloorPlan327':[0.05, 2.0],
        'FloorPlan328':[0.05, 2.0],
        'FloorPlan329':[0.05, 2.0],
        'FloorPlan330':[0.05, 2.0],
        # 400s
        'FloorPlan401':[0.05, 2.0],
        'FloorPlan402':[0.05, 1.9],
        'FloorPlan403':[0.05, 2.1],
        'FloorPlan404':[0.05, 2.0],
        'FloorPlan405':[0.05, 2.0],
        'FloorPlan406':[0.05, 2.0],
        'FloorPlan407':[0.05, 2.0],
        'FloorPlan408':[0.05, 2.0],
        'FloorPlan409':[0.05, 2.0],
        'FloorPlan410':[0.05, 2.0],
        'FloorPlan411':[0.05, 2.0],
        'FloorPlan412':[0.05, 2.0],
        'FloorPlan413':[0.05, 2.0],
        'FloorPlan414':[0.05, 2.0],
        'FloorPlan415':[0.05, 2.5],
        'FloorPlan416':[0.05, 2.5],
        'FloorPlan417':[0.05, 2.0],
        'FloorPlan418':[0.05, 2.0],
        'FloorPlan419':[0.05, 2.0],
        'FloorPlan420':[0.05, 2.0],
        'FloorPlan421':[0.05, 2.0],
        'FloorPlan422':[0.05, 2.5],
        'FloorPlan423':[0.05, 2.0],
        'FloorPlan424':[0.05, 2.4],
        'FloorPlan425':[0.05, 2.0],
        'FloorPlan426':[0.05, 2.0],
        'FloorPlan427':[0.05, 2.0],
        'FloorPlan428':[0.05, 2.1],
        'FloorPlan429':[0.05, 1.9],
        'FloorPlan430':[0.05, 2.5],
    }

    return z_params
