from task_base.tidy_task import TIDEE_TASK
from ai2thor.controller import Controller
import numpy as np

'''
Class for running the full tidy task with random action
'''

class Ai2Thor():
    def __init__(self):   
        
        # example controller
        controller = Controller(
                    visibilityDistance=1.5,
                    gridSize=0.25,
                    width=480,
                    height=480,
                    fieldOfView=90,
                    renderObjectImage=True,
                    renderDepthImage=True,
                    renderInstanceSegmentation=True,
                    # x_display=str(server_port),
                    snapToGrid=False,
                    rotateStepDegrees=90,
                    )

        self.tidee_task = TIDEE_TASK(controller, args.eval_split)

        self.actions = {
            0:"MoveAhead", 
            1:"MoveBack", 
            2:"MoveRight", 
            3:"MoveLeft", 
            4:"RotateLeft",
            5:"RotateRight",
            6:"Stand",
            7:"Crouch",
            8:"LookUp",
            9:"LookDown",
            10:"PickupObject",
            11:"PutObject",
            12:"DropObject",
            13:"Done",
            }

    def main(self):
        
        for i_task in range(self.tidee_task.num_episodes_total):

            episode_name = self.tidee_task.get_episode_name()
            
            print(f'Starting episode {episode_name}')

            self.tidee_task.start_next_episode()

            while not self.tidee_task.is_done():
                # take random action
                action_ind = np.arange(13)
                action = self.actions[action_ind]
                if action in ["PickupObject", "PutObject"]:
                    # randomly select interact point
                    obj_relative_coord = np.random.uniform(0,1,size=2) 
                else:
                    obj_relative_coord = None
                self.tidee_task.step(action, obj_relative_coord)
            
            '''
            save out evaluation images 
            Note: render requires args.save_object_images turned on
            saves to args.image_dir
            '''
            self.tidee_task.render_episode_images()
                
            


                


if __name__ == '__main__':
    Ai2Thor()
        
    


