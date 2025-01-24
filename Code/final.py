import sys
import numpy as np
from copy import deepcopy
from math import pi

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

from grabber import Grabber
from placer import Placer
from search import Searcher

class Controller:

    def __init__(self,arm,team):
        self.grabber = Grabber(team, arm)
        self.searcher = Searcher(team, arm)
        self.placer = Placer(team, arm)

    def select_Static_Block(self,static_blocks):
        selected_block = ""
        # Gets names of already placed blocks
        placed_blocks = self.placer.get_placed_blocks()
        # Gets names of lost blocks (outside reachable environment)
        lost_blocks = self.get_Lost_Static_Blocks(static_blocks)
        block_names = list(static_blocks.keys())
        isPlaced = np.isin(block_names,placed_blocks)
        isLost = np.isin(block_names,lost_blocks)
        
        # Select first static block in array that is not already placed or lost
        for i in range(len(isPlaced)):
            # Check if static block matches placed block
            if(not isPlaced[i] and not isLost[i]):
                # If block has not been placed and is not lost, select it
                selected_block = block_names[i]
                return selected_block

        # All blocks have been placed
        return selected_block

    def select_Dynamic_Block(self,dynamic_blocks,team):
        selected_block = ""
        # Gets names of already placed blocks
        placed_blocks = self.placer.get_placed_blocks()
        # Gets names of lost blocks (outside reachable environment)
        lost_blocks = self.get_Lost_Dynamic_Blocks(dynamic_blocks)
        block_names = list(dynamic_blocks.keys())
        isPlaced = np.isin(block_names,placed_blocks)
        isLost = np.isin(block_names,lost_blocks)
        
        # Select dynamic block in array that is not already placed or lost
        for i in range(len(block_names)):
            # Check if static block matches placed block
            if(not isPlaced[i] and not isLost[i]):
                # If block has not been placed and is not lost, check position
                block_transform, t = dynamic_blocks[block_names[i]]
                print(f"Targeting block: {block_names[i]}________________________________________________________")
                print(f"All blocks: {block_names}")
                print(f"Block: {block_transform}")
                if team == 'blue':
                    # Sets boundary at x=0, max_x will be updated to be positive
                    #max_x = 0
                    # If block is on left/near side of turntable (x>0,y>0 for blue)
                    if(block_transform[1][3]+.99>0.05): #and block_transform[0][3]>max_x):
                        #max_x = block_transform[0][3]
                        success = controller.grab_Dynamic_Block(block_transform, t)
                        if(success):
                          # Place selected block
                          controller.place_Block(i,True)
                          return
                else:
                    # Sets boundary at x=0, min_x will be updated to be negative
                    #min_x = 0
                    # If block is on left/near side of turntable (x<0,y<0 for red)
                    if(block_transform[1][3]-.99<-0.05): #and block_transform[0][3]<min_x):
                        #selected_block = block_names[i]
                        #min_x = block_transform[0][3]
                        # Grab selected block
                        success = controller.grab_Dynamic_Block(block_transform, t)
                        if(success):
                          # Place selected block
                          controller.place_Block(i,True)
                          return
        
        # Check if block was found in near/left quadrant
        if(not selected_block == ""):
            # Near, most left block found! Select block
            return selected_block
        else:
            # No block on left/near side. Try search again
            new_dynamic_blocks = self.searcher.search_dynamic()
            # Recursive function... hopefully doesn't break our code lol
            return self.select_Dynamic_Block(new_dynamic_blocks,team)

        # All blocks have been placed or left most block is selected
        return selected_block

    def grab_Static_Block(self,block_transform):
        return self.grabber.grab_static_block(block_transform)

    def grab_Dynamic_Block(self,block_transform, t):
        return self.grabber.grab_dynamic_block(block_transform, t)

    def get_Placed_Blocks(self):
        return self.placer.get_placed_blocks()

    def get_Lost_Static_Blocks(self,static_blocks):
        lost_blocks = []
        for name in static_blocks:
            # Access current block transformation matrix for given block name
            block_transform = static_blocks[name]
            # Check if block is below horizontal barrier (z < 0.200 meters)
            if(block_transform[2][3] < 0.2):
                lost_blocks.append(name)

        return lost_blocks

    def get_Lost_Dynamic_Blocks(self,dynamic_blocks):
        lost_blocks = []
        for name in dynamic_blocks:
            # Access current block transformation matrix for given block name
            block_transform = dynamic_blocks[name][0]
            # Check if block is below horizontal barrier (z < 0.200 meters)
            if(block_transform[2][3] < 0.2):
                lost_blocks.append(name)

        return lost_blocks

    def place_Block(self, selected_block, isDynamic):
        self.placer.place_block(selected_block, isDynamic)


if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!

    controller = Controller(arm,team)
    static_Mode = True
    
    static_blocks = controller.searcher.search_static()
    while(len(static_blocks) < 4):
        # Searches for static blocks in camera view
        static_blocks = controller.searcher.search_static()
        print(f"static blocks = {static_blocks}")

    while(static_Mode):
        # Select name of static block to pick up
        selected_block = controller.select_Static_Block(static_blocks)
        # Get transformation matrix from selected block
        block_transform = static_blocks.get(selected_block)
        #block_transform[:, 1:3] *= - 1 # CHANGE THIS BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACK
        # Grab selected block
        success = controller.grab_Static_Block(block_transform)
        if(success):
          # Place selected block
          controller.place_Block(selected_block,False)

        # Checks if static blocks have all been placed or lost
        # CHANGE LOGIC BASED ON PERFORMANCE OR TIME
        if(len(controller.get_Placed_Blocks())+len(controller.get_Lost_Static_Blocks(static_blocks)) == 4):
            static_Mode = False

    dynamic_Mode = True

    while(dynamic_Mode):
        # Searches for dynamic blocks in camera view
        dynamic_blocks = controller.searcher.search_dynamic()
        print(f"dynamic blocks = {dynamic_blocks}")
        # Select name of static block to pick up
        controller.select_Dynamic_Block(dynamic_blocks,team)
        if(len(controller.get_Placed_Blocks()) == 8):
            dynamic_Mode = False
