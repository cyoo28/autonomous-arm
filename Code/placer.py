import numpy as np
from math import pi
from math import sin
from math import cos
import rospy
from core.interfaces import ArmController
from core.interfaces import ObjectDetector
from core.utils import time_in_seconds
from lib.calculateIK6 import IK
from copy import deepcopy
from lib.calculateFK import FK

"""
   joint_limits = [{'lower': -2.8973, 'upper': 2.8973}, 
                    {'lower': -1.7628, 'upper': 1.7628}, 
                    {'lower': -2.8973, 'upper': 2.8973}, 
                    {'lower': -3.0718, 'upper': -0.0698}, 
                    {'lower': -2.8973, 'upper': 2.8973}, 
                    {'lower': -0.0175, 'upper': 3.7525}, 
                    {'lower': -2.8973, 'upper': 2.8973}]
"""

t = time_in_seconds

class Placer:

    #FROM THE TOP AND SIDE
    def __init__(self,team,arm):
        #Create an empty list of placed blocks to keep track of pile
        self.IK = IK()
        self.FK = FK()
        self.arm = arm
        self.team = team
        self.placed_blocks=[]
    
    def place_block(self,block_tag,isDynamic):
        i = len(self.placed_blocks)
        #Static
        if not isDynamic:            
            #Rise to hover
            joints,current_pos = self.FK.forward(self.arm.get_positions())
            hover_pos = current_pos
            hover_pos[2,3]+=0.05*(i+1)
            dict_hover = {'R': hover_pos[0:3, 0:3], 't': hover_pos[0:3, 3]}
            hover_angles = self.IK.panda_ik(dict_hover)
            self.arm.safe_move_to_position(hover_angles[0])

            #Go to safe position
            if self.team == 'red':
                #Red Team
                safe_pos = np.array([[1,0,0,.5],[0,-1,0,.1],[0,0,-1,.23+0.06*(i+1)],[0,0,0,1]])
            else:
                #Blue Team
                safe_pos = np.array([[1,0,0,.5],[0,-1,0,-.1],[0,0,-1,.23+0.06*(i+1)],[0,0,0,1]])
            dict_safe = {'R': safe_pos[0:3, 0:3], 't': safe_pos[0:3, 3]}
            safe_angles = self.IK.panda_ik(dict_safe)

            self.arm.safe_move_to_position(safe_angles[0])

            #Solving for the block place position
            #Target relative to base frame of robot
            if self.team == 'red':
                #Red Team
                desired_target = np.array([[1,0,0,.5],[0,-1,0,.1],[0,0,-1,.23+0.05*i],[0,0,0,1]])
            else:
                #Blue Team
                desired_target = np.array([[1,0,0,.5],[0,-1,0,-.1],[0,0,-1,.23+0.05*i],[0,0,0,1]])
            dict_target = {'R': desired_target[0:3, 0:3], 't': desired_target[0:3, 3]}
            #Placing the block
            place_angles = self.IK.panda_ik(dict_target)
            self.arm.safe_move_to_position(place_angles[0])
            self.arm.exec_gripper_cmd(0.08,10)
            self.placed_blocks.append(block_tag)
            #Return to safe position
            self.arm.safe_move_to_position(safe_angles[0])
        #Dynamic
        else:
            #Rise to hover
            joints,current_pos = self.FK.forward(self.arm.get_positions())
            hover_pos = current_pos
            hover_pos[0,3]=0

            if self.team == 'red':
                hover_pos[1,3]=0.5
            else:
                hover_pos[1,3]=-0.5
            
            hover_pos[2,3]+=0.05*(i+1)
            dict_hover = {'R': hover_pos[0:3, 0:3], 't': hover_pos[0:3, 3]}
            hover_angles = self.IK.panda_ik(dict_hover)
            self.arm.safe_move_to_position(hover_angles[0])
	
            #Go to safe position
            if self.team == 'red':
                #Red Team
                safe_pos = np.array([[1,0,0,.5],[0,-1,0,.1],[0,0,-1,.225+0.055*(i+1)],[0,0,0,1]])
            else:
                #Blue Team
                safe_pos = np.array([[1,0,0,.5],[0,-1,0,-.1],[0,0,-1,.225+0.055*(i+1)],[0,0,0,1]])
            dict_safe = {'R': safe_pos[0:3, 0:3], 't': safe_pos[0:3, 3]}
            safe_angles = self.IK.panda_ik(dict_safe)

            self.arm.safe_move_to_position(safe_angles[0])

            #Solving for the block place position
            #Target relative to base frame of robot
            if self.team == 'red':
                #Red Team
                desired_target = np.array([[1,0,0,.5],[0,-1,0,.1],[0,0,-1,.225+0.05*i],[0,0,0,1]])
            else:
                #Blue Team
                desired_target = np.array([[1,0,0,.5],[0,-1,0,-.1],[0,0,-1,.225+0.05*i],[0,0,0,1]])
            dict_target = {'R': desired_target[0:3, 0:3], 't': desired_target[0:3, 3]}
            #Placing the block
            place_angles = self.IK.panda_ik(dict_target)
            self.arm.safe_move_to_position(place_angles[0])
            self.arm.exec_gripper_cmd(0.08,10)
            self.placed_blocks.append(block_tag)
            #Return to safe position
            self.arm.safe_move_to_position(safe_angles[0])

    def get_placed_blocks(self):
        return deepcopy(self.placed_blocks)
    
###################################################################################
# TESTING
    
def main(): 
    rospy.init_node('ArmController', anonymous=True)
    IKsolver = IK()
    FKsolver = FK()
    arm = ArmController()
    BP = Placer('blue',arm)
    
    """
    for angle in np.arange(0,3.14,0.01).tolist():
	    desired_safe = np.array([[0,0,1,0.562],[sin(angle),-cos(angle),0,.169],[cos(angle),sin(angle),0,0.54],[0,0,0,1]])
	    dict_safe = {'R': desired_safe[0:3, 0:3], 't': desired_safe[0:3, 3]}
	    safe_angles = IKsolver.panda_ik(dict_safe)
	    if len(safe_angles)>0:
	    	print(safe_angles)
    """
    """
    # 0.562, 0.169, [0.540,0.615,0.690,0.765]
    guess_angles = np.array([0.02, -0.22,  0., -2.1, 1.5, 2, -1])
    arm.safe_move_to_position(guess_angles)
    print(FKsolver.forward(arm.get_positions()))
    """
    
    
    #Starting position
    print('moving to start')
    start_angles = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_angles)
    arm.exec_gripper_cmd(0.08,10)
    input("Press Enter to start")
    
    #Grabbing position
    print('move to grab')
    """
    #Static test
    #Red
    #desired_grab = np.array([[1,0,0,.562],[0,-1,0,-.169],[0,0,-1,.205],[0,0,0,1]])
    #BLue
    desired_grab = np.array([[1,0,0,.562],[0,-1,0,.169],[0,0,-1,.205],[0,0,0,1]])
    dict_grab = {'R': desired_grab[0:3, 0:3], 't': desired_grab[0:3, 3]}
    grab_angles = IKsolver.panda_ik(dict_grab)
    """
    #Dynamic test
    #Red
    #grab_angles = np.array([1.57, 0,  0., -2, 0, 2, -1])
    #Blue
    grab_angles = np.array([-1.57, 0,  0., -2, 0, 2, -1])
    
    for i in range(4):
        arm.safe_move_to_position(grab_angles)
        arm.exec_gripper_cmd(0.00,10)
        #Placing position
        print('place block')
        BP.place_block(0,True)
    	
    print(BP.placed_list())
	
if __name__ == '__main__':
    main()
