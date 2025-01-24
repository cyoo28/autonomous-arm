import sys
import numpy as np
from numpy import dot
from numpy.linalg import norm
from copy import deepcopy
from math import pi
import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector
# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds
# Gene added
import time
from lib.calculateFK import FK
from lib.calculateIK6 import IK

def cos_sim(a, b):
  # https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
  return dot(a, b)/(norm(a)*norm(b))

class Searcher():

    def __init__(self, team, arm):
        self.arm = arm
        self.team = team
        self.detector = ObjectDetector()
        self.staticBlocks = dict({})
        self.dynamicBlocks = dict({})
        self.angularVelocity = dict({})

    def inverseTransform(self, A):
        B = np.hstack((A[:3,:3].T,(-A[:3,:3].T@A[:3,3]).reshape((3,1))))
        B = np.vstack((B,np.array([0,0,0,1])))
        return B

    def calculateR(self, roll, pitch, yaw):

        rollMatrix = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])
        pitchMatrix = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        yawMatrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])

        R = yawMatrix @ pitchMatrix @ rollMatrix

        return R

    def calculateRPY(self, R):
        # https://stackoverflow.com/questions/11514063/extract-yaw-pitch-and-roll-from-a-rotationmatrix
        yaw=np.arctan2(R[1,0],R[0,0])
        pitch=np.arctan2(-R[2,0],(R[2,1]**2+R[2,2]**2)**0.5)
        roll=np.arctan2(R[2,1],R[2,2])
        return roll, pitch, yaw

    def alignZOrientation(self, pose):

        z_vect = np.array([0,0,1])
        z_col = self.select_vert_axis(pose)

        x_col = 0 if z_col != 0 else 1
        x_vect = pose[:3, x_col]
        x_vect[2] = 0
        x_vect /= norm(x_vect)

        y_vect = np.cross(z_vect, x_vect)
        T = np.eye(4)
        T[:3, 3] = pose[:3, 3]
        T[:3, 0] = x_vect
        T[:3, 1] = y_vect

        return T

    def select_vert_axis(self, pose):
      z = np.array([0,0,1])
      ax_alignments = np.array([abs(cos_sim(z, pose[:3, i])) for i in range(3)])
      return ax_alignments.argmax()

    def cam2base(self, q, pose):
        fk = FK()
        _, T0e = fk.forward(q)
        H_ee_camera = self.detector.get_H_ee_camera()
    
        return T0e@H_ee_camera@pose

    def cam2world(self, q, pose):
        fk = FK()
        _, T0e = fk.forward(q)
        H_ee_camera = self.detector.get_H_ee_camera()
        Tw0 = np.array([[1,0,0,0],
                        [0,1,0,0.99],
                        [0,0,1,0],
                        [0,0,0,1]])

        return Tw0@T0e@H_ee_camera@pose

    def testIK(self, blockType):
        ik = IK()

        if blockType=='static':
            for name in self.staticBlocks:
                R = self.staticBlocks[name][0:3,0:3]
                t = self.staticBlocks[name][0:3,3].reshape((3,1))
                target = {'R': R, 't': t}
                print(name)
                print(target)
                qArr = ik.panda_ik(target, filter_joint_limits=True)
                print(qArr)
                for q in qArr:
                    self.arm.safe_move_to_position(q)
                    self.arm.safe_move_to_position(q)
        else:
            for name in self.dynamicBlocks:
                R = self.dynamicBlocks[name][0:3,0:3]
                t = self.dynamicBlocks[name][0:3,3].reshape((3,1))
                target = {'R': R, 't': t}
                print(name)
                print(target)
                qArr = ik.panda_ik(target, filter_joint_limits=True)
                print(qArr)
                for q in qArr:
                    self.arm.safe_move_to_position(q)
                    self.arm.safe_move_to_position(q)

    def calculateMovement(self, blockName, q, sleepTime=5, max_iter=10):
        self.arm.safe_move_to_position(q)

        i = 0
        R11 = []
        R12 = []
        R13 = []
        R21 = []
        R22 = []
        R23 = []
        R31 = []
        R32 = []
        R33 = []
        T1 = []
        T2 = []
        T3 = []
        BOT1 = []
        BOT2 = []
        BOT3 = []
        BOT4 = []
        while(i<max_iter):
            for (name, pose) in self.detector.get_detections(): 
                if name==blockName:
                    if 'static' in name:
                        H=self.cam2base(q,pose, 'static')
                    else:
                        H=self.cam2base(q,pose, 'dynamic')
                    R11 += [H[0,0]]
                    R12 += [H[0,1]]
                    R13 += [H[0,2]]
                    R21 += [H[1,0]]
                    R22 += [H[1,1]]
                    R23 += [H[1,2]]
                    R31 += [H[2,0]]
                    R32 += [H[2,1]]
                    R33 += [H[2,2]]
                    T1 += [H[0,3]]
                    T2 += [H[1,3]]
                    T3 += [H[2,3]]
                    BOT1 += [H[3,0]]
                    BOT2 += [H[3,1]]
                    BOT3 += [H[3,2]]
                    BOT4 += [H[3,3]]
            time.sleep(sleepTime)
            i+=1

        std = np.array([[np.std(R11),np.std(R12),np.std(R13),np.std(T1)],
        [np.std(R21),np.std(R22),np.std(R23),np.std(T2)],
        [np.std(R31),np.std(R32),np.std(R33),np.std(T3)],
        [np.std(BOT1),np.std(BOT2),np.std(BOT3),np.std(BOT4)]])
        print(std)

        return std

    def calculateVelocity(self, sleepTime=10):
        # Not being used
        newDynamicBlocks = dict({})
        sumTime = 0
        sumVelocities = 0 

        if len(self.angularVelocity)!=0:
            sumTime = list(self.angularVelocity)[-1]
            sumVelocities = self.angularVelocity[sumTime]*sumTime

        # Calculating Angular Velocity
        # NOT WORKING: Simulation angular vel should be 0.03939 radians/sec
        # It returns 0.002 radians/sec
        time.sleep(sleepTime)
        for (name, pose) in self.detector.get_detections(): 
            if 'dynamic' in name:
                timeStamp = time_in_seconds()
                newDynamicBlocks[name]=(self.cam2base(q,pose),timeStamp)
        # print(newDynamicBlocks['cube5_dynamic'])

        for name in newDynamicBlocks:
            if name in self.dynamicBlocks:
                diff_time = newDynamicBlocks[name][1]-self.dynamicBlocks[name][1]
                if self.team=='red':
                    center = np.array([0, 0.99, 0.2])
                else:
                    center = np.array([0, -0.99, 0.2])

                A = newDynamicBlocks[name][0]
                B = self.dynamicBlocks[name][0]
                a = ((A[0,3]-center[0])**2 + (A[1,3]-center[1])**2)**0.5
                b = ((B[0,3]-center[0])**2 + (B[1,3]-center[1])**2)**0.5
                c = ((A[0,3]-B[0,3])**2+(A[1,3]-B[1,3])**2)**0.5
                
                diff_theta = np.arccos((c**2-a**2-b**2)/(-2*a*b))
                vel = diff_theta/diff_time

                # Alternative Method: w = dR/dt * R(t)^(-1)
                # https://gamedev.stackexchange.com/questions/189950/calculate-angular-velocity-from-rotation-matrix-difference
                # diff_R = (newDynamicBlocks[name][0]-dynamicBlocks[name][0])[:3,:3]
                # diff_time = newDynamicBlocks[name][1]-dynamicBlocks[name][1]
                # A = (diff_R/diff_time)@newDynamicBlocks[name][0][:3,:3]
                # B = np.abs(A)
                # C = np.array([(B[0,1]+B[1,0])/2, (B[0,2]+B[2,0])/2, (B[2,1]+B[1,2])/2])
                # vel = np.linalg.norm(C)

                sumVelocities += vel
                sumTime += diff_time
                
        self.angularVelocity[sumTime]=(sumVelocities/sumTime,timeStamp)
        print(self.angularVelocity)

        return self.newDynamicBlocks, self.angularVelocity

    def search_blocks_loop(self):
        while True:
            inputArr = input("\nENTER a q!\n").split(',')
            print(inputArr)
            q = np.array([inputArr[0], inputArr[1], inputArr[2], inputArr[3], inputArr[4], inputArr[5], inputArr[6]], dtype=np.float64)
            self.arm.safe_move_to_position(q)

            # Detect some blocks...
            for (name, pose) in self.detector.get_detections(): 
                if 'static' in name:
                    self.staticBlocks[name]=self.cam2base(q,pose)
                else:
                    self.dynamicBlocks[name]=(self.cam2base(q,pose),time_in_seconds())
            print(self.staticBlocks)
            print(self.dynamicBlocks)

    def search_static(self):
        if self.team=='red':
            q = np.array([-0.25,0,0,-1.77079632679,0,1.7,0.55]) 
        else:
            q = np.array([0.25,0,0,-1.77079632679,0,1.7,1.0]) 

        self.arm.safe_move_to_position(q)
        
        for (name, pose) in self.detector.get_detections(): 
            if 'static' in name:
                self.staticBlocks[name]=self.alignZOrientation(self.cam2base(q,pose))
        
        return self.staticBlocks

    def search_dynamic(self):
        newDynamic = dict({})

        if (self.team=='red'):
            q = np.array([pi/2,pi/4,0,-pi/8,0,1.27,pi/4]) 
        else:
            q = np.array([-pi/2,pi/4,0,-pi/8,0,1.27,pi/4]) 
        
        self.arm.safe_move_to_position(q)
        
        for (name, pose) in self.detector.get_detections(): 
            if 'dynamic' in name:
                newDynamic[name]=(self.alignZOrientation(self.cam2base(q,pose)),time_in_seconds())
    
        self.dynamicBlocks = newDynamic
        return newDynamic

if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    
    arm = ArmController()
    search = Searcher(team,arm)
    # start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    # safe_move_to_position(start_position) # on your mark!

    # print("\n****************")
    # if team == 'blue':
    #     print("** BLUE TEAM  **")
    # else:
    #     print("**  RED TEAM  **")
    # print("****************")
    # input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    # print("Go!\n") # go!
    
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

    search.search_blocks_loop()
    # staticBlocks = search.search_static()
    # dynamicBlocks = search.search_dynamic()
    
