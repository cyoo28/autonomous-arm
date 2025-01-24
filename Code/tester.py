import sys
import rospy
import numpy as np
from math import pi
from lib.calculateIK6 import IK

from core.interfaces import ArmController

rospy.init_node('demo')

arm = ArmController()
ik = IK()
T = np.array([[1, 0, 0, 0],
                [0, -1, 0, -0.69155],
                [0, 0, -1, 0.22],
                [0, 0, 0, 0]])
target = {"R": T[0:3, 0:3], "t": T[0:3, 3]}
ik_outs = ik.panda_ik(target)
q = ik_outs[0]
arm.safe_move_to_position(q)
arm.close_gripper()

