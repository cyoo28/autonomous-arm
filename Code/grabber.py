'''
from calculateIK6 import IK
from IK_velocity import IK_velocity
from calculateFK import FK

'''
from lib.calculateIK6 import IK
from lib.IK_velocity import IK_velocity
from lib.calculateFK import FK
from numpy import dot

import numpy as np
from core.utils import time_in_seconds
from numpy.linalg import norm
import traceback

def cos_sim(a, b):
  # https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
  return dot(a, b)/(norm(a)*norm(b))

class Grabber:
  def __init__(self, team, arm):
    self.arm = arm
    self.fk = FK()
    self.ik = IK()
    self.clearance_height = 0.6
    self.hover_height = 0.3
    self.min_height = 0.22
    self.step = 0.007
    self.grip_open_thresh = 0.038
    self.grip_block_thresh = 0.03
    self.grip_open_target = 0.079
    self.grip_close_force = 10
    self.joint_limits = None
    self.table_center = np.array([0, .99, 0]) * (1 if team == "red" else -1)
    self.table_speed = 0.01
    self.forecast_time = 10
    self.grab_headstart = 5
    self.radial_shift = 0.89
    self.team = team
    self.dynamic_hover_offset = .3

  def grab_static_block(self, tfm):
    try:
      print(f"received tfm = {tfm}")

      tfm = tfm.copy()
      tfm[2,3] = 0.225
      print("verifying gripper open")
      self.open_gripper()

      q = self.pick_hover_cfg(tfm)
      if (q is None):
        print(f"failed to find hover config for tfm = {tfm}")
        return False

      print("moving to hover position")
      self.arm.safe_move_to_position(q)

    
      q = self.vert_offset_cfg(max(self.min_height, tfm[2, 3]))
      if (q is None):
        print(f"failed to find grab config for tfm = {tfm}")
        return False
      print("moving down toward block")
      self.arm.safe_move_to_position(q)

      print("closing gripper")
      self.arm.exec_gripper_cmd(0, self.grip_close_force)

      success = self.is_block_grabbed()
      print(f"success? = {success}")
      return success
    except Exception as e:
      print(f"encountered unexpected error: {e}")
      traceback.print_exc()
      return False
    

  def open_gripper(self):
    gripper_state = self.arm.get_gripper_state()
    gpos = np.array(gripper_state["position"])
    if ((gpos < self.grip_open_thresh).any()):
      self.arm.exec_gripper_cmd(self.grip_open_target)

  def is_block_grabbed(self):
    gripper_state = self.arm.get_gripper_state()
    gpos = np.array(gripper_state["position"])
    print(f"gripper state = {gpos}")
    return gpos.sum() > self.grip_block_thresh

  def pick_hover_cfg(self, tfm):
    hover_tfm = tfm.copy()
    hover_tfm[2, 3] = self.hover_height
    hover_tfm[:, 1:3] *= -1
    z_rot = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])
    current_q = self.arm.get_positions(False)
    qs = []
    for i in range(4):
      target = {"R": hover_tfm[0:3, 0:3], "t": hover_tfm[0:3, 3]} # fixed
      ik_outs = self.ik.panda_ik(target)
      qs += [ik_outs[j] for j in range(ik_outs.shape[0])] # fixed
      hover_tfm[:3, :3] = z_rot @ hover_tfm[:3, :3]
    qs.sort(key = lambda elt: norm(current_q - elt))
    if (len(qs) == 0):
      return None
    return qs[0]

  def get_direction_score(self, q):
    _, T = self.fk.forward(q)
    pos = T[:3, 3]
    center_vect = self.table_center - pos
    center_vect /= norm(center_vect)
    x = T[:3, 0]
    return abs(cos_sim(center_vect, x))


  def pick_hover_cfg_dynamic(self, tfm):
    hover_tfm = tfm.copy()
    hover_tfm[2, 3] = self.hover_height
    hover_tfm[:, 1:3] *= -1
    z_rot = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])
    current_q = self.arm.get_positions(False)
    qs = []
    for i in range(4):
      target = {"R": hover_tfm[0:3, 0:3], "t": hover_tfm[0:3, 3]} # fixed
      ik_outs = self.ik.panda_ik(target)
      qs += [ik_outs[j] for j in range(ik_outs.shape[0])] # fixed
      hover_tfm[:3, :3] = z_rot @ hover_tfm[:3, :3]
    qs.sort(key = lambda elt: norm(current_q[:-1] - elt[:-1]))
    best_dist =  norm(current_q[:-1] - qs[0][:-1])
    qs = [q for q in qs if norm(current_q[:-1] - q[:-1]) < best_dist + 1e-4]
    qs.sort(key = lambda elt: self.get_direction_score(elt))
    if (len(qs) == 0):
      return None
    return qs[0]

  def get_closest_ik(self, T):
    curr_q = self.arm.get_positions(False)
    target = {"R": T[0:3, 0:3], "t": T[0:3, 3]}
    iks = self.ik.panda_ik(target)
    iks = [iks[j] for j in range(iks.shape[0])]
    iks.sort(key = lambda elt: norm(curr_q - elt))
    if (len(iks) == 0):
      return None
    else:
      return iks[0]

  def vert_offset_cfg(self, h):
    q = self.arm.get_positions(False)
    T = self.get_curr_tfm()
    T[2,3] = h
    return self.get_closest_ik(T)

  def get_curr_tfm(self):
    q = self.arm.get_positions(False)
    _, T = self.fk.forward(q)
    return T

  def predict_pos(self, tfm, t_measure, t_target):
    tfm = tfm.copy()
    tfm[:3, 3] -= self.table_center
    angle = (t_target - t_measure) * self.table_speed * 2 * np.pi
    R = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                  [np.sin(angle), np.cos(angle),  0, 0],
                  [0,             0,              1, 0],
                  [0,             0,              0, 1]])
    x = tfm[0,3]
    y = tfm[1,3]
    r = np.sqrt(x**2+y**2)
    print(f"radius={r}")
    new_tfm = R @ tfm
    #new_tfm[0,3] *= self.radial_shift
    #new_tfm[1,3] *= self.radial_shift
    x = new_tfm[0,3]
    y = new_tfm[1,3]
    r = np.sqrt(x**2+y**2)
    print(f"new radius={r}")
    new_tfm[:3, 3] += self.table_center
    return new_tfm

  def wait(self, timestamp):
    start_time = time_in_seconds()
    curr_time = time_in_seconds()
    while(curr_time < timestamp):
      curr_time = time_in_seconds()

  def grab_dynamic_block(self, tfm, t):
    try:
      tfm = tfm.copy()
      print(f"captured block pose: {tfm}")
      tfm[2,3] = 0.225



      print("verifying gripper open")
      self.open_gripper()

      tfm_pred = self.predict_pos(tfm, t, t + self.forecast_time)

      print(f"predicted pose: {tfm_pred}")

      
      q_hover = self.pick_hover_cfg_dynamic(tfm_pred)
      if (q_hover is None):
        print("failed to find hover cfg")
        return False
      print(f"hover cfg = {q_hover}")

      q_grab_test = self.vert_offset_cfg(max(self.min_height, tfm[2, 3]))
      if (q_grab_test is None):
        print(f"failed to find grab config for tfm = {tfm}")
        return False

      print("moving to hover pose")
      self.arm.safe_move_to_position(q_hover)

      print(f"waiting {t + self.forecast_time - self.grab_headstart - time_in_seconds()} seconds")
      self.wait(t + self.forecast_time - self.grab_headstart)

      q_grab = self.vert_offset_cfg(max(self.min_height, tfm[2, 3]))
      print("moving down toward block")
      self.arm.safe_move_to_position(q_grab)

      print("closing gripper")
      self.arm.exec_gripper_cmd(0, self.grip_close_force)

      success = self.is_block_grabbed()
      print(f"success? = {success}")
      return success
    except Exception as e:
      print(f"encountered unexpected error: {e}")
      traceback.print_exc()
      return False




