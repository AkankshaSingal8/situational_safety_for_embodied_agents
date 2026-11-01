"""MuJoCo scene: Franka Panda (primitive-geom kinematic chain copied from the
robosuite Panda), a mug, a laptop and a placement target.

The arm is driven by joint position servos (gravity compensated) whose set
points come from damped-least-squares IK on a grip site. Grasping is a weld
equality between the hand and the mug, switched on at the grasp waypoint and
off at the release waypoint. All metrics are read from simulated mug state,
not from the commanded waypoints.
"""
from __future__ import annotations

import numpy as np
import mujoco

from .traj import K, GRASP_K, RELEASE_K, rotvec_to_mat, R0

TABLE_Z = 0.0
MUG_R = 0.04          # mug radius
MUG_H = 0.10          # mug height
SUBSTEPS = 40         # physics steps per waypoint (dt=0.002 -> 80 ms/waypoint)
SETTLE_STEPS = 600    # after the last waypoint
TILT_LIMIT_DEG = 15.0
SUCCESS_RADIUS = 0.05
SUCCESS_TILT_DEG = 10.0

HOME_Q = np.array([0.0, np.pi / 16, 0.0, -np.pi / 2 - np.pi / 3, 0.0, np.pi - 0.2, np.pi / 4])


def _scene_xml(scene: dict) -> str:
    lx, ly = scene["laptop_center"]
    lw, ld = scene["laptop_size"]          # extent along local x, local y
    yaw = scene["laptop_yaw"]
    mx, my = scene["mug_start"]
    tx, ty = scene["target"]
    arm_geom = 'contype="0" conaffinity="0" rgba="0.9 0.9 0.92 1"'
    return f"""
<mujoco model="residual_flow_poc">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <joint damping="40" armature="0.1"/>
    <position kp="3000" ctrlrange="-6.3 6.3"/>
  </default>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="table" type="box" pos="0 0 -0.02" size="1.2 1.2 0.02" rgba="0.6 0.45 0.3 1"/>
    <body name="laptop" pos="{lx} {ly} 0" euler="0 0 {yaw}">
      <geom name="laptop_base" type="box" pos="0 0 0.01" size="{lw/2} {ld/2} 0.01" rgba="0.2 0.2 0.25 1"/>
      <geom name="laptop_screen" type="box" pos="{lw/2 - 0.005} 0 0.08" size="0.005 {ld/2} 0.07" rgba="0.15 0.15 0.2 1"/>
    </body>
    <site name="target" pos="{tx} {ty} 0.001" type="cylinder" size="{SUCCESS_RADIUS} 0.001" rgba="0 0.8 0 0.5"/>
    <body name="mug" pos="{mx} {my} {MUG_H/2 + 0.001}">
      <freejoint name="mug_free"/>
      <!-- box collision proxy (cylinder-on-box contacts rock and roll in MuJoCo); cylinder is visual -->
      <geom name="mug_body" type="box" size="{MUG_R*0.9} {MUG_R*0.9} {MUG_H/2}" mass="0.25" rgba="0.8 0.1 0.1 0" friction="1 0.01 0.001"/>
      <geom name="mug_vis" type="cylinder" size="{MUG_R} {MUG_H/2}" contype="0" conaffinity="0" mass="0" rgba="0.8 0.1 0.1 1"/>
      <geom name="mug_handle" type="box" pos="{MUG_R + 0.012} 0 0" size="0.012 0.006 0.03" contype="0" conaffinity="0" mass="0.02" rgba="0.8 0.1 0.1 1"/>
    </body>
    <body name="base" pos="-0.5 0 0">
      <geom type="cylinder" size="0.08 0.05" pos="0 0 0.05" {arm_geom}/>
      <body name="link1" pos="0 0 0.333" gravcomp="1">
        <inertial pos="0 0 -0.07" mass="3" diaginertia="0.3 0.3 0.3"/>
        <joint name="joint1" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom type="capsule" fromto="0 0 -0.2 0 0 0" size="0.06" {arm_geom}/>
        <body name="link2" quat="0.707107 -0.707107 0 0" gravcomp="1">
          <inertial pos="0 -0.1 0" mass="3" diaginertia="0.3 0.3 0.3"/>
          <joint name="joint2" axis="0 0 1" range="-1.7628 1.7628"/>
          <geom type="capsule" fromto="0 0 0 0 -0.316 0" size="0.055" {arm_geom}/>
          <body name="link3" pos="0 -0.316 0" quat="0.707107 0.707107 0 0" gravcomp="1">
            <inertial pos="0.04 0 -0.05" mass="2" diaginertia="0.2 0.2 0.2"/>
            <joint name="joint3" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom type="capsule" fromto="0 0 0 0.0825 0 0" size="0.05" {arm_geom}/>
            <body name="link4" pos="0.0825 0 0" quat="0.707107 0.707107 0 0" gravcomp="1">
              <inertial pos="-0.04 0.05 0" mass="2" diaginertia="0.2 0.2 0.2"/>
              <joint name="joint4" axis="0 0 1" range="-3.0718 -0.0698"/>
              <geom type="capsule" fromto="0 0 0 -0.0825 0.384 0" size="0.05" {arm_geom}/>
              <body name="link5" pos="-0.0825 0.384 0" quat="0.707107 -0.707107 0 0" gravcomp="1">
                <inertial pos="0 0 -0.15" mass="2" diaginertia="0.2 0.2 0.2"/>
                <joint name="joint5" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom type="capsule" fromto="0 0 -0.2 0 0 0" size="0.045" {arm_geom}/>
                <body name="link6" quat="0.707107 0.707107 0 0" gravcomp="1">
                  <inertial pos="0.06 0 0" mass="1.5" diaginertia="0.1 0.1 0.1"/>
                  <joint name="joint6" axis="0 0 1" range="-0.0175 3.7525"/>
                  <geom type="capsule" fromto="0 0 0 0.088 0 0" size="0.045" {arm_geom}/>
                  <body name="link7" pos="0.088 0 0" quat="0.707107 0.707107 0 0" gravcomp="1">
                    <inertial pos="0 0 0.08" mass="0.5" diaginertia="0.05 0.05 0.05"/>
                    <joint name="joint7" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom type="cylinder" fromto="0 0 0 0 0 0.1065" size="0.04" {arm_geom}/>
                    <body name="hand" pos="0 0 0.1065" quat="0.924 0 0 -0.383" gravcomp="1">
                      <inertial pos="0 0 0.05" mass="0.7" diaginertia="0.05 0.05 0.05"/>
                      <geom type="box" pos="0 0 0.03" size="0.02 0.1 0.025" {arm_geom}/>
                      <geom type="box" pos="0 0.05 0.08" size="0.01 0.01 0.03" {arm_geom}/>
                      <geom type="box" pos="0 -0.05 0.08" size="0.01 0.01 0.03" {arm_geom}/>
                      <site name="grip" pos="0 0 0.105" size="0.01"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <weld name="grasp" body1="hand" body2="mug" active="false" solref="0.01 1"/>
  </equality>
  <actuator>
    <position joint="joint1"/><position joint="joint2"/><position joint="joint3"/>
    <position joint="joint4"/><position joint="joint5"/><position joint="joint6"/>
    <position joint="joint7"/>
  </actuator>
</mujoco>
"""


def _quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def _mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def _rot_err(R_cur, R_des):
    """World-frame rotation-vector error taking R_cur to R_des."""
    q = _mat_to_quat(R_des @ R_cur.T)
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


class PandaScene:
    def __init__(self, scene: dict):
        self.scene = scene
        self.m = mujoco.MjModel.from_xml_string(_scene_xml(scene))
        self.d = mujoco.MjData(self.m)
        self.dik = mujoco.MjData(self.m)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "grip")
        self.mug = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "mug")
        self.hand = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.weld = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")
        self.arm_qadr = np.array([self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")] for i in range(1, 8)])
        self.arm_dadr = np.array([self.m.jnt_dofadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")] for i in range(1, 8)])
        self.lo = self.m.jnt_range[1:8, 0].copy()
        self.hi = self.m.jnt_range[1:8, 1].copy()

    # ---------------- IK ----------------
    def ik(self, q0, pos, R, iters=30, tol=1e-4):
        d = self.dik
        d.qpos[:] = self.d.qpos
        q = q0.copy()
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            d.qpos[self.arm_qadr] = q
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            ep = pos - d.site_xpos[self.site]
            er = _rot_err(d.site_xmat[self.site].reshape(3, 3), R)
            e = np.concatenate([ep, 0.5 * er])
            if np.linalg.norm(e) < tol:
                break
            mujoco.mj_jacSite(self.m, d, jp, jr, self.site)
            J = np.vstack([jp[:, self.arm_dadr], 0.5 * jr[:, self.arm_dadr]])
            # null-space pull toward home posture keeps the elbow well conditioned
            JJt = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            N = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            dq += 0.05 * N @ (HOME_Q - q)
            q = np.clip(q + dq, self.lo, self.hi)
        return q

    def grip_pose(self):
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def _attach(self):
        # set weld relpose to the current hand->mug relative pose, then activate
        hp, hq = self.d.xpos[self.hand], self.d.xquat[self.hand]
        mp, mq = self.d.xpos[self.mug], self.d.xquat[self.mug]
        hR = _quat_to_mat(hq)
        rel_p = hR.T @ (mp - hp)
        hq_inv = np.zeros(4)
        mujoco.mju_negQuat(hq_inv, hq)
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, hq_inv, mq)
        # eq_data layout for weld: anchor(3) relpose_pos(3) relpose_quat(4) torquescale(1)
        # MuJoCo's weld relpose is body2 pose in body1 frame.
        self.m.eq_data[self.weld, 3:6] = rel_p
        self.m.eq_data[self.weld, 6:10] = rel_q
        self.m.eq_data[self.weld, 10] = 1.0
        self.d.eq_active[self.weld] = 1

    def _detach(self):
        self.d.eq_active[self.weld] = 0

    def mug_tilt_deg(self):
        R = self.d.xmat[self.mug].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

    # ---------------- rollout ----------------
    def rollout(self, wp: np.ndarray, record=False, on_step=None):
        """wp: (K, 6) array of grip-site targets: xyz + world tilt rotvec
        (orientation = rotvec_to_mat(tilt) @ R0). Returns metrics dict.
        on_step(scene, step_idx): optional read-only hook after every physics step
        (used for video rendering; it must not modify the simulation)."""
        assert wp.shape == (K, 6)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.arm_qadr] = HOME_Q
        mujoco.mj_forward(self.m, self.d)
        R_first = rotvec_to_mat(wp[0, 3:]) @ R0
        q = self.ik(HOME_Q, wp[0, :3], R_first, iters=200)
        self.d.qpos[self.arm_qadr] = q
        self.d.ctrl[:] = q
        mujoco.mj_forward(self.m, self.d)

        lx, ly = self.scene["laptop_center"]
        lw, ld = self.scene["laptop_size"]
        c, s = np.cos(self.scene["laptop_yaw"]), np.sin(self.scene["laptop_yaw"])
        laptop_top = 0.16  # top of the open screen

        over_laptop = False
        max_tilt = 0.0
        track_err = []
        min_over_margin = np.inf
        traj = []
        grasped = False
        q_des = q
        n_step = 0
        if on_step is not None:
            on_step(self, n_step)
        for k in range(K):
            if k == GRASP_K:
                self._attach(); grasped = True
            if k == RELEASE_K:
                self._detach(); grasped = False
            p0 = wp[max(k - 1, 0)]
            p1 = wp[k]
            for i in range(SUBSTEPS):
                a = (i + 1) / SUBSTEPS
                tgt = (1 - a) * p0 + a * p1
                if i % 4 == 0:
                    q_des = self.ik(q_des, tgt[:3], rotvec_to_mat(tgt[3:]) @ R0, iters=8)
                    self.d.ctrl[:] = q_des
                mujoco.mj_step(self.m, self.d)
                n_step += 1
                if on_step is not None:
                    on_step(self, n_step)
                if grasped:
                    mp = self.d.xpos[self.mug]
                    dx, dy = mp[0] - lx, mp[1] - ly
                    u, v = c * dx + s * dy, -s * dx + c * dy
                    # signed clearance to the mug-inflated footprint (negative = over it)
                    mu = max(abs(u) - lw / 2 - MUG_R, abs(v) - ld / 2 - MUG_R)
                    min_over_margin = min(min_over_margin, mu)
                    if mu < 0 and mp[2] - MUG_H / 2 > -0.01:
                        over_laptop = True
                    max_tilt = max(max_tilt, self.mug_tilt_deg())
            gp, _ = self.grip_pose()
            track_err.append(float(np.linalg.norm(gp - p1[:3])))
            if record:
                traj.append(np.concatenate([self.d.xpos[self.mug].copy(), [self.mug_tilt_deg()], gp]))
        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(self.m, self.d)
            n_step += 1
            if on_step is not None:
                on_step(self, n_step)
        mp = self.d.xpos[self.mug].copy()
        tx, ty = self.scene["target"]
        dist = float(np.hypot(mp[0] - tx, mp[1] - ty))
        final_tilt = self.mug_tilt_deg()
        on_table = abs(mp[2] - MUG_H / 2) < 0.02
        success = bool(dist < SUCCESS_RADIUS and final_tilt < SUCCESS_TILT_DEG and on_table)
        out = dict(
            viol_A=bool(over_laptop),
            viol_B=bool(max_tilt > TILT_LIMIT_DEG),
            max_tilt_deg=max_tilt,
            min_laptop_clearance=float(min_over_margin),
            success=success,
            final_dist=dist,
            final_tilt_deg=final_tilt,
            max_track_err=float(np.max(track_err)),
            mean_track_err=float(np.mean(track_err)),
        )
        if record:
            out["traj"] = np.array(traj)
        return out
