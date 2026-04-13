import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from types import NoneType
import mujoco
from mujoco.renderer import Renderer
import numpy as np
import open3d as o3d
import ompl.base as ob
import ompl.geometric as og
from scipy.spatial.transform import Rotation as R

import pyoctomap
from scipy import ndimage
import heapq
import math
import sys
import os
from typing import List, Tuple, Dict, Optional
from core.manipulator.kinematics import MorphIManipulator

from zmq import CURVE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from numba import njit
import time
from core.nav.planner.trajectory_opt import *
from python_motion_planning.common import *
from python_motion_planning.path_planner import *
from python_motion_planning.controller import *
from python_motion_planning.curve_generator import Bezier, curve
import numpy as np
import ompl.base as ob
import ompl.geometric as og

def transform_arm_points_to_world(
    base_pose: Tuple[float, float, float, float],
    arm_offset: np.ndarray,
    points_local: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Transform arm-local points to world frame.
    
    Args:
        base_pose: (x_base, y_base, z_base, yaw_base) in world frame
        arm_offset: (dx, dy, dz) from robot center to arm base (in robot local frame)
        points_local: dict with 'p_inner', 'wrist_base', 'ee' in arm-local frame
    
    Returns:
        Same dict, but in world coordinates.
    """
    x_b, y_b, z_b, yaw_b = base_pose
    c, s = np.cos(yaw_b), np.sin(yaw_b)
    Rz_2d = np.array([[c, -s],
                      [s,  c]])

    # Arm base position in world (horizontal only; z added separately)
    arm_base_xy_world = Rz_2d @ arm_offset[:2]
    arm_base_z_world = z_b + arm_offset[2]  # arm_offset[2] is height above base

    world_points = {}
    for key, pt in points_local.items():
        # Rotate point's XY by base yaw
        pt_xy_world = Rz_2d @ pt[:2]
        # Total world position = base + arm_base_offset + rotated point
        world_x = x_b + arm_base_xy_world[0] + pt_xy_world[0]
        world_y = y_b + arm_base_xy_world[1] + pt_xy_world[1]
        world_z = arm_base_z_world + pt[2]  # pt[2] is relative to arm base
        world_points[key] = np.array([world_x, world_y, world_z])

    return world_points

if __name__ == "__main__":
    path = "/home/ziczac/dev/fiverr/071125_arm_obotx_on_mobile_base/mobile-manipulator/src/env/market_world_plain.xml"
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    height = 480
    width = 640
    camera_id = model.cam("pov_1").id
    renderer = Renderer(model, height=height, width=width)

    fov = model.cam_fovy[camera_id]
    theta = np.deg2rad(fov)
    fx = width / 2 / np.tan(theta / 2)
    fy = height / 2 / np.tan(theta / 2)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    cam_pos = data.cam_xpos[camera_id]
    cam_rot = data.cam_xmat[camera_id].reshape(3, 3)
    
    print("CAM POS : ", cam_pos)
    extr = np.eye(4)
    extr[:3, :3] = cam_rot.T
    extr[:3, 3] = cam_rot.T @ cam_pos

    def render_rgbd(renderer, camera_id):
        renderer.update_scene(data, camera=camera_id)
        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()
        rgb = renderer.render()
        return rgb, depth

    def rgbd_to_pointcloud(rgb, depth, intr, extr, depth_trunc=40.0):
        h, w = depth.shape
        cc, rr = np.meshgrid(np.arange(w), np.arange(h))
        valid = (depth > 0) & (depth < depth_trunc)
        z = np.where(valid, depth, np.nan)
        x = (cc - intr[0, 2]) * z / intr[0, 0]
        y = (rr - intr[1, 2]) * z / intr[1, 1]
        xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        color = rgb.reshape(-1, 3) / 255.0
        mask = ~np.isnan(xyz[:, 2])
        xyz = xyz[mask]
        color = color[mask]
        xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
        xyz_world = (extr @ xyz_h.T).T[:, :3]
        return xyz_world, color

    rgb, depth = render_rgbd(renderer, camera_id)
    points, colors = rgbd_to_pointcloud(rgb, depth, intr, extr)

    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(points)

    plane_model, _ = pcd_temp.segment_plane(distance_threshold=0.001, ransac_n=3, num_iterations=1000)
    normal = np.array(plane_model[:3])
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal

    target = np.array([0, 0, 1])
    if np.allclose(normal, target):
        R_align = np.eye(3)
    elif np.allclose(normal, -target):
        R_align = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        v = np.cross(normal, target)
        s = np.linalg.norm(v)
        c = np.dot(normal, target)
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    points_rotated = points @ R_align.T
    points_aligned = points_rotated - points_rotated.min(axis=0)

    R_flip_z = np.array([[1,  0,  0],
                         [0, -1,  0],
                         [0,  0, -1]], dtype=np.float64)
    points_final = points_aligned @ R_flip_z.T
    points_final -= points_final.min(axis=0)
    max_z = points_final[:, 2].max()
    print("Max Z:", max_z)
    print("Shape :", points_final.shape)
    
    bounds_xy = [[np.min(points[:, 0]), np.max(points[:, 0])], [np.min(points[:, 1]), np.max(points[:,1])]]     
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])                      
    bounds = [*bounds_xy, [z_min, z_max]]      

    Lx, Ly, Lz = 0.35, 0.35, 0.14  # full size: 0.6 x 0.5 x 0.3 m
    base_z = 0.142 + 0.14
    resolution = 0.05

    mobile2basearm = {
        "left":     np.array([0.15 + 0.01,   +(0.15 - 0.00465966), 0.127]),
        "right":    np.array([0.15 + 0.01,   -(0.15 - 0.00465966), 0.127])
    }
    # 0.4 0.02 0.04
    basearm2verticalarm = {
        "left_1":       np.array([-0.05,   +0.04,    0]),
        "left_2":       np.array([+0.05,   -0.04,    0]),
        "right_1":      np.array([-0.05,   +0.04,    0]),
        "right_2":      np.array([+0.05,   -0.04,    0]),
    }
    
    vertical_box_size = np.array([0.04, 0.02, 0.715])
    
    start_build_map = time.time()
    map_ = Grid(bounds=bounds, resolution=resolution)

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    z_min, z_max = bounds[2]

    for point in points:
        x, y, z = point
        if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
            continue
        ix = int((x - x_min) / resolution)
        iy = int((y - y_min) / resolution)
        iz = int((z - z_min) / resolution)
        if 0 <= ix < map_.shape[0] and 0 <= iy < map_.shape[1] and 0 <= iz < map_.shape[2]:
            map_.type_map[ix, iy, iz] = (TYPES.OBSTACLE)

    map_.inflate_obstacles(radius=0.1)
    print("Time to generate map : ", time.time()-start_build_map, "s")
    
    start = (3.0, 5.0, base_z)
    goal = (5.0, 2.5, base_z)
    
    start_idx = map_.world_to_map(start) 
    goal_idx = map_.world_to_map(goal)
    
    print(start_idx)
    print(goal_idx) 

    LOCAL_CORNERS = np.array([
        [ dx,  dy,  dz] for dx in (-Lx, Lx)
                        for dy in (-Ly, Ly)
                        for dz in (-Lz, Lz)
    ])  # shape (8, 3)

    def get_box_edges():
        edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(LOCAL_CORNERS[i] != LOCAL_CORNERS[j]) == 1:
                    edges.append((i, j))
        return edges

    BOX_EDGES = get_box_edges()  # 12 edges

    map_.type_map[start_idx] = TYPES.START
    map_.type_map[goal_idx] = TYPES.GOAL
    
    # === ARM INSTANCES ===
    arm_left = MorphIManipulator("left")
    arm_right = MorphIManipulator("right")

    # === STATE SPACE: 13D ===
    space = ob.RealVectorStateSpace(13)

    bounds_ompl = ob.RealVectorBounds(13)
    # Base
    bounds_ompl.setLow(0, map_.bounds[0,0])
    bounds_ompl.setHigh(0, map_.bounds[0,1])
    bounds_ompl.setLow(1, map_.bounds[1,0])
    bounds_ompl.setHigh(1, map_.bounds[1,1])
    bounds_ompl.setLow(2, -np.pi)
    bounds_ompl.setHigh(2, np.pi)

    # Left arm
    bounds_ompl.setLow(3, arm_left.bounds_h[0])
    bounds_ompl.setHigh(3, arm_left.bounds_h[1])
    bounds_ompl.setLow(4, arm_left.bounds_h[0])
    bounds_ompl.setHigh(4, arm_left.bounds_h[1])
    bounds_ompl.setLow(5, arm_left.bounds_a[0])
    bounds_ompl.setHigh(5, arm_left.bounds_a[1])
    bounds_ompl.setLow(6, arm_left.bounds_theta[0])
    bounds_ompl.setHigh(6, arm_left.bounds_theta[1])
    bounds_ompl.setLow(7, arm_left.bounds_phi[0])
    bounds_ompl.setHigh(7, arm_left.bounds_phi[1])

    # Right arm
    bounds_ompl.setLow(8, arm_right.bounds_h[0])
    bounds_ompl.setHigh(8, arm_right.bounds_h[1])
    bounds_ompl.setLow(9, arm_right.bounds_h[0])
    bounds_ompl.setHigh(9, arm_right.bounds_h[1])
    bounds_ompl.setLow(10, arm_right.bounds_a[0])
    bounds_ompl.setHigh(10, arm_right.bounds_a[1])
    bounds_ompl.setLow(11, arm_right.bounds_theta[0])
    bounds_ompl.setHigh(11, arm_right.bounds_theta[1])
    bounds_ompl.setLow(12, arm_right.bounds_phi[0])
    bounds_ompl.setHigh(12, arm_right.bounds_phi[1])

    space.setBounds(bounds_ompl)
    si = ob.SpaceInformation(space)

    # === VALIDITY CHECKER ===
    def is_state_valid_full(state):
        x = state[0]
        y = state[1]
        yaw = state[2]
        q_left = np.array([state[i] for i in range(3, 8)])
        q_right = np.array([state[i] for i in range(8, 13)])

        # Base bounds
        if not (map_.bounds[0,0] <= x <= map_.bounds[0,1] and
                map_.bounds[1,0] <= y <= map_.bounds[1,1]):
            return False

        base_pose = (x, y, base_z, yaw)
        try:
            edges = get_all_robot_edges(
                waypoints_yaw=[base_pose],
                LOCAL_CORNERS=LOCAL_CORNERS,
                BOX_EDGES=BOX_EDGES,
                mobile2basearm=mobile2basearm,
                basearm2verticalarm=basearm2verticalarm,
                vertical_box_size=vertical_box_size,
                arm_left=arm_left,
                arm_right=arm_right,
                q_left=q_left,
                q_right=q_right,
                sampling=1,
                map_frame=False,
                grid_map=map_,
                diamond_radii=(0.05, 0.08, 0.05),
                edge_diagonal=False  # faster collision check
            )
        except Exception:
            return False

        for p1, p2 in edges:
            try:
                m1 = tuple(int(round(v)) for v in map_.world_to_map(p1))
                m2 = tuple(int(round(v)) for v in map_.world_to_map(p2))
            except Exception:
                return False

            if not (0 <= m1[0] < map_.shape[0] and 0 <= m1[1] < map_.shape[1] and 0 <= m1[2] < map_.shape[2]):
                return False
            if not (0 <= m2[0] < map_.shape[0] and 0 <= m2[1] < map_.shape[1] and 0 <= m2[2] < map_.shape[2]):
                return False

            if map_.in_collision(m1, m2):
                return False

        return True

    si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid_full))
    si.setup()

    # === START & GOAL ===
    start_base = (3.0, 5.0, 0.0)
    goal_base = (5.0, 2.5, 1.54)

    q_left_start = np.array([0.5, 0.58, 0.3, 0.0, 0.0])
    q_right_start = np.array([0.5, 0.58, 0.3, 0.0, 0.0])

    q_left_goal = np.array([0.5, 0.58, 0, 0.0, 0.0])
    q_right_goal = np.array([0.5, 0.58, 0, 0.0, 0.0])

    start_state = list(start_base) + list(q_left_start) + list(q_right_start)
    goal_state = list(goal_base) + list(q_left_goal) + list(q_right_goal)

    start = ob.State(space)
    goal = ob.State(space)
    for i in range(13):
        start[i] = start_state[i]
        goal[i] = goal_state[i]

    # === PLANNER ===
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)
    opt = ob.PathLengthOptimizationObjective(si)
    pdef.setOptimizationObjective(opt)

    planner = og.RRTConnect(si)
    planner.setIntermediateStates(True)
    planner.setProblemDefinition(pdef)
    planner.setRange(resolution)
    planner.setup()

    # === SOLVE ===
    print("🔍 Solving in 13D whole-body space...")
    solved = planner.solve(30.0)  # 30 seconds max

    if solved and pdef.hasExactSolution():
        path = pdef.getSolutionPath()
        print(f"✅ Found path with {path.getStateCount()} states.")
        
        # Lists to store trajectories
        base_solver = []      # each: (x, y, yaw)
        q_left_solver = []    # each: [h1, h2, a1, theta, phi]
        q_right_solver = []   # each: [h1, h2, a1, theta, phi]

        for state in path.getStates():
            s = [state[i] for i in range(13)]
            
            # Base: x, y, yaw
            base_solver.append((s[0], s[1], s[2]))
            
            # Left arm: 5 DOF
            q_left_solver.append(np.array(s[3:8]))
            
            # Right arm: 5 DOF
            q_right_solver.append(np.array(s[8:13]))

        # Optional: print first and last
        print("Start configuration:")
        print(f"  Base: {base_solver[0]}")
        print(f"  Left arm: {q_left_solver[0]}")
        print(f"  Right arm: {q_right_solver[0]}")

        print("Goal configuration:")
        print(f"  Base: {base_solver[-1]}")
        print(f"  Left arm: {q_left_solver[-1]}")
        print(f"  Right arm: {q_right_solver[-1]}")
        
        full_path = []
        for state in path.getStates():
            s = [state[i] for i in range(13)]
            full_path.append(s)
            
        waypoints_yaw = [(s[0], s[1], base_z, s[2]) for s in full_path]
        waypoints = [(s[0], s[1], base_z) for s in full_path]
        
        for i, wp in enumerate(waypoints_yaw):
            print(f"  [{i}] → x={wp[0]:.2f}, y={wp[1]:.2f}, z={wp[2]:.2f}, yaw={wp[3]:.2f}")
    else:
        print("❌ No valid path found.")
        full_path = None
        waypoints_yaw = None
        
    # waypoints = map_.path_world_to_map(waypoints)
    # waypoints_yaw = [(state[0], state[1], state[2], state_yaw[3]) 
    #                     for state, state_yaw in zip(waypoints, waypoints_yaw)]  # (x, y, z, yaw)
    
  
    # Visualize
    vis = Visualizer3D()
    alpha_3d={TYPES.FREE: 0.0, 
              TYPES.OBSTACLE: 0.1, 
              TYPES.START: 0.5, 
              TYPES.GOAL: 0.5, 
              TYPES.INFLATION: 0.1, 
              TYPES.EXPAND: 0.01, 
              TYPES.CUSTOM: 0.1}
    vis.plot_grid_map(map_, alpha_3d=alpha_3d)
    # vis.plot_path(waypoints, color="#4bc911")
    # vis.plot_orientation(waypoints_yaw, color="#dae41e")
    # vis.plot_robot_body(
    #     poses=waypoints_yaw,
    #     local_corners=LOCAL_CORNERS,
    #     color="#679ceb",
    #     sampling=50,
    # )
    
    # # Always include pose 0, then sample every N
    # sampling_arm = 50  # adjust as needed
    # indices = [0]
    # if sampling_arm > 1:
    #     indices += list(range(sampling_arm, len(waypoints_yaw), sampling_arm))
    # else:
    #     indices = list(range(len(waypoints_yaw)))
    
    def create_box_from_segment(start, end, radius, color, opacity, plotter, name):
        vec = end - start
        length = np.linalg.norm(vec)
        if length < 1e-6:
            return
        direction = vec / length
        center = (start + end) / 2.0
        create_oriented_box(
            center=center,
            direction=direction,
            length=length,
            radius_x=radius,
            radius_y=0.04,
            color=color,
            opacity=opacity,
            plotter=plotter,
            name=name
        )
        
    def get_oriented_box_edges(start, end, radius_x, radius_y=None):
        if radius_y is None:
            radius_y = radius_x

        vec = end - start
        length = np.linalg.norm(vec)
        if length < 1e-12:
            # Degenerate case: return empty or point
            return []

        direction = vec / length
        center = (start + end) / 2.0

        dir_norm = direction
        if abs(dir_norm[2]) < 0.99:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([1.0, 0.0, 0.0])

        local_x = np.cross(up, dir_norm)
        local_x_norm = np.linalg.norm(local_x)
        if local_x_norm < 1e-12:
            local_x = np.array([1.0, 0.0, 0.0])
        else:
            local_x /= local_x_norm

        local_y = np.cross(dir_norm, local_x)
        local_y /= np.linalg.norm(local_y) + 1e-12

        R = np.column_stack((local_x, local_y, dir_norm))

        # Local corners of box (centered at origin, axis-aligned in local frame)
        dx, dy, dz = radius_x, radius_y, length / 2.0
        local_corners = np.array([
            [-dx, -dy, -dz],
            [+dx, -dy, -dz],
            [+dx, +dy, -dz],
            [-dx, +dy, -dz],
            [-dx, -dy, +dz],
            [+dx, -dy, +dz],
            [+dx, +dy, +dz],
            [-dx, +dy, +dz]
        ])

        # Transform to world
        world_corners = (R @ local_corners.T).T + center

        # Define edges (same as your BOX_EDGES)
        edges_indices = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
        ]

        edges = [(world_corners[i], world_corners[j]) for i, j in edges_indices]
        return edges

    # for idx in indices:
    #     base_pose_map = waypoints_yaw[idx]
    #     base_pose = (
    #         *vis.grid_map.map_to_world(base_pose_map[:3]),
    #         base_pose_map[3]
    #     )
    #     # Left arm
    #     pts_l = arm_left.fk_all_points(q_left_solver[idx])
    #     pts_l_w = transform_arm_points_to_world(base_pose, mobile2basearm["left"], pts_l)
    #     create_box_from_segment(pts_l_w["p_inner"], pts_l_w["wrist_base"], 0.02, "#1f77b4", 0.5, vis.pv_plotter, f"lcol_{idx}")
    #     # create_box_from_segment(pts_l_w["wrist_base"], pts_l_w["ee"], 0.015, "#aec7e8", 0.7, vis.pv_plotter, f"lwrist_{idx}")
    #     create_3radius_diamond(
    #         start=pts_l_w["wrist_base"],
    #         end=pts_l_w["ee"],
    #         r0=0.05,    # near wrist base
    #         r1=0.08,   # narrowest at center
    #         r2=0.05,   # near end-effector (slightly wider)
    #         plotter=vis.pv_plotter,
    #         color="#1f77b4",
    #         opacity=0.8,
    #         name=f"lwrist_diamond_{idx}"
    #     )
        
    #     # Right arm
    #     pts_r = arm_right.fk_all_points(q_right_solver[idx])
    #     pts_r_w = transform_arm_points_to_world(base_pose, mobile2basearm["right"], pts_r)
    #     create_box_from_segment(pts_r_w["p_inner"], pts_r_w["wrist_base"], 0.02, "#ff7f0e", 0.5, vis.pv_plotter, f"rcol_{idx}")
    #     # create_box_from_segment(pts_r_w["wrist_base"], pts_r_w["ee"], 0.015, "#ffbb78", 0.7, vis.pv_plotter, f"rwrist_{idx}")
    #     create_3radius_diamond(
    #         start=pts_r_w["wrist_base"],
    #         end=pts_r_w["ee"],
    #         r0=0.05,    # near wrist base
    #         r1=0.08,   # narrowest at center
    #         r2=0.05,   # near end-effector (slightly wider)
    #         plotter=vis.pv_plotter,
    #         color="#1f77b4",
    #         opacity=0.8,
    #         name=f"rwrist_diamond_{idx}"
    #     )
        
    # vis.plot_dual_arm_robot(
    #     poses=waypoints_yaw,
    #     mobile2basearm=mobile2basearm,
    #     basearm2verticalarm=basearm2verticalarm,
    #     vertical_box_size=vertical_box_size,
    #     color_left="#13ae00",
    #     color_right="#d62728",
    #     q_left=q_left_solver,
    #     q_right=q_right_solver,
    #     opacity=0.6,
    #     sampling=50,
    #     map_frame=True,
    # )
    
    # edges = get_all_robot_edges(
    #     waypoints_yaw=waypoints_yaw,
    #     LOCAL_CORNERS=LOCAL_CORNERS,
    #     BOX_EDGES=BOX_EDGES,
    #     mobile2basearm=mobile2basearm,
    #     basearm2verticalarm=basearm2verticalarm,
    #     vertical_box_size=vertical_box_size,
    #     arm_left=arm_left,
    #     arm_right=arm_right,
    #     q_left=q_left_solver,
    #     q_right=q_left_solver,
    #     sampling=20,
    #     map_frame=True,
    #     grid_map=map_,
    #     diamond_radii=(0.05, 0.08, 0.05)
    # )
    
    # plot_edges_from_list(vis.pv_plotter, edges, color="#000000", line_width=2.0)

    # print(f"Total edges: {len(edges)}")
    # for i, (p1, p2) in enumerate(edges[:5]):
    #     print(f"Edge {i}: {p1} → {p2}")
    
    vis.show()
    vis.close()
    
