import rclpy
import os
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Bool, String, Float32
from scipy.ndimage import distance_transform_edt
import numpy as np
import sys
import json

import heapq
from tinynav.core.math_utils import matrix_to_quat, msg2np, np2msg, estimate_pose, np2tf, rerank_by_pnp_inliers
from sensor_msgs.msg import Image, CameraInfo
from message_filters import TimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import cv2
from codetiming import Timer
import argparse

from tinynav.core.models_trt import LightGlueTRT, Dinov2TRT, SuperPointTRT
import logging
import asyncio
from tf2_ros import TransformBroadcaster
from tinynav.core.build_map_node import TinyNavDB
from tinynav.core.build_map_node import find_loop, solve_pose_graph
import einops
from tinynav.core.build_map_node import OdomPoseRecorder
logger = logging.getLogger(__name__)



def draw_image_match_origin(prev_image: np.ndarray, curr_image: np.ndarray, prev_keypoints: np.ndarray, curr_keypoints: np.ndarray, matches: np.ndarray):
    cv_matches = [cv2.DMatch(_queryIdx=matches[index, 0].item(), _trainIdx=matches[index, 1].item(), _imgIdx=0, _distance=0) for index in range(matches.shape[0])]
    # convert kpts_prev and kpts_curr to cv2.KeyPoint
    cv_kpts_prev = [cv2.KeyPoint(x=prev_keypoints[index, 0].item(), y=prev_keypoints[index, 1].item(), size=20) for index in range(prev_keypoints.shape[0])]
    cv_kpts_curr = [cv2.KeyPoint(x=curr_keypoints[index, 0].item(), y=curr_keypoints[index, 1].item(), size=20) for index in range(curr_keypoints.shape[0])]
    output_image = cv2.drawMatches(prev_image, cv_kpts_prev, curr_image, cv_kpts_curr, cv_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return output_image

def depth_to_cloud(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert depth image to point cloud.
    :param depth: (H, W) depth image.
    :param K: (3, 3) camera intrinsic matrix.
    :return: (N, 3) point cloud in camera coordinates.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.flatten()

    x = (u.flatten() - K[0, 2]) * z / K[0, 0]
    y = (v.flatten() - K[1, 2]) * z / K[1, 1]

    points_3d = np.vstack((x, y, z)).T
    return points_3d[~np.isnan(points_3d).any(axis=1)]

def transform_point_cloud(point_cloud: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform a point cloud with a transformation matrix.
    :param point_cloud: (N, 3) numpy array of points in the point cloud.
    :param T: (4, 4) transformation matrix.
    :return: (N, 3) transformed point cloud.
    """
    assert point_cloud.shape[1] == 3, "Point cloud must be of shape (N, 3)"
    assert T.shape == (4, 4), "Transformation matrix must be of shape (4, 4)"

    # Convert to homogeneous coordinates
    ones = np.ones((point_cloud.shape[0], 1))
    homogeneous_points = np.hstack((point_cloud, ones))
    # Apply transformation
    transformed_points = homogeneous_points @ T.T
    return transformed_points[:, :3]

def heuristic(start, goal, resolution):
    vec_start = np.array(start)
    vec_goal = np.array(goal)
    return np.linalg.norm((vec_start - vec_goal) * resolution) + 20 * np.abs(vec_start[2] - vec_goal[2]) * resolution

def reconstruct_path_sdf(parent:dict, current:tuple):
    path = []
    while current in parent:
        path.append(current)
        if current == parent[current]:
            break
        current = parent[current]
    return path[::-1]

def search_close_to_sdf_map(start_index:tuple, sdf_map:np.ndarray, occupancy_map:np.ndarray, stop_distance:np.ndarray):
    start_index = tuple(start_index.flatten()) if isinstance(start_index, np.ndarray) else start_index
    open_heap = [(sdf_map[start_index], start_index)]
    open_heap_set = set()
    open_heap_set.add(start_index)
    parent = {start_index: start_index}
    visited = set()
    while len(open_heap) > 0:
        current_sdf, current = heapq.heappop(open_heap)
        open_heap_set.remove(current)
        visited.add(current)
        if current_sdf < stop_distance:
            return reconstruct_path_sdf(parent, current)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if (0 <= neighbor[0] < sdf_map.shape[0] and
                            0 <= neighbor[1] < sdf_map.shape[1] and
                            0 <= neighbor[2] < sdf_map.shape[2]):
                        if neighbor not in open_heap_set and neighbor not in visited and occupancy_map[neighbor] != 2:
                            open_heap_set.add(neighbor)
                            heapq.heappush(open_heap, (sdf_map[neighbor], neighbor))
                            parent[neighbor] = current
    return []

def search_within_sdf_map( start:tuple, goal:tuple, sdf_map:np.ndarray, occupancy_map:np.ndarray, resolution: float):
    start = tuple(start.flatten()) if isinstance(start, np.ndarray) else start
    goal = tuple(goal.flatten()) if isinstance(goal, np.ndarray) else goal
    sdf_bins = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    def get_queue_index(sdf_value: float) -> int:
        for idx, threshold in enumerate(sdf_bins):
            if sdf_value < threshold:
                return idx
        return len(sdf_bins)

    open_heaps = [[] for _ in range(len(sdf_bins) + 1)]
    open_sets = [set() for _ in range(len(sdf_bins) + 1)]
    start_queue_idx = get_queue_index(float(sdf_map[start]))
    heapq.heappush(open_heaps[start_queue_idx], (heuristic(start, goal, resolution), start))
    open_sets[start_queue_idx].add(start)
    parent = {start: start}
    visited = set()

    while True:
        queue_idx = -1
        for i, q in enumerate(open_heaps):
            if len(q) > 0:
                queue_idx = i
                break
        if queue_idx == -1:
            break

        current_cost, current = heapq.heappop(open_heaps[queue_idx])
        open_sets[queue_idx].remove(current)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return reconstruct_path_sdf(parent, current)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if (0 <= neighbor[0] < sdf_map.shape[0] and
                            0 <= neighbor[1] < sdf_map.shape[1] and
                            0 <= neighbor[2] < sdf_map.shape[2]):
                        if neighbor in visited or occupancy_map[neighbor] == 2:
                            continue
                        neighbor_sdf = float(sdf_map[neighbor])
                        neighbor_queue_idx = get_queue_index(neighbor_sdf)
                        if neighbor in open_sets[neighbor_queue_idx]:
                            continue
                        open_sets[neighbor_queue_idx].add(neighbor)
                        heapq.heappush(
                            open_heaps[neighbor_queue_idx],
                            (heuristic(neighbor, goal, resolution), neighbor),
                        )
                        if neighbor not in parent:
                            parent[neighbor] = current
    return []

class MapNode(Node):
    def __init__(self, tinynav_db_path: str, tinynav_map_path: str, verbose_timer: bool = True):
        """Initialization

        Args:
            tinynav_db_path (str): Directory to store output data.
            tinynav_map_path (str): Directory to load the pre-built map.
            verbose_timer (bool): Whether to use verbose timer output.
        """
        super().__init__('map_node')
        self.logger = logging.getLogger(__name__)
        self.timer_logger = self.logger.info if verbose_timer else self.logger.debug
        self.super_point_extractor = SuperPointTRT()
        self.light_glue_matcher = LightGlueTRT()
        self.dinov2_model = Dinov2TRT()
        self.tinynav_db_path = tinynav_db_path

        self.bridge = CvBridge()

        # subs
        self.depth_sub = Subscriber(self, Image, '/slam/keyframe_depth')
        self.keyframe_image_sub = Subscriber(self, Image, '/slam/keyframe_image')
        self.keyframe_odom_sub = Subscriber(self, Odometry, '/slam/keyframe_odom')
        self.continuous_odom_sub = self.create_subscription(Odometry, '/slam/odometry', self.continuous_odom_callback, 100)
        self.pois_sub = self.create_subscription(String, '/mapping/cmd_pois', self.pois_callback, 10)

        # pubs
        self.pose_graph_trajectory_pub = self.create_publisher(Path, "/mapping/pose_graph_trajectory", 10)
        self.relocation_pub = self.create_publisher(Odometry, '/map/relocalization', 10)
        self.current_pose_in_map_pub = self.create_publisher(Odometry, "/mapping/current_pose_in_map", 10)

        # Add stop signal subscription and data saved publisher
        self.localization_stop_sub = self.create_subscription(Bool, '/benchmark/stop', self.localization_stop_callback, 10)
        self.localization_data_saved_pub = self.create_publisher(Bool, '/benchmark/data_saved', 10)
        self.ts = TimeSynchronizer([self.keyframe_image_sub, self.keyframe_odom_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.keyframe_callback)

        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/infra2/camera_info', self.info_callback, 10)
        self.K = None
        self.baseline = None
        self.last_keyframe_image = None
        self.continuous_odom_recorder = OdomPoseRecorder(tinynav_db_path, "localization")

        self.odom = {}
        self.pose_graph_used_pose = {}
        self.relative_pose_constraint = []
        self.last_keyframe_timestamp = None
        # timestamp -> DINOv2 embedding, populated as keyframes arrive (see keyframe_mapping).
        self._embedding_cache = {}

        self.loop_similarity_threshold = 0.90
        self.loop_top_k = 1

        self.relocalization_threshold = 0.85

        os.makedirs(f"{tinynav_db_path}/nav_temp", exist_ok=True)
        self.nav_temp_db = TinyNavDB(f"{tinynav_db_path}/nav_temp", is_scratch=True)
        self.map_poses = np.load(f"{tinynav_map_path}/poses.npy", allow_pickle=True).item()
        self.map_K = np.load(f"{tinynav_map_path}/intrinsics.npy")
        self.db = TinyNavDB(tinynav_map_path, is_scratch=False)
        self.map_embeddings_idx_to_timestamp = {idx: timestamp for idx, timestamp in enumerate(self.map_poses.keys())}
        self.map_embeddings = np.stack([self.db.get_embedding(timestamp) for idx, timestamp in self.map_embeddings_idx_to_timestamp.items()])
        self.occupancy_map = np.load(f"{tinynav_map_path}/occupancy_grid.npy")
        self.occupancy_map_meta = np.load(f"{tinynav_map_path}/occupancy_meta.npy")
        self.sdf_map = np.load(f"{tinynav_map_path}/sdf_map.npy")

        print(f"sdf_map.shape: {self.sdf_map.shape}")
        print(f"occupancy_map.shape: {self.occupancy_map.shape}")

        self.relocalization_poses = {}
        self.relocalization_pose_weights = {}
        self.failed_relocalizations = []

        # Lock-once relocalization: relocalize every keyframe only until we have a
        # consistent fix, then freeze T_from_map_to_odom and ride odom from there.
        # DINOv2 retrieval can match a wrong-but-similar place on a single frame, so
        # we don't trust the first success -- we require the most recent few
        # observations of T_from_map_to_odom to agree spatially before locking.
        # Sliding window (not fill-then-clear) so a stray bad observation can't keep
        # resetting the count.
        self.reloc_lock_window = 3          # recent observations that must agree
        self.reloc_lock_tol = 0.3           # meters; max pairwise translation spread
        self._reloc_obs_window = []         # recent observation_T_from_map_to_odom (4x4)

        self.T_from_map_to_odom = None

        self.pois = {}
        self.poi_meta = {}
        self.poi_index = -1
        self._nav_completed = False
        self._leg_initial_length: float | None = None
        self._leg_start_time: float | None = None
        self._speed_estimate: float | None = None

        self.poi_pub = self.create_publisher(Odometry, "/mapping/poi", 10)
        self.poi_change_pub = self.create_publisher(Odometry, "/mapping/poi_change", 10)
        self.nav_done_pub = self.create_publisher(Bool, '/mapping/nav_done', 10)
        self.nav_progress_pub = self.create_publisher(String, '/mapping/nav_progress', 10)

        self.current_pose_pub = self.create_publisher(Odometry, "/mapping/current_pose", 10)
        self.global_plan_pub = self.create_publisher(Path, '/mapping/global_plan', 10)
        self.target_pose_pub = self.create_publisher(Odometry, "/control/target_pose", 10)
        # Static openness prior: min obstacle-distance over the upcoming global segment.
        self.path_openness_pub = self.create_publisher(Float32, '/mapping/path_openness', 10)
        self._openness2d = None  # lazily-built 2D obstacle EDT (m) over the robot z-band
        self._openness_map_id = None  # id() of occupancy_map the EDT was built from

        self.tf_broadcaster = TransformBroadcaster(self)

        self._save_completed = False

        # Persistent event loop reused across all TRT inferences. asyncio.run()
        # builds and tears down a fresh loop on every call, which is pure overhead
        # at keyframe rate.
        self._loop = asyncio.new_event_loop()

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def pois_callback(self, msg: String):
        self.get_logger().info("Received POIs from planner: " + msg.data)
        try:
            raw_pois = json.loads(msg.data)

            pois_dict = {}
            poi_meta = {}
            keys = sorted([int(key) for key in raw_pois.keys()])
            for index, key in enumerate(keys):
                raw_poi = raw_pois[str(key)]
                pois_dict[index] = np.array(raw_poi["position"])
                poi_meta[index] = {
                    "id": raw_poi.get("id", key),
                    "name": raw_poi.get("name"),
                }
            self.pois = pois_dict
            self.poi_meta = poi_meta

            if not self.pois:
                self.poi_index = -1
                # Signal planning_node to clear target_pose so it stops publishing paths
                dummy_pose = np.eye(4)
                self.poi_change_pub.publish(np2msg(dummy_pose, self.get_clock().now().to_msg(), "world", "map"))
                self.poi_meta = {}
                self.get_logger().info("POIs cleared, navigation cancelled")
                return

            self.poi_index = min(0, len(self.pois) - 1)
            self._nav_completed = False
            self._leg_initial_length = None
            self._leg_start_time = None
            self._speed_estimate = None
            self.get_logger().info(f"Parsed POIs: {self.pois}")
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse POIs JSON: {e}")
            self.pois = {}
            self.poi_meta = {}

    def _nav_progress_payload(self, *, percent: float, path_remaining_m: float,
                              path_total_m: float, estimated_remaining_s: float) -> dict:
        meta = self.poi_meta.get(self.poi_index, {})
        return {
            "poi_index": self.poi_index,  # route index in the current command queue
            "poi_id": meta.get("id"),
            "poi_name": meta.get("name"),
            "percent": percent,
            "path_remaining_m": path_remaining_m,
            "path_total_m": path_total_m,
            "estimated_remaining_s": estimated_remaining_s,
        }

    def info_callback(self, msg:CameraInfo):
        if self.K is None:
            self.get_logger().info("Camera intrinsics received.")
            self.K = np.array(msg.k).reshape(3, 3)
            fx = self.K[0, 0]
            Tx = msg.p[3]
            self.baseline = -Tx / fx
            self.destroy_subscription(self.camera_info_sub)

    def continuous_odom_callback(self, odom_msg: Odometry):
        self.continuous_odom_recorder.record_odometry_msg(odom_msg)

    def localization_stop_callback(self, msg: Bool):
        if msg.data:
            self.get_logger().info("Received benchmark stop signal, starting save process...")
            try:
                self.save_relocalization_poses()
                self.get_logger().info("Localization save completed successfully")

                # Publish save finished signal
                save_finished_msg = Bool()
                save_finished_msg.data = True
                self.localization_data_saved_pub.publish(save_finished_msg)
                self.get_logger().info("Published data save finished signal")

            except Exception as e:
                self.get_logger().error(f"Error during localization save: {e}")
                # Still publish completion signal even if there was an error
                save_finished_msg = Bool()
                save_finished_msg.data = False
                self.localization_data_saved_pub.publish(save_finished_msg)

    def keyframe_callback(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, depth_msg:Image):
        if self.K is None:
            return
        image = self.bridge.imgmsg_to_cv2(keyframe_image_msg, desired_encoding="mono8")
        # Compute SuperPoint features and DINOv2 embedding ONCE here and reuse them
        # in both mapping and relocalization. Previously each was recomputed inside
        # keyframe_mapping and keyframe_relocalization, doubling the TRT inference
        # cost of every keyframe.
        features = self._run(self.super_point_extractor.infer(image))
        embedding = self.get_embeddings(image)

        self.keyframe_mapping(keyframe_image_msg, keyframe_odom_msg, depth_msg, image, features, embedding)

        keyframe_image_timestamp_ns = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(keyframe_image_msg.header.stamp.nanosec)
        # Once locked, T_from_map_to_odom is frozen and we ride odom -- skip
        # relocalization entirely (also saves the TRT match/PnP cost per keyframe).
        if self.T_from_map_to_odom is None:
            success, pose_in_world = self.keyframe_relocalization(keyframe_image_msg.header.stamp, image, features, embedding)
            if success:
                self.try_lock_transform_from_map_to_odom(keyframe_image_timestamp_ns)

        with Timer(name = "nav path", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            self.try_publish_nav_path(keyframe_image_timestamp_ns)
            # timer or queue for publish the nav path
            # and record the map pose
            # compute the coordinate transform from the map pose to the keyframe pose
            # publish the nav path from the map pose to the keyframe pose with the cost map

    def keyframe_mapping_with_timer(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, depth_msg:Image, image, features, embedding):
        with Timer(name="Mapping Loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            self.keyframe_mapping(keyframe_image_msg, keyframe_odom_msg, depth_msg, image, features, embedding)

    def keyframe_mapping(self, keyframe_image_msg:Image, keyframe_odom_msg:Odometry, depth_msg:Image, image, features, embedding):
        if self.K is None:
            return
        keyframe_image_timestamp = int(keyframe_image_msg.header.stamp.sec * 1e9) + int(keyframe_image_msg.header.stamp.nanosec)
        keyframe_odom_timestamp = int(keyframe_odom_msg.header.stamp.sec * 1e9) + int(keyframe_odom_msg.header.stamp.nanosec)
        depth_timestamp = int(depth_msg.header.stamp.sec * 1e9) + int(depth_msg.header.stamp.nanosec)
        assert keyframe_image_timestamp == keyframe_odom_timestamp
        assert keyframe_image_timestamp == depth_timestamp
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        odom, _ = msg2np(keyframe_odom_msg)
        rgb_image_place_holder = einops.repeat(image, "h w -> h w c", c = 3)

        self.nav_temp_db.set_entry(keyframe_image_timestamp, depth = depth, infra1_image = image, rgb_image = rgb_image_place_holder)
        self.nav_temp_db.set_entry(keyframe_image_timestamp, embedding = embedding)
        self.nav_temp_db.set_entry(keyframe_image_timestamp, features = features)
        # In-memory embedding cache so find_loop does not re-read every keyframe's
        # embedding from the DB on each call (was O(N) DB reads per keyframe -> O(N^2)).
        self._embedding_cache[keyframe_image_timestamp] = embedding

        if len(self.odom) == 0 and self.last_keyframe_timestamp is None:
            self.odom[keyframe_odom_timestamp] = odom
            self.pose_graph_used_pose[keyframe_odom_timestamp] = odom
        else:
            last_keyframe_odom_pose = self.odom[self.last_keyframe_timestamp]
            T_prev_curr = np.linalg.inv(last_keyframe_odom_pose) @ odom
            self.relative_pose_constraint.append((keyframe_image_timestamp, self.last_keyframe_timestamp, T_prev_curr))
            self.pose_graph_used_pose[keyframe_image_timestamp] = odom
            self.odom[keyframe_image_timestamp] = odom
            def find_loop_and_pose_graph(timestamp):
                    target_embedding = self._embedding_cache[timestamp]
                    valid_timestamp = [t for t in self.pose_graph_used_pose.keys() if t + 10 * 1e9 < timestamp]
                    valid_embeddings = np.array([self._embedding_cache[t] for t in valid_timestamp])

                    idx_to_timestamp = {i:t for i, t in enumerate(valid_timestamp)}
                    with Timer(name = "find loop", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        loop_list = find_loop(target_embedding, valid_embeddings, self.loop_similarity_threshold, self.loop_top_k)
                    with Timer(name = "Relative pose estimation", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        for idx, similarity in loop_list:
                            prev_timestamp = idx_to_timestamp[idx]
                            curr_timestamp = timestamp
                            prev_depth, _, prev_features, _, _ = self.nav_temp_db.get_depth_embedding_features_images(prev_timestamp)
                            curr_depth, _, curr_features, _, _ = self.nav_temp_db.get_depth_embedding_features_images(curr_timestamp)
                            prev_matched_keypoints, curr_matched_keypoints, matches = self.match_keypoints(prev_features, curr_features)
                            success, T_prev_curr, _, _, inliers = estimate_pose(prev_matched_keypoints, curr_matched_keypoints, curr_depth, self.K)
                            if success and len(inliers) >= 100:
                                self.relative_pose_constraint.append((curr_timestamp, prev_timestamp, T_prev_curr))
                                #print(f"Added loop relative pose constraint: {curr_timestamp} -> {prev_timestamp}")
                    with Timer(name = "solve pose graph", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                        self.pose_graph_used_pose = solve_pose_graph(self.pose_graph_used_pose, self.relative_pose_constraint, max_iteration_num = 5)
            find_loop_and_pose_graph(keyframe_image_timestamp)
            self.pose_graph_trajectory_publish(keyframe_image_timestamp)
        self.last_keyframe_timestamp = keyframe_odom_timestamp
        self.last_keyframe_image = image


    def get_embeddings(self, image: np.ndarray) -> np.ndarray:
        # shape: (1, 768)
        return self._run(self.dinov2_model.infer(image))

    def match_keypoints(self, feats0:dict, feats1:dict, image_shape = np.array([848, 480], dtype = np.int64)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        match_result = self._run(self.light_glue_matcher.infer(feats0["kpts"], feats1["kpts"], feats0['descps'], feats1['descps'], feats0['mask'], feats1['mask'], image_shape, image_shape))
        match_indices = match_result["match_indices"][0]
        valid_mask = match_indices != -1
        keypoints0 = feats0["kpts"][0][valid_mask]
        keypoints1 = feats1["kpts"][0][match_indices[valid_mask]]
        matches = []
        for i, index in enumerate(match_indices):
            if index != -1:
                matches.append([i, index])
        return keypoints0, keypoints1, np.array(matches, dtype=np.int64)

    def pose_graph_trajectory_publish(self, timestamp):
        path_msg = Path()
        path_msg.header.stamp.sec = int(timestamp / 1e9)
        path_msg.header.stamp.nanosec = int(timestamp % 1e9)
        path_msg.header.frame_id = "world"
        for t, pose_in_world in self.pose_graph_used_pose.items():
            pose = PoseStamped()
            pose.header = path_msg.header
            t = pose_in_world[:3, 3]
            quat = matrix_to_quat(pose_in_world[:3, :3])
            pose.pose.position.x = t[0]
            pose.pose.position.y = t[1]
            pose.pose.position.z = t[2]
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            path_msg.poses.append(pose)
        self.pose_graph_trajectory_pub.publish(path_msg)

    def relocalize_with_depth(self, keyframe: np.ndarray, keyframe_features: dict, K: np.ndarray | None, query_embedding: np.ndarray) -> tuple[bool, np.ndarray, float]:
        if K is None:
            return False, np.eye(4), -np.inf
        query_embedding_normed = query_embedding / np.linalg.norm(query_embedding)

        idx_and_similarity_array = find_loop(query_embedding_normed, self.map_embeddings, self.relocalization_threshold, 1)
        max_similarity = np.max([similarity for _, similarity in idx_and_similarity_array]) if len(idx_and_similarity_array) > 0 else 0
        if len(idx_and_similarity_array) == 0:
            print(f"not enough similar embeddings to relocalize, {len(idx_and_similarity_array)}, max_similarity : {max_similarity}")
            return False, np.eye(4), -np.inf

        pnp_candidates = []
        for idx_in_map, similarity in idx_and_similarity_array:
            timestamp_in_map = self.map_embeddings_idx_to_timestamp[idx_in_map]
            reference_keyframe_pose = self.map_poses[timestamp_in_map]
            reference_depth, _, reference_features, _, _ = self.db.get_depth_embedding_features_images(timestamp_in_map)
            reference_matched_keypoints, keyframe_matched_keypoints, matches = self.match_keypoints(reference_features, keyframe_features)
            if len(matches) < 50:
                print(f"not enough matched features to relocalize, {len(matches)} < 50")
                continue

            point_3d_in_world, inliers = self.keypoint_with_depth_to_3d(reference_matched_keypoints, reference_depth, reference_keyframe_pose, self.map_K)
            point_3d_in_world_list = point_3d_in_world[inliers]
            point_2d_in_keyframe_list = keyframe_matched_keypoints[inliers]
            point_count = len(point_2d_in_keyframe_list)
            if point_count <= 80:
                print(f"not enough landmarks to relocalize, {point_count}")
                continue
            pnp_candidates.append((point_3d_in_world_list, point_2d_in_keyframe_list))

        success, best_pose_in_camera, pose_cov_weight, _, _, _ = rerank_by_pnp_inliers(pnp_candidates, self.map_K)
        if success:
            print(f"relocalization pose : {best_pose_in_camera}")
            return True, best_pose_in_camera, pose_cov_weight

        print("no valid PnP relocalization candidate found")
        return False, np.eye(4), -np.inf

    def keypoint_with_depth_to_3d(self, keypoints:np.ndarray, depth:np.ndarray, pose_from_camera_to_world:np.ndarray, K:np.ndarray):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        # Vectorized projection of all keypoints (was a per-keypoint Python loop).
        u = keypoints[:, 0].astype(np.int64)
        v = keypoints[:, 1].astype(np.int64)
        Z = depth[v, u]
        inliers = (Z > 0) & (Z < 50)
        X = np.where(inliers, (u - cx) * Z / fx, 0.0)
        Y = np.where(inliers, (v - cy) * Z / fy, 0.0)
        # shape: (N, 3)
        point_in_camera = np.stack([X, Y, Z], axis=1)
        rotation = pose_from_camera_to_world[:3, :3]
        translation = pose_from_camera_to_world[:3,3]

        point_in_world = (rotation @ point_in_camera.T).T + translation
        return point_in_world, inliers

    @Timer(name="Relocalization loop", text="\n\n[{name}] Elapsed time: {milliseconds:.0f} ms")
    def keyframe_relocalization(self, timestamp, image:np.ndarray, features:dict, embedding:np.ndarray) -> tuple[bool, np.ndarray]:
        res, pose_in_camera, pose_cov_weight = self.relocalize_with_depth(image, features, self.K, embedding)
        if res:
            # publish the relocalization pose for debug
            pose_in_world = np.linalg.inv(pose_in_camera)
            timestamp_ns = int(timestamp.sec * 1e9) + int(timestamp.nanosec)
            self.relocation_pub.publish(np2msg(pose_in_world, timestamp, "world", "camera"))
            self.relocalization_poses[timestamp_ns] = pose_in_world
            self.relocalization_pose_weights[timestamp_ns] = pose_cov_weight
            return True, pose_in_world
        else:
            self.failed_relocalizations.append(timestamp)
            return False, np.eye(4)

    def save_relocalization_poses(self):
        if self._save_completed:
            self.get_logger().info("Relocalization data already saved, skipping duplicate save")
            return

        print("saving localization data...")
        self.continuous_odom_recorder.save_to_disk()

        if len(self.relocalization_poses) == 0:
            self.get_logger().warning("No relocalization poses found - not saving")
            return

        np.save(f"{self.tinynav_db_path}/relocalization_poses.npy", self.relocalization_poses, allow_pickle=True)
        np.save(f"{self.tinynav_db_path}/relocalization_pose_weights.npy", self.relocalization_pose_weights, allow_pickle=True)
        np.save(f"{self.tinynav_db_path}/failed_relocalizations.npy", self.failed_relocalizations, allow_pickle=True)
        np.save(f"{self.tinynav_db_path}/poses.npy", self.pose_graph_used_pose, allow_pickle=True)

        logging.info(f"Saved {len(self.relocalization_poses)} relocalization poses to {self.tinynav_db_path}")
        logging.info(f"Failed relocalizations count: {len(self.failed_relocalizations)}")

        self._save_completed = True

    def destroy_node(self):
        try:
            self.save_relocalization_poses()
            self.nav_temp_db.close()
            self.db.close()
            super().destroy_node()
        except Exception:
            # Ignore errors during destruction as resources may already be freed
            pass


    def try_lock_transform_from_map_to_odom(self, timestamp: int):
        """Lock T_from_map_to_odom from a single consistent burst of relocalizations.

        Each successful relocalization gives one observation of the (constant)
        map->odom transform. DINOv2 retrieval can occasionally match a wrong place,
        so we don't trust one observation: we keep a sliding window of the most
        recent few and only lock once they all agree spatially (pairwise translation
        spread <= reloc_lock_tol). Once locked we never recompute -- the caller stops
        relocalizing and rides odom.
        """
        if timestamp not in self.pose_graph_used_pose:
            return
        camera_in_map_world = self.relocalization_poses[timestamp]
        camera_in_odom_world = self.pose_graph_used_pose[timestamp]
        observation_T_from_map_to_odom = camera_in_odom_world @ np.linalg.inv(camera_in_map_world)

        self._reloc_obs_window.append(observation_T_from_map_to_odom)
        self._reloc_obs_window = self._reloc_obs_window[-self.reloc_lock_window:]
        if len(self._reloc_obs_window) < self.reloc_lock_window:
            return

        translations = np.array([T[:3, 3] for T in self._reloc_obs_window])
        spread = float(np.max([
            np.linalg.norm(translations[i] - translations[j])
            for i in range(len(translations))
            for j in range(i + 1, len(translations))
        ]))
        if spread > self.reloc_lock_tol:
            self.get_logger().info(
                f"[reloc-lock] {len(self._reloc_obs_window)} obs not consistent yet, "
                f"spread={spread:.2f}m > tol={self.reloc_lock_tol}m")
            return

        # Consistent burst -> lock to the most recent observation and freeze.
        self.T_from_map_to_odom = observation_T_from_map_to_odom
        self.get_logger().info(
            f"[reloc-lock] locked T_from_map_to_odom (spread={spread:.2f}m over "
            f"{self.reloc_lock_window} obs)")

    def try_publish_nav_path(self, timestamp: int):
        if self.T_from_map_to_odom is None:
            self.get_logger().info("Relocalization not successful yet, skip publishing nav path")
            return

        if self.poi_index == -1:
            self.get_logger().info("No POI found, skip publishing nav path")
            return

        if self.poi_index >= len(self.pois):
            self.get_logger().info("All POIs have been visited, skip publishing nav path")
            return

        poi = self.pois[self.poi_index]
        poi_pose = np.eye(4)
        poi_pose[:3, 3] = poi
        self.poi_pub.publish(np2msg(poi_pose, self.get_clock().now().to_msg(), "world", "map"))
        # get the pose from the map to the odom
        pose_in_map = np.linalg.inv(self.T_from_map_to_odom) @ self.pose_graph_used_pose[timestamp]
        self.current_pose_in_map_pub.publish(np2msg(pose_in_map, self.get_clock().now().to_msg(), "world", "map"))

        pose_in_map_position = pose_in_map[:3, 3]

        while self.poi_index < len(self.pois):
            poi = self.pois[self.poi_index]
            diff_position_norm_xy = np.linalg.norm(poi[:2] - pose_in_map_position[:2])
            diff_position_norm_z = np.linalg.norm(poi[2] - pose_in_map_position[2])
            if diff_position_norm_xy < 0.5 and diff_position_norm_z < 2.0:
                arrived_msg = String()
                arrived_msg.data = json.dumps(self._nav_progress_payload(
                    percent=100.0,
                    path_remaining_m=0.0,
                    path_total_m=round(self._leg_initial_length or 0.0, 2),
                    estimated_remaining_s=0.0,
                ))
                self.nav_progress_pub.publish(arrived_msg)
                self.poi_index += 1
                self._leg_initial_length = None
                self._leg_start_time = None
                dummy_pose = np.eye(4)

                stamp_msg = self.get_clock().now().to_msg()
                stamp_msg.sec = int(timestamp / 1e9)
                stamp_msg.nanosec = int(timestamp % 1e9)
                self.poi_change_pub.publish(np2msg(dummy_pose, stamp_msg, "world", "map"))
                continue
            else:
                break

        if self.poi_index >= len(self.pois):
            if not self._nav_completed:
                self._nav_completed = True
                self.get_logger().info("All POIs have been visited, nav done")
                self.nav_done_pub.publish(Bool(data=True))
            return

        target_poi = self.pois[self.poi_index]
        with Timer(name = "generate nav path in map", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
            paths_in_map = self.generate_nav_path_in_map(pose_in_map = pose_in_map, target_poi = target_poi)

        if paths_in_map is not None:
            remaining_length = sum(
                np.linalg.norm(paths_in_map[i + 1] - paths_in_map[i])
                for i in range(len(paths_in_map) - 1)
            ) if len(paths_in_map) > 1 else 0.0

            now = time.time()
            if self._leg_initial_length is None:
                self._leg_initial_length = remaining_length
                self._leg_start_time = now

            covered = self._leg_initial_length - remaining_length
            elapsed = now - self._leg_start_time
            if covered > 0.1 and elapsed > 1.0:
                self._speed_estimate = covered / elapsed

            initial = self._leg_initial_length
            percent = max(0.0, min(100.0, covered / initial * 100.0)) if initial > 0 else 0.0
            estimated_remaining_s = remaining_length / self._speed_estimate if self._speed_estimate else -1.0

            progress_msg = String()
            progress_msg.data = json.dumps(self._nav_progress_payload(
                percent=round(percent, 1),
                path_remaining_m=round(remaining_length, 2),
                path_total_m=round(initial, 2),
                estimated_remaining_s=round(estimated_remaining_s, 1),
            ))
            self.nav_progress_pub.publish(progress_msg)

            # use the max_speed to publish the position the robot should be after 5 seconds
            with Timer(name = "Find target position", text="[{name}] Elapsed time: {milliseconds:.0f} ms", logger=self.timer_logger):
                max_speed = 0.5

                # local target = furthest point on the path reachable from the robot
                # before the heading turns past TURN_THRESH (a corner) or LOOKAHEAD_MAX
                # is reached. Drives to corners instead of slicing across them.
                start_i = local_i = 0
                target_position = paths_in_map[0]
                if len(paths_in_map) > 1:
                    robot_xy = np.asarray(pose_in_map_position[:2])
                    start_i = int(np.argmin([np.linalg.norm(np.asarray(p[:2]) - robot_xy) for p in paths_in_map]))
                    local_i = self._local_target_index(paths_in_map, start_i, lookahead_max=max_speed * 5)
                    target_position = paths_in_map[local_i]

                target_position_in_map = np.asarray(target_position[:3])
                pose_in_origin_odom = self.odom[timestamp]
                T = pose_in_origin_odom @ np.linalg.inv(pose_in_map)
                target_position_in_odom = T[:3, :3] @ target_position_in_map + T[:3, 3]
                dummy_pose = np.eye(4)
                dummy_pose[:3, 3] = target_position_in_odom

                self.target_pose_pub.publish(np2msg(dummy_pose, self.get_clock().now().to_msg(), "world", "camera"))
                path_msg = Path()
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.header.frame_id = "map"
                for x, y, z in paths_in_map:
                    pose = PoseStamped()
                    pose.header = path_msg.header
                    pose.pose.position.x = x
                    pose.pose.position.y = y
                    pose.pose.position.z = z
                    pose.pose.orientation.x = 0.0
                    pose.pose.orientation.y = 0.0
                    pose.pose.orientation.z = 0.0
                    pose.pose.orientation.w = 1.0
                    path_msg.poses.append(pose)
                self.global_plan_pub.publish(path_msg)

                # Static openness prior: min obstacle-distance over the robot -> local
                # target segment, for the planner to size safety.
                self._ensure_openness_map()
                openness = self._path_openness(paths_in_map[start_i:local_i + 1])
                if np.isfinite(openness):
                    self.path_openness_pub.publish(Float32(data=float(openness)))

                self.tf_broadcaster.sendTransform(np2tf(T, self.get_clock().now().to_msg(), "world", "map"))
        else:
            logging.info("No path found in map")

    def _ensure_openness_map(self):
        """Build a 2D obstacle-distance field (m) by collapsing occupied cells over a
        z-band around the working height (median z of the recorded mapping poses) +-
        0.4 m, then EDT. Reflects static corridor width. Recomputed whenever the
        occupancy_map is reloaded (map switch / handoff)."""
        if self._openness2d is not None and self._openness_map_id == id(self.occupancy_map):
            return
        self._openness_map_id = id(self.occupancy_map)
        origin = self.occupancy_map_meta[:3]
        res = float(self.occupancy_map_meta[3])
        work_z = float(np.median([np.asarray(p)[2, 3] for p in self.map_poses.values()]))
        z_dim = self.occupancy_map.shape[2]
        z_world = origin[2] + (np.arange(z_dim) + 0.5) * res
        band = (z_world >= work_z - 0.4) & (z_world <= work_z + 0.4)
        if band.any():
            occ2d = (self.occupancy_map[:, :, band] == 2).any(axis=2)
        else:
            occ2d = (self.occupancy_map == 2).any(axis=2)
        self._openness2d = distance_transform_edt(~occ2d) * res

    def _local_target_index(self, path, start_i, lookahead_max, min_lookahead=1.0,
                            turn_thresh=np.deg2rad(45.0), smooth_m=0.4):
        """Index of the local target: walk forward from start_i, stop at the first
        point (beyond min_lookahead) whose smoothed heading has turned >= turn_thresh
        from the entry heading (a corner), or when lookahead_max arc length is reached.
        Never returns a point closer than min_lookahead in arc length."""
        pxy = [np.asarray(p[:2], dtype=np.float64) for p in path]
        n = len(pxy)
        cum = [0.0] * n
        for i in range(start_i + 1, n):
            cum[i] = cum[i - 1] + float(np.linalg.norm(pxy[i] - pxy[i - 1]))

        def sdir(i):
            j = i
            while j < n - 1 and (cum[j] - cum[i]) < smooth_m:
                j += 1
            d = pxy[j] - pxy[i]
            L = float(np.linalg.norm(d))
            return d / L if L > 1e-6 else None

        entry = sdir(start_i)
        li = start_i
        for k in range(start_i + 1, n):
            if cum[k] - cum[start_i] >= lookahead_max:
                li = k
                break
            dk = sdir(k)
            if (cum[k] - cum[start_i]) >= min_lookahead and entry is not None and dk is not None:
                turn = abs(np.arctan2(dk[0] * entry[1] - dk[1] * entry[0], float(dk @ entry)))
                if turn >= turn_thresh:
                    li = k
                    break
            li = k
        return li

    def _path_openness(self, path_in_map: np.ndarray) -> float:
        """Min static obstacle-distance (m) over the given path segment."""
        if self._openness2d is None or len(path_in_map) == 0:
            return float('inf')
        origin = self.occupancy_map_meta[:3]
        res = float(self.occupancy_map_meta[3])
        nx, ny = self._openness2d.shape
        vals = []
        for p in path_in_map:
            ix = int((p[0] - origin[0]) / res); iy = int((p[1] - origin[1]) / res)
            if 0 <= ix < nx and 0 <= iy < ny:
                vals.append(float(self._openness2d[ix, iy]))
        return min(vals) if vals else float('inf')

    def generate_nav_path_in_map(self, pose_in_map: np.ndarray, target_poi: np.ndarray) -> np.ndarray:
        dummy_poi_pose = np.eye(4)
        dummy_poi_pose[:3, 3] = target_poi
        self.poi_pub.publish(np2msg(dummy_poi_pose, self.get_clock().now().to_msg(), "world", "map"))
        occupancy_map_origin = self.occupancy_map_meta[:3]
        resolution = self.occupancy_map_meta[3]
        start_idx = np.array([
            int((pose_in_map[0, 3] - occupancy_map_origin[0]) / resolution),
            int((pose_in_map[1, 3] - occupancy_map_origin[1]) / resolution),
            int((pose_in_map[2, 3] - occupancy_map_origin[2]) / resolution)
        ], dtype=np.int32)
        poi_goal_idx = np.array([
            int((target_poi[0] - occupancy_map_origin[0]) / resolution),
            int((target_poi[1] - occupancy_map_origin[1]) / resolution),
            int((target_poi[2] - occupancy_map_origin[2]) / resolution)
        ], dtype=np.int32)

        if (
            start_idx[0] < 0
            or start_idx[0] >= self.occupancy_map.shape[0]
            or start_idx[1] < 0
            or start_idx[1] >= self.occupancy_map.shape[1]
            or start_idx[2] < 0
            or start_idx[2] >= self.occupancy_map.shape[2]
            or poi_goal_idx[0] < 0
            or poi_goal_idx[0] >= self.occupancy_map.shape[0]
            or poi_goal_idx[1] < 0
            or poi_goal_idx[1] >= self.occupancy_map.shape[1]
            or poi_goal_idx[2] < 0
            or poi_goal_idx[2] >= self.occupancy_map.shape[2]
        ):
            return None 
        sdf_start_path = search_close_to_sdf_map(start_idx, self.sdf_map, self.occupancy_map, 0.2)
        sdf_goal_path = search_close_to_sdf_map(poi_goal_idx, self.sdf_map, self.occupancy_map, 0.2)

        # search_close_to_sdf_map returns [] when no free cell within stop_distance is
        # reachable from start/goal (robot or POI sits in/against an obstacle). Treat it
        # like an out-of-bounds index: bail with None instead of indexing [-1] and
        # crashing the whole map_node process (which stalls nav -> nav_done never fires
        # -> map handoff never triggers).
        if len(sdf_start_path) == 0 or len(sdf_goal_path) == 0:
            self.get_logger().warning(
                f"search_close_to_sdf_map found no free cell: "
                f"start_empty={len(sdf_start_path) == 0}, goal_empty={len(sdf_goal_path) == 0}"
            )
            return None

        sdf_start_sdf = sdf_start_path[-1]
        sdf_goal_sdf = sdf_goal_path[-1]
        path_sdf = search_within_sdf_map(sdf_start_sdf, sdf_goal_sdf, self.sdf_map, self.occupancy_map, resolution)
        if len(path_sdf) == 0:
            self.get_logger().warning(
                f"search_within_sdf_map returned empty path: start_idx={tuple(sdf_start_sdf)}, goal_idx={tuple(sdf_goal_sdf)}"
            )
        path = sdf_start_path + path_sdf + sdf_goal_path[::-1]
        if len(path) > 0:
            converted_path = np.array(path) * resolution + occupancy_map_origin
            return converted_path
        return None

def main(args=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinynav_db_path", type=str, default="tinynav_temp")
    parser.add_argument("--tinynav_map_path", type=str, required=True)
    parser.add_argument("--verbose_timer", action="store_true", default=True, help="Enable verbose timer output")
    parser.add_argument("--no_verbose_timer", dest="verbose_timer", action="store_false", help="Disable verbose timer output")
    parsed_args, unknown_args = parser.parse_known_args(sys.argv[1:])
    node = MapNode(tinynav_db_path=parsed_args.tinynav_db_path,
                   tinynav_map_path=parsed_args.tinynav_map_path,
                   verbose_timer=parsed_args.verbose_timer)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
