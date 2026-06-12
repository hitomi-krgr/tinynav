"""Visualize the computed WALKABLE region (size + connectivity) in RViz.

Pure observability add-on: subscribes to an obstacle OccupancyGrid (default the
existing /planning/obstacle_mask), computes the free-space connected components,
and publishes a MarkerArray that colors each walkable component and annotates
its area in m^2. Lets you SEE, on the real robot, how big the actually-computed
walkable region is and whether it stays connected.

Touches nothing in planning_node. Run alongside:
    python3 tinynav/core/walkable_viz_node.py
    # or point at a different mask topic:
    python3 tinynav/core/walkable_viz_node.py --mask_topic /planning/obstacle_mask

Decode convention matches planning_node.publish_obstacle_mask:
    data = mask.ravel(order='F'); width = mask.shape[1] (NY); height = mask.shape[0] (NX)
    cell (i,j) center world = (origin.x + (i+0.5)*res, origin.y + (j+0.5)*res)
    obstacle == 100, free == 0.
"""
import argparse
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Point
from scipy.ndimage import label

# distinct-ish color palette for components (RGB 0..1)
_PALETTE = [
    (0.20, 0.80, 0.20), (0.20, 0.55, 0.95), (0.95, 0.75, 0.10),
    (0.80, 0.30, 0.80), (0.30, 0.85, 0.85), (0.95, 0.45, 0.20),
    (0.60, 0.80, 0.30), (0.50, 0.50, 0.95),
]


class WalkableVizNode(Node):
    def __init__(self, mask_topic: str, min_area_m2: float, z_offset: float):
        super().__init__('walkable_viz_node')
        self.min_area_m2 = min_area_m2
        self.z_offset = z_offset       # lower markers below the ESDF cloud
        self.sub = self.create_subscription(OccupancyGrid, mask_topic, self.cb, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/planning/walkable_viz', 10)
        self.get_logger().info(
            f"walkable_viz: subscribing {mask_topic} -> /planning/walkable_viz "
            f"(min_area highlight={min_area_m2} m^2)")

    def cb(self, msg: OccupancyGrid):
        w = msg.info.width        # NY
        h = msg.info.height       # NX
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        oz = msg.info.origin.position.z + self.z_offset
        if w == 0 or h == 0:
            return
        data = np.asarray(msg.data, dtype=np.int16).reshape((h, w), order='F')  # (NX,NY)
        free = data == 0          # walkable cells (obstacle==100, unknown(-1) excluded)

        lbl, n = label(free, structure=np.ones((3, 3)))   # 8-connectivity
        cell_area = res * res
        markers = MarkerArray()
        markers.markers.append(self._clear_marker(msg.header))

        areas = []
        for c in range(1, n + 1):
            ii, jj = np.where(lbl == c)
            area = len(ii) * cell_area
            areas.append((c, area, ii, jj))
        # largest first so colors are stable-ish and big regions get palette[0]
        areas.sort(key=lambda t: -t[1])

        for rank, (c, area, ii, jj) in enumerate(areas):
            color = _PALETTE[rank % len(_PALETTE)]
            big = area >= self.min_area_m2
            a = 0.85 if big else 0.35      # fade tiny components
            cube = Marker()
            cube.header = msg.header
            cube.ns = 'walkable_cells'
            cube.id = rank
            cube.type = Marker.CUBE_LIST
            cube.action = Marker.ADD
            cube.scale.x = res
            cube.scale.y = res
            cube.scale.z = 0.02
            cube.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=a)
            cube.pose.orientation.w = 1.0
            cube.points = [
                Point(x=float(ox + (i + 0.5) * res),
                      y=float(oy + (j + 0.5) * res),
                      z=float(oz))
                for i, j in zip(ii.tolist(), jj.tolist())
            ]
            markers.markers.append(cube)

            # area label at centroid
            txt = Marker()
            txt.header = msg.header
            txt.ns = 'walkable_area'
            txt.id = rank
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.scale.z = 0.18
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            cx = ox + (float(ii.mean()) + 0.5) * res
            cy = oy + (float(jj.mean()) + 0.5) * res
            txt.pose.position = Point(x=float(cx), y=float(cy), z=float(oz + 0.25))
            txt.pose.orientation.w = 1.0
            txt.text = f"{area:.2f} m^2"
            markers.markers.append(txt)

        self.marker_pub.publish(markers)
        total = sum(a for _, a, _, _ in areas)
        biggest = areas[0][1] if areas else 0.0
        self.get_logger().info(
            f"walkable: {n} components, total={total:.2f} m^2, largest={biggest:.2f} m^2",
            throttle_duration_sec=1.0)

    def _clear_marker(self, header: Header) -> Marker:
        m = Marker()
        m.header = header
        m.action = Marker.DELETEALL
        return m


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_topic', type=str, default='/planning/obstacle_mask')
    parser.add_argument('--min_area_m2', type=float, default=0.5,
                        help='components below this are faded (likely noise/obstacle islands)')
    parser.add_argument('--z_offset', type=float, default=-0.6,
                        help='vertical offset for markers; negative drops them below '
                             'the ESDF cloud so they stop occluding it')
    args, _ = parser.parse_known_args((argv or sys.argv)[1:])

    rclpy.init()
    node = WalkableVizNode(args.mask_topic, args.min_area_m2, args.z_offset)
    try:
        rclpy.spin(node)
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
