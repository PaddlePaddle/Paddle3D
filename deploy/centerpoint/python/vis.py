# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import open3d
from infer import init_predictor, parse_args, preprocess, run


def draw_result(points, result_boxes=None, color=(0, 1, 0)):

    # config
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # raw point cloud
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    vis.add_geometry(pts)

    # result_boxes
    for i in range(result_boxes.shape[0]):
        lines = boxes_to_lines(result_boxes[i])
        lines.paint_uniform_color(color)
        vis.add_geometry(lines)

    vis.run()
    vis.destroy_window()


def boxes_to_lines(boxes):
    """
       4-------- 6
     /|         /|
    5 -------- 3 .
    | |        | |
    . 7 -------- 1
    |/         |/
    2 -------- 0
    """
    center = boxes[0:3]
    lwh = boxes[3:6]
    angles = np.array([0, 0, boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)
    return open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)


def parse_result_vis(box3d_lidar,
                     label_preds,
                     scores,
                     points,
                     scores_thres=0.25):
    num_bbox3d, bbox3d_dims = box3d_lidar.shape
    result_boxes = []
    for box_idx in range(num_bbox3d):
        # filter fake results: score = -1
        if scores[box_idx] < scores_thres:
            continue
        if bbox3d_dims == 9:
            print(
                "Score: {} Label: {} Box(x_c, y_c, z_c, w, l, h, vec_x, vec_y, -rot): {} {} {} {} {} {} {} {} {}"
                .format(scores[box_idx], label_preds[box_idx],
                        box3d_lidar[box_idx, 0], box3d_lidar[box_idx, 1],
                        box3d_lidar[box_idx, 2], box3d_lidar[box_idx, 3],
                        box3d_lidar[box_idx, 4], box3d_lidar[box_idx, 5],
                        box3d_lidar[box_idx, 6], box3d_lidar[box_idx, 7],
                        box3d_lidar[box_idx, 8]))
        elif bbox3d_dims == 7:
            print(
                "Score: {} Label: {} Box(x_c, y_c, z_c, w, l, h, -rot): {} {} {} {} {} {} {}"
                .format(scores[box_idx], label_preds[box_idx],
                        box3d_lidar[box_idx, 0], box3d_lidar[box_idx, 1],
                        box3d_lidar[box_idx, 2], box3d_lidar[box_idx, 3],
                        box3d_lidar[box_idx, 4], box3d_lidar[box_idx, 5],
                        box3d_lidar[box_idx, 6]))
        # draw result
        result_boxes.append([
            box3d_lidar[box_idx, 0], box3d_lidar[box_idx, 1],
            box3d_lidar[box_idx, 2], box3d_lidar[box_idx, 3],
            box3d_lidar[box_idx, 4], box3d_lidar[box_idx, 5],
            box3d_lidar[box_idx, -1]
        ])

    draw_result(points=points, result_boxes=np.asarray(result_boxes))


def main(args):
    predictor = init_predictor(args.model_file, args.params_file, args.gpu_id,
                               args.use_trt, args.trt_precision,
                               args.trt_use_static, args.trt_static_dir,
                               args.collect_shape_info, args.dynamic_shape_file)
    points = preprocess(args.lidar_file, args.num_point_dim, args.use_timelag)
    box3d_lidar, label_preds, scores = run(predictor, points)
    parse_result_vis(box3d_lidar, label_preds, scores, points)


if __name__ == '__main__':
    args = parse_args()

    main(args)
