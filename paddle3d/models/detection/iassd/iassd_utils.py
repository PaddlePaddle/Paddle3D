import numpy as np
import paddle


def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    """
    Args:
        boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        extra_width: [extra_x, extra_y, extra_z]
    Returns:
        large_boxes3d: [x, y, z, dx, dy, dz, heading]
    """
    large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += paddle.to_tensor(
        extra_width, dtype=boxes3d.dtype)[None, :]

    return large_boxes3d


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    cosa = paddle.cos(angle)
    sina = paddle.sin(angle)
    zeros = paddle.zeros([points.shape[0]], dtype=angle.dtype)
    ones = paddle.ones([points.shape[0]], dtype=angle.dtype)
    rot_matrix = paddle.stack(
        [cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones],
        axis=1).reshape([-1, 3, 3]).astype('float32')
    points_rot = paddle.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = paddle.concat((points_rot, points[:, :, 3:]), axis=-1)
    return points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """

    template = paddle.to_tensor([
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1],
    ],
                                dtype=boxes3d.dtype) / 2

    corners3d = boxes3d[:, None, 3:6].expand([boxes3d.shape[0], 8, 3
                                              ]) * template[None, :, :]
    corners3d = rotate_points_along_z(
        corners3d.reshape([-1, 8, 3]), boxes3d[:, 6]).reshape([-1, 8, 3])
    corners3d += boxes3d[:, None, 0:3]

    return corners3d
