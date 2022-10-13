import numpy as np


def limit_period(val, offset: float = 0.5, period: float = np.pi):
    return val - np.floor(val / period + offset) * period


def boxes3d_kitti_lidar_to_lidar(boxes3d_lidar):
    """
    convert boxes from [x, y, z, w, l, h, yaw] to [x, y, z, l, w, h, heading], bottom_center -> obj_center
    """
    w, l, h, r = boxes3d_lidar[:, 3:
                               4], boxes3d_lidar[:, 4:
                                                 5], boxes3d_lidar[:, 5:
                                                                   6], boxes3d_lidar[:,
                                                                                     6:
                                                                                     7]
    boxes3d_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([boxes3d_lidar[:, 0:3], l, w, h, -(r + np.pi / 2)],
                          axis=-1)


def boxes3d_lidar_to_kitti_lidar(boxes3d_lidar):
    """
    convert boxes from [x, y, z, l, w, h, heading] to [x, y, z, w, l, h, yaw], obj_center -> bottom_center
    """
    l, w, h, heading = boxes3d_lidar[:, 3:
                                     4], boxes3d_lidar[:, 4:
                                                       5], boxes3d_lidar[:, 5:
                                                                         6], boxes3d_lidar[:,
                                                                                           6:
                                                                                           7]
    boxes3d_lidar[:, 2] -= h[:, 0] / 2
    return np.concatenate(
        [boxes3d_lidar[:, 0:3], w, l, h, -(heading + np.pi / 2)], axis=-1)
