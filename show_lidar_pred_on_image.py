import argparse
import os
import os.path as osp

import cv2
import numpy as np

from paddle3d.datasets.kitti.kitti_utils import box_lidar_to_camera
from paddle3d.geometries import BBoxes3D, CoordMode
from paddle3d.sample import Sample

classmap = {0: 'Car', 1: 'Cyclist', 2: 'Pedestrain'}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--calib_file', dest='calib_file', help='calibration file', type=str)
    parser.add_argument(
        '--image_file', dest='image_file', help='image file', type=str)
    parser.add_argument(
        '--label_file', dest='label_file', help='label file', type=str)
    parser.add_argument(
        '--pred_file',
        dest='pred_file',
        help='prediction results file',
        type=str)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='the path to save visualized result',
        type=str)
    parser.add_argument(
        '--draw_threshold',
        dest='draw_threshold',
        help=
        'prediction whose confidence is lower than threshold would not been shown',
        type=float)
    return parser.parse_args()


class Calib:
    def __init__(self, dict_calib):
        super(Calib, self).__init__()
        self.P0 = dict_calib['P0'].reshape(3, 4)
        self.P1 = dict_calib['P1'].reshape(3, 4)
        self.P2 = dict_calib['P2'].reshape(3, 4)
        self.P3 = dict_calib['P3'].reshape(3, 4)
        self.R0_rect = dict_calib['R0_rect'].reshape(3, 3)
        self.Tr_velo_to_cam = dict_calib['Tr_velo_to_cam'].reshape(3, 4)
        self.Tr_imu_to_velo = dict_calib['Tr_imu_to_velo'].reshape(3, 4)


class Object3d:
    def __init__(self, content):
        super(Object3d, self).__init__()
        lines = content.split()
        lines = list(filter(lambda x: len(x), lines))
        self.name, self.truncated, self.occluded, self.alpha = lines[0], float(
            lines[1]), float(lines[2]), float(lines[3])
        self.bbox = [lines[4], lines[5], lines[6], lines[7]]
        self.bbox = np.array([float(x) for x in self.bbox])
        self.dimensions = [lines[8], lines[9], lines[10]]
        self.dimensions = np.array([float(x) for x in self.dimensions])
        self.location = [lines[11], lines[12], lines[13]]
        self.location = np.array([float(x) for x in self.location])
        self.rotation_y = float(lines[14])
        if len(lines) == 16:
            self.score = float(lines[15])


def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return R


def parse_gt_info(calib_path, label_path):

    with open(calib_path) as f:
        lines = f.readlines()
    lines = list(filter(lambda x: len(x) and x != '\n', lines))
    dict_calib = {}
    for line in lines:
        key, value = line.split(":")
        dict_calib[key] = np.array([float(x) for x in value.split()])
    calib = Calib(dict_calib)

    with open(label_path, 'r') as f:
        lines = f.readlines()
        lines = list(filter(lambda x: len(x) and x != '\n', lines))
    obj = [Object3d(x) for x in lines]
    return calib, obj


def predictions_to_kitti_format(pred):
    num_boxes = pred.bboxes_3d.shape[0]
    names = np.array([classmap[label] for label in pred.labels])
    calibs = pred.calibs
    if pred.bboxes_3d.coordmode != CoordMode.KittiCamera:
        bboxes_3d = box_lidar_to_camera(pred.bboxes_3d, calibs)
    else:
        bboxes_3d = pred.bboxes_3d

    if bboxes_3d.origin != [.5, 1., .5]:
        bboxes_3d[:, :3] += bboxes_3d[:, 3:6] * (
            np.array([.5, 1., .5]) - np.array(bboxes_3d.origin))
        bboxes_3d.origin = [.5, 1., .5]

    loc = bboxes_3d[:, :3]
    dim = bboxes_3d[:, 3:6]

    contents = []
    for i in range(num_boxes):
        # In kitti records, dimensions order is hwl format
        content = "{} 0 0 0 0 0 0 0 {} {} {} {} {} {} {} {}".format(
            names[i], dim[i, 2], dim[i, 1], dim[i, 0], loc[i, 0], loc[i, 1],
            loc[i, 2], bboxes_3d[i, 6], pred.confidences[i])
        contents.append(content)

    obj = [Object3d(x) for x in contents]
    return obj


def parse_pred_info(pred_path, calib):
    with open(pred_path, 'r') as f:
        lines = f.readlines()
        lines = list(filter(lambda x: len(x) and x != '\n', lines))

    scores = []
    labels = []
    boxes_3d = []
    for res in lines:
        score = float(res.split("Score: ")[-1].split(" ")[0])
        label = int(res.split("Label: ")[-1].split(" ")[0])
        box_3d = res.split("Box(x_c, y_c, z_c, w, l, h, -rot): ")[-1].split(" ")
        box_3d = [float(b) for b in box_3d]
        scores.append(score)
        labels.append(label)
        boxes_3d.append(box_3d)
    scores = np.stack(scores)
    labels = np.stack(labels)
    boxes_3d = np.stack(boxes_3d)
    data = Sample(pred_path, 'lidar')
    data.bboxes_3d = BBoxes3D(boxes_3d)
    data.bboxes_3d.coordmode = 'Lidar'
    data.bboxes_3d.origin = [0.5, 0.5, 0.5]
    data.bboxes_3d.rot_axis = 2
    data.labels = labels
    data.confidences = scores
    data.calibs = calib

    return data


def visualize(image_path, calib, obj, title, draw_threshold=None):
    img = cv2.imread(image_path)
    for i in range(len(obj)):
        if obj[i].name in ['Car', 'Pedestrian', 'Cyclist']:
            if draw_threshold is not None and hasattr(obj[i], 'score'):
                if obj[i].score < draw_threshold:
                    continue
            R = rot_y(obj[i].rotation_y)
            h, w, l = obj[i].dimensions[0], obj[i].dimensions[1], obj[
                i].dimensions[2]
            x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
            y = [0, 0, 0, 0, -h, -h, -h, -h]
            z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
            corner_3d = np.vstack([x, y, z])
            corner_3d = np.dot(R, corner_3d)

            corner_3d[0, :] += obj[i].location[0]
            corner_3d[1, :] += obj[i].location[1]
            corner_3d[2, :] += obj[i].location[2]

            corner_3d = np.vstack((corner_3d, np.zeros((1,
                                                        corner_3d.shape[-1]))))
            corner_2d = np.dot(calib.P2, corner_3d)
            corner_2d[0, :] /= corner_2d[2, :]
            corner_2d[1, :] /= corner_2d[2, :]

            if obj[i].name == 'Car':
                color = [20, 20, 255]
            elif obj[i].name == 'Pedestrian':
                color = [0, 255, 255]
            else:
                color = [255, 0, 255]

            thickness = 1
            for corner_i in range(0, 4):
                ii, ij = corner_i, (corner_i + 1) % 4
                corner_2d = corner_2d.astype('int32')
                cv2.line(img, (corner_2d[0, ii], corner_2d[1, ii]),
                         (corner_2d[0, ij], corner_2d[1, ij]), color, thickness)
                ii, ij = corner_i + 4, (corner_i + 1) % 4 + 4
                cv2.line(img, (corner_2d[0, ii], corner_2d[1, ii]),
                         (corner_2d[0, ij], corner_2d[1, ij]), color, thickness)
                ii, ij = corner_i, corner_i + 4
                cv2.line(img, (corner_2d[0, ii], corner_2d[1, ii]),
                         (corner_2d[0, ij], corner_2d[1, ij]), color, thickness)
            box_text = obj[i].name
            if hasattr(obj[i], 'score'):
                box_text += ': {:.2}'.format(obj[i].score)
            cv2.putText(img, box_text,
                        (min(corner_2d[0, :]), min(corner_2d[1, :]) - 2),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
    cv2.putText(img, title, (int(img.shape[1] / 2), 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 100, 0), 2)

    return img


def main(args):
    calib, gt_obj = parse_gt_info(args.calib_file, args.label_file)
    gt_image = visualize(args.image_file, calib, gt_obj, title='GroundTruth')
    pred = parse_pred_info(args.pred_file, [
        calib.P0, calib.P1, calib.P2, calib.P3, calib.R0_rect,
        calib.Tr_velo_to_cam, calib.Tr_imu_to_velo
    ])
    preds = predictions_to_kitti_format(pred)
    pred_image = visualize(
        args.image_file,
        calib,
        preds,
        title='Prediction',
        draw_threshold=args.draw_threshold)
    show_image = np.vstack([gt_image, pred_image])
    cv2.imwrite(
        osp.join(args.save_dir,
                 osp.split(args.image_file)[-1]), show_image)


if __name__ == '__main__':
    args = parse_args()
    main(args)
