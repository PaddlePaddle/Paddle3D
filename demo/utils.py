import cv2
import numpy as np

from paddle3d.transforms.target_generator import encode_label


def get_ratio(ori_img_size, output_size, down_ratio=(4, 4)):
    return np.array([[
        down_ratio[1] * ori_img_size[1] / output_size[1],
        down_ratio[0] * ori_img_size[0] / output_size[0]
    ]], np.float32)


def get_img(img_path):
    img = cv2.imread(img_path)
    origin_shape = img.shape
    img = cv2.resize(img, (1280, 384))

    target_shape = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0
    img = np.subtract(img, np.array([0.485, 0.456, 0.406]))
    img = np.true_divide(img, np.array([0.229, 0.224, 0.225]))
    img = np.array(img, np.float32)

    img = img.transpose(2, 0, 1)
    img = img[None, :, :, :]

    return img, origin_shape, target_shape


def total_pred_by_conf_to_kitti_records(total_pred,
                                        conf,
                                        class_names=[
                                            "Car", "Cyclist", "Pedestrian"
                                        ]):
    """convert total_pred to kitti_records"""
    kitti_records_list = []
    for p in total_pred:
        if p[-1] > conf:
            p = list(p)
            p[0] = class_names[int(p[0])]
            # default, to kitti_records formate
            p.insert(1, 0.0)
            p.insert(2, 0)
            kitti_records_list.append(p)
    kitti_records = np.array(kitti_records_list)

    return kitti_records


def make_imgpts_list(bboxes_3d, K):
    """to 8 points on image"""
    # external parameters do not transform
    rvec = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tvec = np.array([[0.0], [0.0], [0.0]])

    imgpts_list = []
    for box3d in bboxes_3d:

        locs = np.array(box3d[0:3])
        rot_y = np.array(box3d[6])

        height, width, length = box3d[3:6]
        _, box2d, box3d = encode_label(K, rot_y,
                                       np.array([length, height, width]), locs)

        if np.all(box2d == 0):
            continue

        imgpts, _ = cv2.projectPoints(box3d.T, rvec, tvec, K, 0)
        imgpts_list.append(imgpts)

    return imgpts_list


def draw_mono_3d(img, imgpts_list):
    """draw smoke result to photo"""
    connect_line_id = [
        [1, 0],
        [2, 7],
        [3, 6],
        [4, 5],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
        [0, 7],
        [7, 6],
        [6, 5],
        [5, 0],
    ]

    img_draw = img.copy()

    for imgpts in imgpts_list:
        for p in imgpts:
            p_x, p_y = int(p[0][0]), int(p[0][1])
            cv2.circle(img_draw, (p_x, p_y), 1, (0, 255, 0), -1)
        for i, line_id in enumerate(connect_line_id):

            p1 = (int(imgpts[line_id[0]][0][0]), int(imgpts[line_id[0]][0][1]))
            p2 = (int(imgpts[line_id[1]][0][0]), int(imgpts[line_id[1]][0][1]))

            if i <= 3:  # body
                color = (255, 0, 0)
            elif i <= 7:  # head
                color = (0, 0, 255)
            else:  # tail
                color = (255, 255, 0)

            cv2.line(img_draw, p1, p2, color, 1)

    return img_draw
