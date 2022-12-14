# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import cv2
import numpy as np
from PIL import Image
from nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils import splits as nuscenes_split

import paddle
from paddle import inference
from paddle3d.geometries import BBoxes3D

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        help="Model filename, Specify this when your model is a combined model.",
        required=True)
    parser.add_argument(
        "--params_file",
        type=str,
        help=
        "Parameter filename, Specify this when your model is a combined model.",
        required=True)
    parser.add_argument(
        '--data_root',
        type=str,
        # nargs='+',
        help='The dataroot of nuscenes dataset.',
        required=True)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU card id.")
    parser.add_argument(
        "--use_trt",
        action='store_true',
        help="Whether to use tensorrt to accelerate when using gpu.")
    parser.add_argument(
        "--trt_precision",
        type=int,
        default=0,
        help="Precision type of tensorrt, 0: kFloat32, 1: kHalf.")
    parser.add_argument(
        "--trt_use_static",
        action='store_true',
        help="Whether to load the tensorrt graph optimization from a disk path."
    )
    parser.add_argument(
        "--trt_static_dir",
        type=str,
        help="Path of a tensorrt graph optimization directory.")
    parser.add_argument(
        "--collect_shape_info",
        action='store_true',
        help="Whether to collect dynamic shape before using tensorrt.")
    parser.add_argument(
        "--dynamic_shape_file",
        type=str,
        default="petr_shape_info.txt",
        help="Path of a dynamic shape file for tensorrt.")
    parser.add_argument(
        "--with_timestamp",
        action='store_true',
        help="Whether to timestamp(for petrv2).")
    return parser.parse_args()


def load_predictor(model_file,
                   params_file,
                   gpu_id=0,
                   use_trt=False,
                   trt_precision=0,
                   trt_use_static=False,
                   trt_static_dir=None,
                   collect_shape_info=False,
                   dynamic_shape_file=None):
    """load_predictor
    initialize the inference engine
    """
    config = inference.Config(model_file, params_file)
    config.enable_use_gpu(1000, gpu_id)

    # enable memory optim
    config.enable_memory_optim()
    # config.disable_glog_info()

    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    if use_trt:
        precision_mode = paddle.inference.PrecisionType.Float32
        if trt_precision == 1:
            precision_mode = paddle.inference.PrecisionType.Half
        config.enable_tensorrt_engine(
            workspace_size=1 << 20,
            max_batch_size=1,
            min_subgraph_size=30,
            precision_mode=precision_mode,
            use_static=trt_use_static,
            use_calib_mode=False)

        if collect_shape_info:
            config.collect_shape_range_info(dynamic_shape_file)
        else:
            config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file, True)
        if trt_use_static:
            config.set_optim_cache_dir(trt_static_dir)
    predictor = inference.create_predictor(config)

    return predictor


def imnormalize(img, mean, std, to_rgb=True):
    """normalize an image with mean and std.
    """
    # cv2 inplace normalization does not accept uint8
    img = img.copy().astype(np.float32)

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def get_resize_crop_shape(img_shape, target_shape):
    H, W = img_shape
    fH, fW = target_shape

    resize = max(fH / H, fW / W)
    resize_shape = (int(W * resize), int(H * resize))
    newW, newH = resize_shape
    crop_h = int(newH) - fH
    crop_w = int(max(0, newW - fW) / 2)

    crop_shape = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    
    return resize, resize_shape, crop_shape

def _get_rot(h):

    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def _img_transform(img, resize, resize_dims, crop, rotate=0):
    ida_rot = np.eye(2)
    ida_tran = np.zeros(2)
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)

    img = img.rotate(rotate)

    # post-homography transformation
    ida_rot *= resize
    ida_tran -= np.array(crop[:2])

    A = _get_rot(rotate / 180 * np.pi)
    b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = np.matmul(A, -b) + b
    ida_rot = np.matmul(A, ida_rot)
    ida_tran = np.matmul(A, ida_tran) + b
    ida_mat = np.eye(3)
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 2] = ida_tran
    return img, ida_mat

def get_image(filenames, intrinsics, extrinsics):
    """
    Loads image for a sample
    Args:
        idx [int]: Index of the image sample
    Returns:
        image [np.ndarray(H, W, 3)]: RGB Image
    """
    img = np.stack([cv2.imread(name, cv2.IMREAD_UNCHANGED) for name in filenames], axis=-1)
    imgs = [img[..., i] for i in range(img.shape[-1])]

    new_imgs = []

    target_shape = (320, 800)

    for i in range(len(imgs)):
        img_shape = imgs[i].shape[:2]
        resize, resize_shape, crop_shape = get_resize_crop_shape(
            img_shape, target_shape)
        img = Image.fromarray(np.uint8(imgs[i]))

        img, ida_mat = _img_transform(img, resize, resize_shape, crop_shape)
        new_imgs.append(np.array(img).astype(np.float32))
        intrinsics[
                i][:3, :3] = ida_mat @ intrinsics[i][:3, :3]
    lidar2imgs = [
            intrinsics[i] @ extrinsics[i].T
            for i in range(len(extrinsics))
        ]
    mean = np.array([103.530, 116.280, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.120, 58.395], dtype=np.float32)

    new_imgs = [imnormalize(img, mean, std, False) for img in new_imgs]

    return np.array(new_imgs).transpose([0, 3, 1, 2])[np.newaxis, ...], lidar2imgs

def parse_results_for_nus(results):
    num_samples = 1
    new_results = []

    data = dict()
    bboxes_3d = results["pred_boxes"].numpy()
    labels = results["pred_labels"].numpy()
    confidences = results["pred_scores"].numpy()
    bottom_center = bboxes_3d[:, :3]
    gravity_center = np.zeros_like(bottom_center)
    gravity_center[:, :2] = bottom_center[:, :2]
    gravity_center[:, 2] = bottom_center[:, 2] + bboxes_3d[:, 5] * 0.5
    bboxes_3d[:, :3] = gravity_center
    data['bboxes_3d'] = BBoxes3D(bboxes_3d[:, 0:7])
    data['bboxes_3d'].coordmode = 'Lidar'
    data['bboxes_3d'].origin = [0.5, 0.5, 0.5]
    data['bboxes_3d'].rot_axis = 2
    data['bboxes_3d'].velocities = bboxes_3d[:, 7:9]

    data['labels'] = labels
    data['confidences'] = confidences
    # data.meta = SampleMeta(id=sample["meta"][i]['id'])

    new_results.append(data)
    return new_results

def run(predictor, img, with_timestamp, lidar2imgs):
    input_names = predictor.get_input_names()

    input_tensor0 = predictor.get_input_handle(input_names[0])
    input_tensor1 = predictor.get_input_handle(input_names[1])

    num_cams = 6
    if with_timestamp:
        input_tensor2 = predictor.get_input_handle(input_names[2])
        num_cams = 12
    
    img2lidars = [np.linalg.inv(lidar2img) for lidar2img in lidar2imgs]

    if with_timestamp:
        img2lidars += img2lidars

    img2lidars = np.array(img2lidars).reshape([num_cams, 4,
                                               4]).astype('float32')

    input_tensor0.reshape([1, num_cams, 3, 320, 800])
    input_tensor0.copy_from_cpu(img)

    input_tensor1.reshape([num_cams, 4, 4])
    input_tensor1.copy_from_cpu(img2lidars)

    if with_timestamp:
        timestamp = np.zeros([num_cams]).astype('float32')
        timestamp[num_cams // 2:] = 1.0
        input_tensor2.reshape([1, num_cams])
        input_tensor2.copy_from_cpu(timestamp)

    predictor.run()
    outs = []
    output_names = predictor.get_output_names()
    for name in output_names:
        out = predictor.get_output_handle(name)
        out = out.copy_to_cpu()
        out = paddle.to_tensor(out)
        outs.append(out)

    result = {}
    result['pred_boxes'] = outs[0]
    result['pred_scores'] = outs[1]
    result['pred_labels'] = outs[2]

    return result


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T +
         e2g_t_s) @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def get_input_data_from_nus(data_root):
    version = 'v1.0-trainval'
    # dataset_root = '/ssd3/datasets/nuscene/'
    max_sweeps = 10
    nusc = NuScenes(version=version, dataroot=data_root, verbose=False)
    
    sample_idx = 80

    sample = nusc.sample[sample_idx]
    lidar_token = sample['data']['LIDAR_TOP']
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor',
                            sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

    assert os.path.exists(lidar_path)

    info = {
        'lidar_token': lidar_token,
        'lidar_path': lidar_path,
        'token': sample['token'],
        'sweeps': [],
        'cams': dict(),
        'lidar2ego_translation': cs_record['translation'],
        'lidar2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sample['timestamp'],
    }

    l2e_r = info['lidar2ego_rotation']
    l2e_t = info['lidar2ego_translation']
    e2g_r = info['ego2global_rotation']
    e2g_t = info['ego2global_translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # obtain 6 image's information per frame
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    for cam in camera_types:
        cam_token = sample['data'][cam]
        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
        cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                        e2g_t, e2g_r_mat, cam)
        cam_info.update(cam_intrinsic=cam_intrinsic)
        info['cams'].update({cam: cam_info})

    # obtain sweeps for a single key-frame
    sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    sweeps = []
    while len(sweeps) < max_sweeps:
        if not sd_rec['prev'] == '':
            sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                        l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
            sweeps.append(sweep)
            sd_rec = nusc.get('sample_data', sd_rec['prev'])
        else:
            break
    info['sweeps'] = sweeps

    image_paths = []
    lidar2img_rts = []
    intrinsics = []
    extrinsics = []
    img_timestamp = []
    for cam_type, cam_info in info['cams'].items():
        img_timestamp.append(cam_info['timestamp'] / 1e6)
        image_paths.append(cam_info['data_path'])
        # obtain lidar to image transformation matrix
        lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
        lidar2cam_t = cam_info[
            'sensor2lidar_translation'] @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        intrinsic = cam_info['cam_intrinsic']
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
        lidar2img_rt = (viewpad @ lidar2cam_rt.T)
        intrinsics.append(viewpad)

        extrinsics.append(lidar2cam_rt)
        lidar2img_rts.append(lidar2img_rt)

    return image_paths, intrinsics, extrinsics

def main(args):
    predictor = load_predictor(args.model_file, args.params_file, args.gpu_id,
                               args.use_trt, args.trt_precision,
                               args.trt_use_static, args.trt_static_dir,
                               args.collect_shape_info, args.dynamic_shape_file)
    img_paths, intrinsics, extrinsics = get_input_data_from_nus(args.data_root)

    image, lidar2imgs = get_image(img_paths, intrinsics, extrinsics)

    result = run(predictor, image, args.with_timestamp, lidar2imgs)

    for k, v in result.items():
        print(k, v.shape, v.dtype)

    nuscenes_result = parse_results_for_nus(result)


if __name__ == '__main__':
    args = parse_args()
    main(args)
