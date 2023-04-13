import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from paddle3d.apis import manager
import cv2
from paddle3d.transforms.base import TransformABC
from paddle3d.geometries.bbox import BBoxes3D
from paddle3d.sample import Sample


@manager.TRANSFORMS.add_component
class LoadPointsFromFile(TransformABC):
    """Load Points From File.
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.
        """

        points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def __call__(self, results):
        """Call function to load points data from file.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        results['points'] = points

        return results


@manager.TRANSFORMS.add_component
class PointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = np.zeros((height, width), dtype='float32')
        coor = np.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (coor[:, 1] >= 0) & (
            coor[:, 1] < height) & (depth < self.grid_config['depth'][1]) & (
                depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = np.ones((coor.shape[0], ), dtype=np.bool_)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.astype(np.int64)

        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']

            cam2img = np.eye(4, dtype=np.float32)
            cam2img[:3, :3] = intrins[cid].astype(np.float32)

            lidar2cam = np.matmul(
                np.linalg.inv(np.matmul(camego2global, cam2camego)),
                np.matmul(lidarego2global, lidar2lidarego))
            lidar2img = np.matmul(cam2img, lidar2cam)
            points_img = np.matmul(points_lidar[:, :3],
                                   lidar2img[:3, :3].T) + np.expand_dims(
                                       lidar2img[:3, 3], axis=0)
            points_img = np.concatenate(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                axis=1)
            points_img = np.matmul(
                points_img, post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = np.stack(depth_map_list, axis=0)
        results['gt_depth'] = depth_map

        return results


def imnormalize(img, mean, std, to_rgb):
    img = img.copy().astype(np.float32)
    # cv2 inplace normalization does not accept uint8
    #assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def mmlab2pdNormalize(img):
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    to_rgb = True
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = img.astype(np.float32).transpose((2, 0, 1))
    return img


@manager.TRANSFORMS.add_component
class PrepareImageInputs(TransformABC):
    """Load multi channel images from a list of separate channel files.
    """

    def __init__(
            self,
            data_config,
            is_train=False,
            sequential=False,
            ego_cam='CAM_FRONT',
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlab2pdNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam

    def get_rot(self, h):
        return np.array([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= crop[:2]
        if flip:
            A = np.array([[-1, 0], [0, 1]], dtype=np.float32)
            b = np.array([crop[2] - crop[0], 0], dtype=np.float32)
            post_rot = np.matmul(A, post_rot)
            post_tran = np.matmul(A, post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)  # slight diff
        b = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = np.matmul(A, -b) + b
        post_rot = np.matmul(A, post_rot)
        post_tran = np.matmul(A, post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            # np.random.seed(0)
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            # np.random.seed(0)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            # np.random.seed(0)
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            # np.random.seed(0)
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            # np.random.seed(0)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            # np.random.seed(0)
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor2ego_transformation(self,
                                      cam_info,
                                      key_info,
                                      cam_name,
                                      ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sweepsensor2sweepego_rot = Quaternion(
            w, x, y, z).rotation_matrix.astype(np.float32)
        sweepsensor2sweepego_tran = cam_info['cams'][cam_name][
            'sensor2ego_translation']
        sweepsensor2sweepego = np.zeros((4, 4),
                                        dtype=sweepsensor2sweepego_rot.dtype)
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = Quaternion(w, x, y,
                                         z).rotation_matrix.astype(np.float32)
        sweepego2global_tran = cam_info['cams'][cam_name][
            'ego2global_translation']
        sweepego2global = np.zeros((4, 4), dtype=sweepego2global_rot.dtype)
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
        keyego2global_rot = Quaternion(w, x, y,
                                       z).rotation_matrix.astype(np.float32)
        keyego2global_tran = key_info['cams'][ego_cam]['ego2global_translation']
        keyego2global = np.zeros((4, 4), dtype=keyego2global_rot.dtype)
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = np.linalg.inv(keyego2global)

        sweepsensor2keyego = np.matmul(
            np.matmul(global2keyego, sweepego2global), sweepsensor2sweepego)

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][cam_name]['ego2global_rotation']
        keyego2global_rot = Quaternion(w, x, y,
                                       z).rotation_matrix.astype(np.float32)
        keyego2global_tran = key_info['cams'][cam_name][
            'ego2global_translation']
        keyego2global = np.zeros((4, 4), dtype=keyego2global_rot.dtype)
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = np.linalg.inv(keyego2global)
        # cur ego to sensor
        w, x, y, z = key_info['cams'][cam_name]['sensor2ego_rotation']
        keysensor2keyego_rot = Quaternion(w, x, y,
                                          z).rotation_matrix.astype(np.float32)
        keysensor2keyego_tran = key_info['cams'][cam_name][
            'sensor2ego_translation']
        keysensor2keyego = np.zeros((4, 4), dtype=keysensor2keyego_rot.dtype)
        keysensor2keyego[3, 3] = 1
        keysensor2keyego[:3, :3] = keysensor2keyego_rot
        keysensor2keyego[:3, -1] = keysensor2keyego_tran
        keyego2keysensor = np.linalg.inv(keysensor2keyego)
        keysensor2sweepsensor = np.linalg.inv(
            np.matmul(
                np.matmul(
                    np.matmul(keyego2keysensor, global2keyego),
                    sweepego2global), sweepsensor2sweepego))
        return sweepsensor2keyego, keysensor2sweepsensor

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        sensor2sensors = []

        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']

            img = Image.open(filename)
            post_rot = np.eye(2)
            post_tran = np.zeros((2, ))

            intrin = cam_data['cam_intrinsic']
            # intrinsics in a list same size as img_filename

            sensor2keyego, sensor2sensor = \
                self.get_sensor2ego_transformation(results['curr'],
                                                   results['curr'],
                                                   cam_name,
                                                   self.ego_cam)
            rot = sensor2keyego[:3, :3]
            tran = sensor2keyego[:3, 3]
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = np.zeros((3, ))
            post_rot = np.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            sensor2sensors.append(sensor2sensor)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                trans_adj = []
                rots_adj = []
                sensor2sensors_adj = []
                for cam_name in cam_names:
                    adjsensor2keyego, sensor2sensor = \
                        self.get_sensor2ego_transformation(adj_info,
                                                           results['curr'],
                                                           cam_name,
                                                           self.ego_cam)
                    rot = adjsensor2keyego[:3, :3]
                    tran = adjsensor2keyego[:3, 3]
                    rots_adj.append(rot)
                    trans_adj.append(tran)
                    sensor2sensors_adj.append(sensor2sensor)

                rots.extend(rots_adj)
                trans.extend(trans_adj)
                sensor2sensors.extend(sensor2sensors_adj)

        imgs = np.stack(imgs)

        rots = np.stack(rots)
        trans = np.stack(trans)
        intrins = np.stack(intrins)
        post_rots = np.stack(post_rots)
        post_trans = np.stack(post_trans)
        sensor2sensors = np.stack(sensor2sensors)

        results['canvas'] = canvas
        results['sensor2sensors'] = sensor2sensors
        return (imgs, rots, trans, intrins.astype(np.float32),
                post_rots.astype(np.float32), post_trans.astype(np.float32))

    def __call__(self, sample):
        sample['img_inputs'] = self.get_inputs(sample)
        return sample


@manager.TRANSFORMS.add_component
class LoadAnnotationsBEVDepth(TransformABC):
    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            # np.random.seed(0)
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            # np.random.seed(0)
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            # np.random.seed(0)
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            # np.random.seed(0)
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = np.array(rotate_angle / 180 * np.pi)
        rot_sin = np.sin(rotate_angle)
        rot_cos = np.cos(rotate_angle)
        rot_mat = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                            [0, 0, 1]])
        scale_mat = np.array([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                              [0, 0, scale_ratio]])
        flip_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = np.squeeze(
                (rot_mat @ np.expand_dims(gt_boxes[:, :3], -1)), -1)

            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:, 6] = 2 * np.arcsin(np.array([1.0])) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = np.squeeze(
                (rot_mat[:2, :2] @ np.expand_dims(gt_boxes[:, 7:], -1)), -1)
        return gt_boxes, rot_mat

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes, gt_labels = np.array(gt_boxes), np.array(gt_labels)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        bda_mat = np.zeros((4, 4))
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        bda_rot = bda_rot.astype(np.float32)
        bda_mat[:3, :3] = bda_rot
        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((0, 9))

        # change origin to (0,5, 0.5, 0) for bbox
        dst = np.array((0.5, 0.5, 0))
        src = np.array((0.5, 0.5, 0.5))
        gt_boxes[:, :3] += gt_boxes[:, 3:6] * (dst - src)

        results['gt_bboxes_3d'] = BBoxes3D(gt_boxes)
        results['gt_labels_3d'] = gt_labels
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]

        # use 'img' instead of img_inputs in sample
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot)
        return results


@manager.TRANSFORMS.add_component
class ConvertToSample(TransformABC):
    def __init__(self):
        super(ConvertToSample, self).__init__()

    def __call__(self, results):
        sample = Sample(path=None, modality='multiview')
        for key in results:
            if key in ['gt_bboxes_3d', 'gt_labels_3d']:
                sample[key] = [results[key]]

            sample[key] = results[key]
        sample.meta.id = sample['sample_idx']
        sample.data = results['img_inputs'][0]
        return sample
