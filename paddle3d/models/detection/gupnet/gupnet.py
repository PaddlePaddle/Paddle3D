import paddle
import paddle.nn as nn
import numpy as np
from typing import List
from paddle.vision.ops import roi_align
from paddle3d.datasets.kitti.kitti_gupnet import GUPKittiMonoDataset
from paddle3d.models.detection.gupnet.gupnet_dla import GUP_DLA34, GUP_DLAUp
from paddle3d.apis import manager
from paddle3d.models.detection.gupnet.gupnet_helper import _nms, _topk, extract_input_from_tensor
from paddle3d.models.detection.gupnet.gupnet_predictor import extract_dets_from_outputs, decode_detections
from paddle3d.models.base import BaseMonoModel
from paddle3d.geometries import BBoxes2D, BBoxes3D, CoordMode
from paddle3d.sample import Sample
from paddle3d.models.layers import param_init


def fill_fc_weights(layer):
    if isinstance(layer, nn.Conv2D):
        param_init.normal_init(layer.weight, std=0.001)
        if layer.bias is not None:
            param_init.constant_init(layer.bias, value=0.0)


def weights_init_xavier(layer):
    if isinstance(layer, nn.Linear):
        param_init.normal_init(layer.weight, std=0.001)
        param_init.constant_init(layer.bias, value=0.0)

    if isinstance(layer, nn.Conv2D):
        param_init.xavier_uniform_init(layer.weight)
        if layer.bias is not None:
            param_init.constant_init(layer.bias, value=0.0)

    elif isinstance(layer, nn.BatchNorm2D):
        param_init.constant_init(layer.weight, value=1.0)
        param_init.constant_init(layer.bias, value=0.0)


@manager.MODELS.add_component
class GUPNET(BaseMonoModel):
    """
    """
    def __init__(self,
                 max_detection: int = 50,
                 downsample=4,
                 threshold=0.2,
                 box_with_velocity: bool = False,
                 val_dataset_root=None,
                 pretrained_model=None):
        super(GUPNET, self).__init__(box_with_velocity=box_with_velocity,
                                     need_camera_to_image=True,
                                     need_lidar_to_camera=False,
                                     need_down_ratios=True)

        assert downsample in [4, 8, 16, 32]
        super(GUPNET, self).__init__()
        self.threshold = threshold
        self.max_detection = max_detection
        self.val_dataset = GUPKittiMonoDataset(dataset_root=val_dataset_root,
                                               mode='val')
        self.head_conv = 256  # default setting for head conv
        self.mean_size = np.array(
            [[1.76255119, 0.66068622, 0.84422524],
             [1.52563191462, 1.62856739989, 3.88311640418],
             [1.73698127, 0.59706367, 1.76282397]])
        self.mean_size = paddle.to_tensor(self.mean_size, dtype=paddle.float32)
        self.cls_num = self.mean_size.shape[0]
        self.backbone = GUP_DLA34(pretrained_model, return_levels=True)
        channels = self.backbone.channels
        self.first_level = int(np.log2(downsample))
        scales = [2**i for i in range(len(channels[self.first_level:]))]
        self.feat_up = GUP_DLAUp(in_channels_list=channels[self.first_level:],
                                 scales_list=scales)

        # initialize the head of pipeline, according to heads setting.
        self.heatmap = nn.Sequential(
            nn.Conv2D(channels[self.first_level],
                      self.head_conv,
                      kernel_size=3,
                      padding=1,
                      bias_attr=True), nn.ReLU(),
            nn.Conv2D(self.head_conv,
                      3,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias_attr=nn.initializer.Constant(value=-2.19)))
        self.offset_2d = nn.Sequential(
            nn.Conv2D(channels[self.first_level],
                      self.head_conv,
                      kernel_size=3,
                      padding=1,
                      bias_attr=True), nn.ReLU(),
            nn.Conv2D(self.head_conv,
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias_attr=True))
        self.size_2d = nn.Sequential(
            nn.Conv2D(channels[self.first_level],
                      self.head_conv,
                      kernel_size=3,
                      padding=1,
                      bias_attr=True), nn.ReLU(),
            nn.Conv2D(self.head_conv,
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias_attr=True))

        self.depth = nn.Sequential(
            nn.Conv2D(channels[self.first_level] + 2 + self.cls_num,
                      self.head_conv,
                      kernel_size=3,
                      padding=1,
                      bias_attr=True), nn.BatchNorm2D(self.head_conv),
            nn.ReLU(), nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(self.head_conv,
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias_attr=True))
        self.offset_3d = nn.Sequential(
            nn.Conv2D(channels[self.first_level] + 2 + self.cls_num,
                      self.head_conv,
                      kernel_size=3,
                      padding=1,
                      bias_attr=True), nn.BatchNorm2D(self.head_conv),
            nn.ReLU(), nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(self.head_conv,
                      2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias_attr=True))
        self.size_3d = nn.Sequential(
            nn.Conv2D(channels[self.first_level] + 2 + self.cls_num,
                      self.head_conv,
                      kernel_size=3,
                      padding=1,
                      bias_attr=True), nn.BatchNorm2D(self.head_conv),
            nn.ReLU(), nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(self.head_conv,
                      4,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias_attr=True))
        self.heading = nn.Sequential(
            nn.Conv2D(channels[self.first_level] + 2 + self.cls_num,
                      self.head_conv,
                      kernel_size=3,
                      padding=1,
                      bias_attr=True), nn.BatchNorm2D(self.head_conv),
            nn.ReLU(), nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(self.head_conv,
                      24,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias_attr=True))

        # init layers
        self.offset_2d.apply(fill_fc_weights)
        self.size_2d.apply(fill_fc_weights)

        self.depth.apply(weights_init_xavier)
        self.offset_3d.apply(weights_init_xavier)
        self.size_3d.apply(weights_init_xavier)
        self.heading.apply(weights_init_xavier)

    def export_forward(self, samples):
        pass

    def train_forward(self, samples):
        input, calibs, coord_ranges, targets, info, sample = samples
        feat = self.backbone(input)
        feat = self.feat_up(feat[self.first_level:])
        ret = {}
        ret['heatmap'] = self.heatmap(feat)
        ret['offset_2d'] = self.offset_2d(feat)
        ret['size_2d'] = self.size_2d(feat)
        mode = 'train'
        # two stage
        assert (mode in ['train', 'val', 'test'])
        inds, cls_ids = targets['indices'], targets['cls_ids']
        masks = targets['mask_2d'].astype(paddle.bool)
        ret.update(
            self.get_roi_feat(feat, inds, masks, ret, calibs, coord_ranges,
                              cls_ids, self.max_detection))
        return ret

    def test_forward(self, samples):
        input, calibs, coord_ranges, targets, info, sample = samples
        feat = self.backbone(input)
        feat = self.feat_up(feat[self.first_level:])
        ret = {}
        ret['heatmap'] = self.heatmap(feat)
        ret['offset_2d'] = self.offset_2d(feat)
        ret['size_2d'] = self.size_2d(feat)
        inds, cls_ids = _topk(_nms(
            paddle.clip(paddle.nn.functional.sigmoid(ret['heatmap']),
                        min=1e-4,
                        max=1 - 1e-4)),
                              K=self.max_detection)[1:3]
        masks = paddle.ones(inds.shape).astype(paddle.bool)
        ret.update(
            self.get_roi_feat(feat, inds, masks, ret, calibs, coord_ranges,
                              cls_ids, self.max_detection))

        # prediction result convert
        predictions = extract_dets_from_outputs(ret, K=50)
        predictions = predictions.detach().cpu().numpy()

        # get corresponding calibs & transform tensor to numpy
        calibs = [
            self.val_dataset.get_calib(int(index)) for index in info['img_id']
        ]
        info = {key: val.detach().cpu().numpy() for key, val in info.items()}
        cls_mean_size = self.val_dataset.cls_mean_size
        predictions = decode_detections(dets=predictions,
                                        info=info,
                                        calibs=calibs,
                                        cls_mean_size=cls_mean_size,
                                        threshold=self.threshold)
        res = []
        for id, img_id in enumerate(predictions.keys()):
            res.append(
                self._parse_results_to_sample(predictions[img_id], sample, id))

        return {'preds': res}

    def _parse_results_to_sample(self, results: paddle.Tensor, sample: dict,
                                 index: int):
        ret = Sample(sample['path'][index], sample['modality'][index])
        ret.meta.update(
            {key: value[index]
             for key, value in sample['meta'].items()})

        if 'calibs' in sample:
            ret.calibs = [
                sample['calibs'][i][index] for i in range(len(sample['calibs']))
            ]

        if len(results):
            results = paddle.to_tensor(results)
            # results = results[results[:, 14] == index][:, :14]
            results = results.numpy()
            clas = results[:, 0]
            bboxes_2d = BBoxes2D(results[:, 2:6])

            # TODO: fix hard code here
            bboxes_3d = BBoxes3D(results[:, [9, 10, 11, 8, 6, 7, 12]],
                                 coordmode=CoordMode.KittiCamera,
                                 origin=(0.5, 1, 0.5),
                                 rot_axis=1)

            confidences = results[:, 13]

            ret.confidences = confidences
            ret.bboxes_2d = bboxes_2d
            ret.bboxes_3d = bboxes_3d

            # 类别顺序不对应
            # - paddle：{0: 'Car', 1: 'Cyclist', 2: 'Pedestrian'}
            # - 源码：{'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
            for i in range(len(clas)):
                clas[i] = clas[i] - 1 if clas[i] > 0 else 2
            ret.labels = clas

        return ret

    def get_roi_feat_by_mask(self, feat, box2d_maps, inds, mask, calibs,
                             coord_ranges, cls_ids, K):
        BATCH_SIZE, _, HEIGHT, WIDE = feat.shape
        num_masked_bin = mask.sum()
        res = {}
        if num_masked_bin != 0:
            # get box2d of each roi region
            box2d_masked = extract_input_from_tensor(box2d_maps, inds, mask)

            # get roi feature
            boxes_num = paddle.to_tensor([0] * BATCH_SIZE).astype('int32')
            for x in box2d_masked[:, 0]:
                boxes_num[x.astype('int32')] += 1
            roi_feature_masked = roi_align(feat,
                                           box2d_masked[:, 1:],
                                           boxes_num=boxes_num,
                                           output_size=[7, 7],
                                           aligned=False)

            # get coord range of each roi
            coord_ranges_mask2d = coord_ranges[box2d_masked[:, 0].astype(
                paddle.int32)]

            # map box2d coordinate from feature map size domain to original image size domain
            box2d_masked = paddle.concat([
                box2d_masked[:, 0:1], box2d_masked[:, 1:2] / WIDE *
                (coord_ranges_mask2d[:, 1, 0:1] -
                 coord_ranges_mask2d[:, 0, 0:1]) +
                coord_ranges_mask2d[:, 0, 0:1], box2d_masked[:, 2:3] / HEIGHT *
                (coord_ranges_mask2d[:, 1, 1:2] -
                 coord_ranges_mask2d[:, 0, 1:2]) +
                coord_ranges_mask2d[:, 0, 1:2], box2d_masked[:, 3:4] / WIDE *
                (coord_ranges_mask2d[:, 1, 0:1] -
                 coord_ranges_mask2d[:, 0, 0:1]) +
                coord_ranges_mask2d[:, 0, 0:1], box2d_masked[:, 4:5] / HEIGHT *
                (coord_ranges_mask2d[:, 1, 1:2] -
                 coord_ranges_mask2d[:, 0, 1:2]) +
                coord_ranges_mask2d[:, 0, 1:2]
            ], 1)
            roi_calibs = calibs[box2d_masked[:, 0].astype(paddle.int32)]

            # project the coordinate in the normal image to the camera coord by calibs
            coords_in_camera_coord = paddle.concat([
                self.project2rect(
                    roi_calibs,
                    paddle.concat([
                        box2d_masked[:, 1:3],
                        paddle.ones([num_masked_bin, 1])
                    ], -1))[:, :2],
                self.project2rect(
                    roi_calibs,
                    paddle.concat([
                        box2d_masked[:, 3:5],
                        paddle.ones([num_masked_bin, 1])
                    ], -1))[:, :2]
            ], -1)

            coords_in_camera_coord = paddle.concat(
                [box2d_masked[:, 0:1], coords_in_camera_coord], -1)
            # generate coord maps
            coord_maps = paddle.concat([
                paddle.tile(paddle.concat([
                    coords_in_camera_coord[:, 1:2] + i *
                    (coords_in_camera_coord[:, 3:4] -
                     coords_in_camera_coord[:, 1:2]) / 6 for i in range(7)
                ], -1).unsqueeze(1),
                            repeat_times=([1, 7, 1])).unsqueeze(1),
                paddle.tile(paddle.concat([
                    coords_in_camera_coord[:, 2:3] + i *
                    (coords_in_camera_coord[:, 4:5] -
                     coords_in_camera_coord[:, 2:3]) / 6 for i in range(7)
                ], -1).unsqueeze(2),
                            repeat_times=([1, 1, 7])).unsqueeze(1)
            ], 1)

            # concatenate coord maps with feature maps in the channel dim
            cls_hots = paddle.zeros([num_masked_bin, self.cls_num])
            cls_hots[paddle.arange(num_masked_bin),
                     cls_ids[mask].astype(paddle.int32)] = 1.0

            roi_feature_masked = paddle.concat([
                roi_feature_masked, coord_maps,
                paddle.tile(cls_hots.unsqueeze(-1).unsqueeze(-1),
                            repeat_times=([1, 1, 7, 7]))
            ], 1)

            # compute heights of projected objects
            box2d_height = paddle.clip(box2d_masked[:, 4] - box2d_masked[:, 2],
                                       min=1.0)
            # compute real 3d height
            size3d_offset = self.size_3d(roi_feature_masked)[:, :, 0, 0]
            h3d_log_std = size3d_offset[:, 3:4]
            size3d_offset = size3d_offset[:, :3]
            size_3d = (self.mean_size[cls_ids[mask].astype(paddle.int32)] +
                       size3d_offset)
            depth_geo = size_3d[:, 0] / \
                box2d_height.squeeze() * roi_calibs[:, 0, 0]
            depth_net_out = self.depth(roi_feature_masked)[:, :, 0, 0]
            # σ_p^2
            depth_geo_log_std = (
                h3d_log_std.squeeze() + 2 *
                (roi_calibs[:, 0, 0].log() - box2d_height.log())).unsqueeze(-1)
            # log(σ_d^2) = log(σ_p^2 + σ_b^2)
            depth_net_log_std = paddle.logsumexp(paddle.concat(
                [depth_net_out[:, 1:2], depth_geo_log_std], -1),
                                                 -1,
                                                 keepdim=True)
            depth_net_out = paddle.concat(
                [(1. /
                  (paddle.nn.functional.sigmoid(depth_net_out[:, 0:1]) + 1e-6) -
                  1.) + depth_geo.unsqueeze(-1), depth_net_log_std], -1)

            res['train_tag'] = paddle.ones(num_masked_bin).astype(paddle.bool)
            res['heading'] = self.heading(roi_feature_masked)[:, :, 0, 0]
            res['depth'] = depth_net_out
            res['offset_3d'] = self.offset_3d(roi_feature_masked)[:, :, 0, 0]
            res['size_3d'] = size3d_offset
            res['h3d_log_variance'] = h3d_log_std
        else:
            res['depth'] = paddle.zeros([1, 2])
            res['offset_3d'] = paddle.zeros([1, 2])
            res['size_3d'] = paddle.zeros([1, 3])
            res['train_tag'] = paddle.zeros([1]).astype(paddle.bool)
            res['heading'] = paddle.zeros([1, 24])
            res['h3d_log_variance'] = paddle.zeros([1, 1])
        return res

    def get_roi_feat(self, feat, inds, mask, ret, calibs, coord_ranges, cls_ids,
                     K):
        BATCH_SIZE, _, HEIGHT, WIDE = feat.shape
        coord_map = paddle.tile(paddle.concat([
            paddle.tile(paddle.arange(WIDE).unsqueeze(0),
                        repeat_times=([HEIGHT, 1])).unsqueeze(0),
            paddle.tile(paddle.arange(HEIGHT).unsqueeze(-1),
                        repeat_times=([1, WIDE])).unsqueeze(0)
        ],
                                              axis=0).unsqueeze(0),
                                repeat_times=([BATCH_SIZE, 1, 1,
                                               1])).astype('float32')

        box2d_centre = coord_map + ret['offset_2d']
        box2d_maps = paddle.concat([
            box2d_centre - ret['size_2d'] / 2, box2d_centre + ret['size_2d'] / 2
        ], 1)
        box2d_maps = paddle.concat([
            paddle.tile(paddle.arange(BATCH_SIZE).unsqueeze(-1).unsqueeze(
                -1).unsqueeze(-1),
                        repeat_times=([1, 1, HEIGHT, WIDE])).astype('float32'),
            box2d_maps
        ], 1)
        # box2d_maps is box2d in each bin
        res = self.get_roi_feat_by_mask(feat, box2d_maps, inds, mask, calibs,
                                        coord_ranges, cls_ids, K)
        return res

    def project2rect(self, calib, point_img):
        c_u = calib[:, 0, 2]
        c_v = calib[:, 1, 2]
        f_u = calib[:, 0, 0]
        f_v = calib[:, 1, 1]
        b_x = calib[:, 0, 3] / (-f_u)  # relative
        b_y = calib[:, 1, 3] / (-f_v)
        x = (point_img[:, 0] - c_u) * point_img[:, 2] / f_u + b_x
        y = (point_img[:, 1] - c_v) * point_img[:, 2] / f_v + b_y
        z = point_img[:, 2]
        centre_by_obj = paddle.concat(
            [x.unsqueeze(-1), y.unsqueeze(-1),
             z.unsqueeze(-1)], -1)
        return centre_by_obj

    def logsumexp(self, x):
        x_max = x.data.max()
        return paddle.log(paddle.sum(paddle.exp(x - x_max), 1,
                                     keepdim=True)) + x_max

    def init_weight(self, layers):
        for m in layers.sublayers():
            if isinstance(m, nn.Conv2D):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @property
    def inputs(self) -> List[dict]:
        images = {
            'name': 'images',
            'dtype': 'float32',
            'shape': [1, 3, self.image_height, self.image_width]
        }
        res = [images]

        intrinsics = {
            'name': 'trans_cam_to_img',
            'dtype': 'float32',
            'shape': [1, 3, 3]
        }
        res.append(intrinsics)

        down_ratios = {
            'name': 'down_ratios',
            'dtype': 'float32',
            'shape': [1, 2]
        }
        res.append(down_ratios)
        return res

    @property
    def outputs(self) -> List[dict]:
        data = {
            'name': 'gupnet_output',
            'dtype': 'float32',
            'shape': [self.max_detection, 14]
        }
        return [data]
