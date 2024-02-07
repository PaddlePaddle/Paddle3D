# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# ------------------------------------------------------------------------
# Modified from BEV-LaneDet (https://github.com/gigo-team/bev_lane_det)
# ------------------------------------------------------------------------

import os
import json
import tempfile
import os.path as osp
import numpy as np
from tqdm import tqdm
from .min_cost_flow import SolveMinCostFlow
from paddle3d.datasets.metrics import MetricABC

from pprint import pprint
import paddle
from .cluster import embedding_post
from .post_process import bev_instance2points_with_offset_z
from scipy.interpolate import interp1d


def prune_3d_lane_by_range(lane_3d, x_min, x_max):
    # TODO: solve hard coded range later
    # remove points with y out of range
    # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
    # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200
                                     ), ...]

    # remove lane points out of x range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                     lane_3d[:, 0] < x_max), ...]
    return lane_3d


def resample_laneline_in_y(input_lane, y_steps, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert (input_lane.shape[0] >= 2)

    y_min = np.min(input_lane[:, 1]) - 5
    y_max = np.max(input_lane[:, 1]) + 5

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate(
            [input_lane,
             np.zeros([input_lane.shape[0], 1], dtype=np.float32)],
            axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)

    if out_vis:
        output_visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
        return x_values, z_values, output_visibility.astype(np.float32) + 1e-9
    return x_values, z_values


class LaneEval(object):
    def __init__(self):
        '''
            args.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
            args.anchor_y_steps = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100])
        '''

        self.x_min = -10
        self.x_max = 10
        self.y_min = 3
        self.y_max = 103
        self.y_samples = np.linspace(
            self.y_min, self.y_max, num=100, endpoint=False)
        self.dist_th = 1.5
        self.ratio_th = 0.75
        self.close_range = 40
        self.laneline_x_error_close = []
        self.laneline_x_error_far = []
        self.laneline_z_error_close = []
        self.laneline_z_error_far = []
        self.r_list = []
        self.p_list = []
        self.cnt_gt_list = []
        self.cnt_pred_list = []

    def bench(self, pred_lanes, gt_lanes):
        """
            Matching predicted lanes and ground-truth lanes in their IPM projection, ignoring z attributes.
            x error, y_error, and z error are all considered, although the matching does not rely on z
            The input of prediction and ground-truth lanes are in ground coordinate, x-right, y-forward, z-up
            The fundamental assumption is: 1. there are no two points from different lanes with identical x, y
                                              but different z's
                                           2. there are no two points from a single lane having identical x, y
                                              but different z's
            If the interest area is within the current drivable road, the above assumptions are almost always valid.

        :param pred_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param gt_lanes: N X 2 or N X 3 lists depending on 2D or 3D
        :param raw_file: file path rooted in dataset folder
        :param gt_cam_height: camera height given in ground-truth loader
        :param gt_cam_pitch: camera pitch given in ground-truth loader
        :return:
        """
        # change this properly
        close_range_idx = np.where(self.y_samples > self.close_range)[0][0]

        r_lane, p_lane, c_lane = 0., 0., 0.
        x_error_close = []
        x_error_far = []
        z_error_close = []
        z_error_far = []

        # only keep the visible portion
        gt_lanes = [np.array(gt_lane) for k, gt_lane in enumerate(gt_lanes)]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        # only consider those gt lanes overlapping with sampling range 有交集的部分
        gt_lanes = [
            lane for lane in gt_lanes if lane[0, 1] < self.y_samples[-1]
            and lane[-1, 1] > self.y_samples[0]
        ]

        gt_lanes = [
            prune_3d_lane_by_range(
                np.array(lane), 3 * self.x_min, 3 * self.x_max)
            for lane in gt_lanes
        ]

        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

        cnt_gt = len(gt_lanes)
        cnt_pred = len(pred_lanes)

        gt_visibility_mat = np.zeros((cnt_gt, 100))
        pred_visibility_mat = np.zeros((cnt_pred, 100))

        # resample gt and pred at y_samples
        for i in range(cnt_gt):
            min_y = np.min(np.array(gt_lanes[i])[:, 1])
            max_y = np.max(np.array(gt_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(
                np.array(gt_lanes[i]), self.y_samples, out_vis=True)
            gt_lanes[i] = np.vstack([x_values, z_values]).T
            gt_visibility_mat[i, :] = np.logical_and(
                x_values >= self.x_min,
                np.logical_and(
                    x_values <= self.x_max,
                    np.logical_and(self.y_samples >= min_y,
                                   self.y_samples <= max_y)))
            gt_visibility_mat[i, :] = np.logical_and(gt_visibility_mat[i, :],
                                                     visibility_vec)

        for i in range(cnt_pred):
            # # ATTENTION: ensure y mono increase before interpolation: but it can reduce size
            # pred_lanes[i] = make_lane_y_mono_inc(np.array(pred_lanes[i]))
            # pred_lane = prune_3d_lane_by_range(np.array(pred_lanes[i]), self.x_min, self.x_max)
            min_y = np.min(np.array(pred_lanes[i])[:, 1])
            max_y = np.max(np.array(pred_lanes[i])[:, 1])
            x_values, z_values, visibility_vec = resample_laneline_in_y(
                np.array(pred_lanes[i]), self.y_samples, out_vis=True)
            pred_lanes[i] = np.vstack([x_values, z_values]).T
            pred_visibility_mat[i, :] = np.logical_and(
                x_values >= self.x_min,
                np.logical_and(
                    x_values <= self.x_max,
                    np.logical_and(self.y_samples >= min_y,
                                   self.y_samples <= max_y)))
            pred_visibility_mat[i, :] = np.logical_and(
                pred_visibility_mat[i, :], visibility_vec)

        adj_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
        cost_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
        cost_mat.fill(1000)
        num_match_mat = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_close.fill(1000.)
        x_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=float)
        x_dist_mat_far.fill(1000.)
        z_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=float)
        z_dist_mat_close.fill(1000.)
        z_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=float)
        z_dist_mat_far.fill(1000.)

        # compute curve to curve distance
        for i in range(cnt_gt):
            for j in range(cnt_pred):  #
                x_dist = np.abs(gt_lanes[i][:, 0] - pred_lanes[j][:, 0])
                z_dist = np.abs(gt_lanes[i][:, 1] - pred_lanes[j][:, 1])
                euclidean_dist = np.sqrt(x_dist**2 + z_dist**2)  #

                # apply visibility to penalize different partial matching accordingly
                euclidean_dist[np.logical_or(
                    gt_visibility_mat[i, :] < 0.5,
                    pred_visibility_mat[j, :] < 0.5)] = self.dist_th

                # if np.average(euclidean_dist) < 2*self.dist_th: # don't prune here to encourage finding perfect match
                num_match_mat[i, j] = np.sum(euclidean_dist < self.dist_th)
                adj_mat[i, j] = 1
                # ATTENTION: use the sum as int type to meet the requirements of min cost flow optimization (int type)
                # using num_match_mat as cost does not work?
                cost_mat[i, j] = np.sum(euclidean_dist).astype(int)  #
                # cost_mat[i, j] = num_match_mat[i, j]

                # use the both visible portion to calculate distance error
                both_visible_indices = np.logical_and(
                    gt_visibility_mat[i, :] > 0.5,
                    pred_visibility_mat[j, :] > 0.5)
                if np.sum(both_visible_indices[:close_range_idx]) > 0:
                    x_dist_mat_close[i, j] = np.sum(
                        x_dist[:close_range_idx] *
                        both_visible_indices[:close_range_idx]) / np.sum(
                            both_visible_indices[:close_range_idx])
                    z_dist_mat_close[i, j] = np.sum(
                        z_dist[:close_range_idx] *
                        both_visible_indices[:close_range_idx]) / np.sum(
                            both_visible_indices[:close_range_idx])
                else:
                    x_dist_mat_close[i, j] = self.dist_th
                    z_dist_mat_close[i, j] = self.dist_th

                if np.sum(both_visible_indices[close_range_idx:]) > 0:
                    x_dist_mat_far[i, j] = np.sum(
                        x_dist[close_range_idx:] *
                        both_visible_indices[close_range_idx:]) / np.sum(
                            both_visible_indices[close_range_idx:])
                    z_dist_mat_far[i, j] = np.sum(
                        z_dist[close_range_idx:] *
                        both_visible_indices[close_range_idx:]) / np.sum(
                            both_visible_indices[close_range_idx:])
                else:
                    x_dist_mat_far[i, j] = self.dist_th
                    z_dist_mat_far[i, j] = self.dist_th

        # solve bipartite matching vis min cost flow solver
        match_results = SolveMinCostFlow(adj_mat, cost_mat)
        match_results = np.array(match_results)

        # only a match with avg cost < self.dist_th is consider valid one
        match_gt_ids = []
        match_pred_ids = []
        match_num = 0
        if match_results.shape[0] > 0:
            for i in range(len(match_results)):
                if match_results[i, 2] < self.dist_th * self.y_samples.shape[0]:
                    match_num += 1
                    gt_i = match_results[i, 0]
                    pred_i = match_results[i, 1]
                    # consider match when the matched points is above a ratio
                    if num_match_mat[gt_i, pred_i] / np.sum(
                            gt_visibility_mat[gt_i, :]) >= self.ratio_th:
                        r_lane += 1
                        match_gt_ids.append(gt_i)
                    if num_match_mat[gt_i, pred_i] / np.sum(
                            pred_visibility_mat[pred_i, :]) >= self.ratio_th:
                        p_lane += 1
                        match_pred_ids.append(pred_i)

                    x_error_close.append(x_dist_mat_close[gt_i, pred_i])
                    x_error_far.append(x_dist_mat_far[gt_i, pred_i])
                    z_error_close.append(z_dist_mat_close[gt_i, pred_i])
                    z_error_far.append(z_dist_mat_far[gt_i, pred_i])
        return r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, x_error_close, x_error_far, z_error_close, z_error_far

    def bench_all(self, pred_lanes, gt_lanes):
        r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, \
        x_error_close, x_error_far, \
        z_error_close, z_error_far = self.bench(pred_lanes,
                                                gt_lanes)
        # laneline_stats.append(np.array([r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num]))
        # consider x_error z_error only for the matched lanes
        # if r_lane > 0 and p_lane > 0:
        self.r_list.append(r_lane)
        self.p_list.append(p_lane)
        self.cnt_gt_list.append(cnt_gt)
        self.cnt_pred_list.append(cnt_pred)
        self.laneline_x_error_close.extend(x_error_close)
        self.laneline_x_error_far.extend(x_error_far)
        self.laneline_z_error_close.extend(z_error_close)
        self.laneline_z_error_far.extend(z_error_far)

    def show(self):
        r_lane = np.sum(self.r_list)
        p_lane = np.sum(self.p_list)
        cnt_gt = np.sum(self.cnt_gt_list)
        cnt_pred = np.sum(self.cnt_pred_list)
        Recall = r_lane / (cnt_gt + 1e-6)
        Precision = p_lane / (cnt_pred + 1e-6)
        f1_score = 2 * Recall * Precision / (Recall + Precision + 1e-6)
        dict_res = {
            'x_error_close': np.average(self.laneline_x_error_close),
            'x_error_far': np.average(self.laneline_x_error_far),
            'z_error_close': np.average(self.laneline_z_error_close),
            'z_error_far': np.average(self.laneline_z_error_far),
            'recall': Recall,
            'precision': Precision,
            'f1_score': f1_score
        }
        pprint(dict_res)
        return dict_res


class PostProcessDataset(paddle.io.Dataset):
    def __init__(self, model_res_save_path, postprocess_save_path,
                 test_json_paths, x_range, meter_per_pixel):
        self.valid_data = os.listdir(model_res_save_path)
        self.postprocess_save_path = postprocess_save_path
        os.makedirs(self.postprocess_save_path, exist_ok=True)

        self.model_res_save_path = model_res_save_path
        self.x_range = x_range
        self.meter_per_pixel = meter_per_pixel
        self.post_conf = 0.9  # Minimum confidence on the segmentation map for clustering
        self.post_emb_margin = 6.0  # embeding margin of different clusters
        self.post_min_cluster_size = 15  # The minimum number of points in a cluster
        d_gt_res = {}
        with open(test_json_paths, 'r') as f:
            for i in f.readlines():
                line_content = json.loads(i.strip())

                lanes = []
                for lane_idx in range(len(line_content['laneLines'])):
                    lane_selected = np.array(
                        line_content['laneLines'][lane_idx])[
                            np.array(line_content['laneLines_visibility']
                                     [lane_idx]) > 0.5]
                    lanes.append(lane_selected.tolist())
                d_gt_res[line_content['raw_file']] = lanes
        self.d_gt_res = d_gt_res

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, item):
        loaded = np.load(
            os.path.join(self.model_res_save_path, self.valid_data[item]))
        prediction = (loaded[:, 0:1, :, :], loaded[:, 1:3, :, :])
        offset_y = loaded[:, 3:4, :, :][0][0]
        z_pred = loaded[:, 4:5, :, :][0][0]
        files = self.valid_data[item].split('.')[0].split('__')
        canvas, ids = embedding_post(
            prediction,
            self.post_conf,
            emb_margin=self.post_emb_margin,
            min_cluster_size=self.post_min_cluster_size,
            canvas_color=False)
        lines = bev_instance2points_with_offset_z(
            canvas,
            max_x=self.x_range[1],
            meter_per_pixal=(self.meter_per_pixel, self.meter_per_pixel),
            offset_y=offset_y,
            Z=z_pred)
        frame_lanes_pred = []
        for lane in lines:
            pred_in_persformer = np.array([-1 * lane[1], lane[0], lane[2]])
            y = np.linspace(
                min(pred_in_persformer[1]), max(pred_in_persformer[1]), 40)
            f_x = np.polyfit(pred_in_persformer[1], pred_in_persformer[0], 3)
            f_z = np.polyfit(pred_in_persformer[1], pred_in_persformer[2], 3)
            pred_in_persformer = np.array(
                [np.poly1d(f_x)(y), y, np.poly1d(f_z)(y)])
            frame_lanes_pred.append(pred_in_persformer.T.tolist())
        gt_key = 'images' + '/' + files[0] + '/' + files[1] + '.jpg'
        frame_lanes_gt = self.d_gt_res[gt_key]
        with open(
                os.path.join(self.postprocess_save_path,
                             files[0] + '_' + files[1] + '.json'), 'w') as f1:
            json.dump([frame_lanes_pred, frame_lanes_gt], f1)
        return np.zeros((3, 3))


class ApolloLaneMetric(MetricABC):
    """
    """

    def __init__(self, test_json_paths, x_range, meter_per_pixel):
        self.test_json_paths = test_json_paths
        self.x_range = x_range
        self.meter_per_pixel = meter_per_pixel
        tmp_dir = tempfile.TemporaryDirectory()

        self.res_save_path = osp.join(tmp_dir.name, 'result')

        os.makedirs(self.res_save_path, exist_ok=True)
        self.predictions = []

    def update(self, predictions, **kwargs):
        """
        """
        self.predictions += predictions

    def compute(self, **kwargs) -> dict:
        """
        """
        np_save_path = self.predictions[0]
        print('np save path', np_save_path)
        res_save_path = self.res_save_path
        test_json_paths = self.test_json_paths

        postprocess = PostProcessDataset(np_save_path, res_save_path,
                                         test_json_paths, self.x_range,
                                         self.meter_per_pixel)
        postprocess_loader = paddle.io.DataLoader(
            dataset=postprocess, batch_size=32, num_workers=4, shuffle=False)
        for item in tqdm(postprocess_loader):
            continue

        lane_eval = LaneEval()
        res_list = os.listdir(res_save_path)
        for item in tqdm(res_list):
            with open(os.path.join(res_save_path, item), 'r') as f:
                res = json.load(f)
            lane_eval.bench_all(res[0], res[1])
        lane_eval.show()
        return None
