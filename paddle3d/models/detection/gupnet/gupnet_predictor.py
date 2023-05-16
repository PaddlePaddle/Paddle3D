import numpy as np
import paddle
from paddle3d.models.detection.gupnet.gupnet_helper import _nms, _topk, _transpose_and_gather_feat

num_heading_bin = 12


def class2angle(cls, residual, to_label_format=False):
    ''' Inverse function to angle2class. '''
    angle_per_class = 2 * np.pi / float(num_heading_bin)
    angle_center = cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)


def decode_detections(dets, info, calibs, cls_mean_size, threshold):
    '''NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    '''
    results = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold:
                continue

            # 2d bboxs decoding
            x = dets[i, j, 2] * info['bbox_downsample_ratio'][i][0]
            y = dets[i, j, 3] * info['bbox_downsample_ratio'][i][1]
            w = dets[i, j, 4] * info['bbox_downsample_ratio'][i][0]
            h = dets[i, j, 5] * info['bbox_downsample_ratio'][i][1]
            bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

            # 3d bboxs decoding
            # depth decoding
            depth = dets[i, j, 6]

            # heading angle decoding
            alpha = get_heading_angle(dets[i, j, 7:31])
            ry = calibs[i].alpha2ry(alpha, x)

            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]
            if True in (dimensions < 0.0):
                continue

            # positions decoding
            x3d = dets[i, j, 34] * info['bbox_downsample_ratio'][i][0]
            y3d = dets[i, j, 35] * info['bbox_downsample_ratio'][i][1]
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            preds.append([cls_id, alpha] + bbox + dimensions.tolist() +
                         locations.tolist() + [ry, score])
        results[info['img_id'][i]] = preds
    return results


# two stage style
def extract_dets_from_outputs(outputs, K=50):
    # get src outputs
    heatmap = outputs['heatmap']
    size_2d = outputs['size_2d']
    offset_2d = outputs['offset_2d']

    batch, channel, height, width = heatmap.shape  # get shape

    heading = outputs['heading'].reshape((batch, K, -1))
    depth = outputs['depth'].reshape((batch, K, -1))[:, :, 0:1]
    size_3d = outputs['size_3d'].reshape((batch, K, -1))
    offset_3d = outputs['offset_3d'].reshape((batch, K, -1))

    heatmap = paddle.clip(paddle.nn.functional.sigmoid(heatmap),
                          min=1e-4,
                          max=1 - 1e-4)

    # perform nms on heatmaps
    heatmap = _nms(heatmap)
    scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)

    offset_2d = _transpose_and_gather_feat(offset_2d, inds)
    offset_2d = offset_2d.reshape((batch, K, 2))
    xs2d = xs.reshape((batch, K, 1)) + offset_2d[:, :, 0:1]
    ys2d = ys.reshape((batch, K, 1)) + offset_2d[:, :, 1:2]

    xs3d = xs.reshape((batch, K, 1)) + offset_3d[:, :, 0:1]
    ys3d = ys.reshape((batch, K, 1)) + offset_3d[:, :, 1:2]

    cls_ids = cls_ids.reshape((batch, K, 1)).astype('float32')
    depth_score = (-(0.5 * outputs['depth'].reshape(
        (batch, K, -1))[:, :, 1:2]).exp()).exp()
    scores = scores.reshape((batch, K, 1)) * depth_score

    # check shape
    xs2d = xs2d.reshape((batch, K, 1))
    ys2d = ys2d.reshape((batch, K, 1))
    xs3d = xs3d.reshape((batch, K, 1))
    ys3d = ys3d.reshape((batch, K, 1))

    size_2d = _transpose_and_gather_feat(size_2d, inds)
    size_2d = size_2d.reshape((batch, K, 2))

    detections = paddle.concat([
        cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d,
        ys3d
    ],
                               axis=2)

    return detections
