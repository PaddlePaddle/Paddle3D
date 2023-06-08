import paddle
import paddle.nn as nn


def pairwise_dist(A, B):
    A = A.unsqueeze(-2)
    B = B.unsqueeze(-3)

    return paddle.abs(A - B).sum(-1)


class NDPushPullLoss(nn.Layer):
    """
    An embedding loss to min var of per cluster and max distance between different clusters.

    So, for easier cluster, margin_var should be small, and margin_dist should be larger

    Inputs:
    featmap: prediction of network, [b,N,h,w], float tensor
    gt: gt, [b,N,h,w], long tensor, all val >= ignore_label will NOT be contributed to loss.

    loss = var_weight * var_loss + dist_weight * dist_loss

    Args:
        var_weight (float):
        dist_weight (float):
        margin_var (float): margin for var, any var < this margin will NOT be counted in loss
        margin_dist (float): margin for distance, any distance > this margin will NOT be counted in loss
        ignore_label: val in gt >= this arg, will be ignored.
    """

    def __init__(self, var_weight, dist_weight, margin_var, margin_dist,
                 ignore_label):
        super(NDPushPullLoss, self).__init__()
        self.var_weight = var_weight
        self.dist_weight = dist_weight
        self.margin_var = margin_var
        self.margin_dist = margin_dist
        self.ignore_label = ignore_label

    def forward(self, featmap, gt):
        assert (featmap.shape[2:] == gt.shape[2:])
        pull_loss = []
        push_loss = []
        C = gt[gt < self.ignore_label].max().item()
        # [B, N, H, W] = fm, [B, 1, H, W]  = gt
        # TODO not an optimized implement here. Should not expand B dim.
        for b in range(featmap.shape[0]):
            bfeat = featmap[b]
            bgt = gt[b][0]
            instance_centers = {}
            for i in range(1, int(C) + 1):
                instance_mask = bgt == i
                if instance_mask.sum() == 0:
                    continue
                # pos_featmap = bfeat[:, instance_mask].T#.contiguous()  #  mask_num x N
                first_dim = bfeat.shape[0]
                # print('before', instance_mask.shape)
                instance_mask = instance_mask.expand_as(bfeat)
                # print('post', instance_mask.shape)
                pos_featmap = paddle.masked_select(
                    bfeat, instance_mask)  #.contiguous()  #  mask_num x N
                # print('pos_featmap', pos_featmap.shape)
                pos_featmap = pos_featmap.reshape([first_dim,
                                                   -1]).transpose([1, 0])
                instance_center = pos_featmap.mean(
                    axis=0, keepdim=True)  # N x mask_num (mean)-> N x 1
                instance_centers[i] = instance_center
                # TODO xxx
                instance_loss = paddle.clip(
                    pairwise_dist(pos_featmap, instance_center) -
                    self.margin_var,
                    min=0.0)
                pull_loss.append(instance_loss.mean())
            for i in range(1, int(C) + 1):
                for j in range(1, int(C) + 1):
                    if i == j:
                        continue  # No need to push
                    if i not in instance_centers or j not in instance_centers:
                        continue
                    instance_loss = paddle.clip(
                        2 * self.margin_dist - pairwise_dist(
                            instance_centers[i], instance_centers[j]),
                        min=0.0)
                    push_loss.append(instance_loss)
        if len(pull_loss) > 0:
            pull_loss = paddle.concat([item.unsqueeze(0) for item in pull_loss
                                       ]).mean() * self.var_weight
        else:
            pull_loss = 0.0 * featmap.mean()  # Fake loss

        if len(push_loss) > 0:
            push_loss = paddle.concat([item.unsqueeze(0) for item in push_loss
                                       ]).mean() * self.dist_weight
        else:
            push_loss = 0.0 * featmap.mean()  # Fake loss
        return push_loss + pull_loss


class IoULoss(nn.Layer):
    def __init__(self, ignore_index=255):
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        mask = (targets != self.ignore_index).astype('float32')
        targets = targets.astype('float32')
        num = paddle.sum(outputs * targets * mask)
        den = paddle.sum(outputs * mask + targets * mask -
                         outputs * targets * mask)
        return 1 - num / den
