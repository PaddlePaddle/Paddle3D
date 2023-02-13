import paddle
import paddle.nn.functional as F
from paddle import nn

from paddle3d.apis import manager


@manager.LOSSES.add_component
class SigmoidCELoss(nn.Layer):
    def __init__(self,  loss_weight=1.0, reduction='mean'):
        super(SigmoidCELoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """Forward function to calculate accuracy.
        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.
        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        pos_weight = paddle.to_tensor((targets == 0), dtype='float32').sum(axis=1) / \
                    paddle.to_tensor((targets == 1), dtype='float32').sum(axis=1).clip(min=1.0)
        pos_weight = pos_weight.unsqueeze(1)
        weight_loss = targets * pos_weight + (1 - targets)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction, weight=weight_loss)
        return self.loss_weight * loss


@manager.LOSSES.add_component
class FocalDiceLoss(nn.Layer):
    def __init__(self, alpha=1, gamma=2, loss_weight=1.0, reduction='mean'):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.reduction = reduction
        
    def focal_loss(self, inputs, labels):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, labels, reduction='none')
        pt = paddle.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return paddle.mean(F_loss)
        elif self.reduction == 'sum':
            return paddle.sum(F_loss)

    def dice_loss(self, inputs, labels, smooth=1):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten()
        labels = labels.flatten()
        intersection = (inputs * labels).sum() 
        dice = (2. * intersection + smooth) / (inputs.sum() + labels.sum() + smooth)
        return 1 - dice
    
    def forward(self, inputs, labels, smooth=1):
        focal = self.focal_loss(inputs, labels)
        dice = self.dice_loss(inputs, labels, smooth=smooth)
        return focal + dice


if __name__ == '__main__':
    import numpy as np
    logit = paddle.randn([256, 768])
    arr = np.random.randn(256 * 768).reshape([256, 768])
    arr = np.clip(arr, 0, 1)
    arr = arr.astype(np.int8).astype(np.float32)
    label = paddle.to_tensor(arr)
    #loss = SigmoidCELoss()
    loss = FocalDiceLoss()
    loss_value = loss(logit, label)
    print('loss_value: ', loss_value)
    
    
    