import torch.nn as nn
import torch.nn.functional as F


class DFPRegressionLoss(nn.Module):

    def forward(self, prediction, target, target_mask):
        """

        :param prediction: raw values,  [B, TD]
        :param target: ground truth, [B, T, D] , torch.Tensor
        :param target_mask: valid targets mask, [B, T, D]

        :return: loss: loss value, torch.float
        """

        B, T, D = target.shape

        prediction = prediction.view(B, T, D)

        # Replace invalid (exceeding episode length) time steps to cancel their gradients
        # NOTE: We replace the invalid targets with copies of the predictions.
        pred_clones = prediction.clone().detach()
        mask_invalid = ~target_mask
        target[mask_invalid] = pred_clones[mask_invalid]

        loss = F.mse_loss(prediction, target, reduction='sum')
        return loss
