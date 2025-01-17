import torch
from torch import nn
import torch.nn.functional as F


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEWithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, base_loss=None, gamma=1.5, alpha=0.25):
        """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)"""
        super().__init__()
        if base_loss is None:
            # must be nn.BCEWithLogitsLoss()
            base_loss = nn.BCEWithLogitsLoss()

        self.base_loss = base_loss
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = base_loss.reduction
        self.base_loss.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.base_loss(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class IouLoss(nn.Module):
    """
    gamma:
        focal iou:
            (iou)^gamma*Loss
            if used, set in 3 which is the default value of paper
            https://arxiv.org/pdf/2101.08158.pdf
    """

    def __init__(self, iou_method=None, gamma=0):
        super().__init__()
        if iou_method is None:
            from .object_detection import Iou
            iou_method = Iou().c_iou1D

        self.iou_method = iou_method
        self.gamma = gamma

    def forward(self, box1, box2):
        if 'w_iou1D' in str(self.iou_method):
            iou, r, ori_iou = self.iou_method(box1, box2)
            iou = iou.squeeze()
            loss = r * (1 - iou)
        else:
            iou, ori_iou = self.iou_method(box1, box2)
            iou = iou.squeeze()
            loss = 1 - iou

        ori_iou = ori_iou.detach().squeeze()
        loss = torch.pow(ori_iou, self.gamma) * loss
        loss = loss.mean()
        return loss, iou


class HingeGanLoss(nn.Module):
    def forward(self, real_y, fake_y):
        return F.relu(1. - real_y).mean() + F.relu(1. + fake_y).mean()
