import torch
from torch import nn
import torchvision


class NMS(nn.Module):
    def __init__(self, conf_thres=0.25, iou_thres=0.45, keep_shape=False, max_anchors=50):
        super().__init__()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.keep_shape = keep_shape
        self.max_anchors = max_anchors

    def forward(self, x):
        preds = [torch.zeros((0, 6), device=x.device)] * x.shape[0]

        for i, bx in enumerate(x):
            bx = bx[bx[..., 4] > self.conf_thres]

            idx = self.nms(bx)
            bx = bx[idx]

            if self.keep_shape:
                if bx.shape[0] < self.max_anchors:
                    bx = torch.cat([bx, torch.zeros((50 - bx.shape[0], bx.shape[1]), device=bx.device, dtype=bx.dtype) - 1])
                else:
                    # [bs, 1000]
                    idx = bx[..., 4].topk(k=50)[1]
                    idx = idx.unsqueeze(-1).expand(-1, 6)
                    bx = bx.gather(0, idx)

            preds[i] = bx

        if self.keep_shape:
            preds = torch.tensor(preds)

        return preds

    def nms(self, bx):
        """see also `torchvision.ops.nms`"""

        arg = torch.argsort(bx, dim=-1, descending=True)

        idx = []
        while arg.numel() > 0:
            i = arg[0]
            idx.append(i)

            if arg.numel() == 1:
                break

            iou = torchvision.ops.box_iou(bx[i, :4], bx[:, 4])
            fi = iou < self.iou_thres
            arg = arg[fi]

        return torch.tensor(idx, device=bx.device)

