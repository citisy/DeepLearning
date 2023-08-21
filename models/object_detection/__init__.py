import torch
from torch import nn
import torchvision


def cls_nms(boxes, scores, classes, nms_thresh):
    max_coordinate = boxes.max()
    offsets = classes.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]

    return torchvision.ops.nms(boxes_for_nms, scores, nms_thresh)


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


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / torch.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


class ProjectionPooling(nn.Module):
    def __init__(self, horizontal=True):
        super().__init__()
        self.horizontal = horizontal

    def forward(self, x):
        if self.horizontal:
            return x.mean(-2).unsqueeze(-2).expand_as(x)
        else:
            return x.mean(-1).unsqueeze(-1).expand_as(x)
