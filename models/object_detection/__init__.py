import torch
from torch import nn
import torchvision


def cls_nms(bboxes, scores, classes, nms_method=torchvision.ops.nms, **nms_kwargs):
    max_coordinate = bboxes.max()
    offsets = classes.to(bboxes) * (max_coordinate + 1)
    boxes_for_nms = bboxes + offsets[:, None]

    return nms_method(boxes_for_nms, scores, **nms_kwargs)


def soft_nms(bboxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.25):
    """https://arxiv.org/pdf/1704.04503.pdf"""
    bboxes = bboxes.contiguous()
    _, arg = scores.sort(0, descending=True)
    keep = []

    while arg.numel() > 0:
        if arg.numel() == 1:
            i = arg.item()
            keep.append(i)
            break
        else:
            i = arg[0].item()
            keep.append(i)

        iou = torchvision.ops.box_iou(bboxes[i], bboxes[arg[1:]])
        idx = (iou > iou_threshold).nonzero().squeeze()
        if idx.numel() > 0:
            iou = iou[idx]
            new_scores = torch.exp(-torch.pow(iou, 2) / sigma)
            scores[arg[idx + 1]] *= new_scores

        new_idx = (scores[arg[1:]] > score_threshold).nonzero().squeeze()
        if new_idx.numel() == 0:
            break
        else:
            new_scores = scores[arg[new_idx + 1]]
            max_score_index = torch.argmax(new_scores)

            # make sure that score in index 0 is the max value of scores
            if max_score_index != 0:
                new_idx[[0, max_score_index]] = new_idx[[max_score_index, 0]]

            # cause new_idx begin from index 1 of arg, so it must add 1
            arg = arg[new_idx + 1]

    return torch.tensor(keep, device=bboxes.device)


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

            keep = self.nms(bx)
            bx = bx[keep]

            if self.keep_shape:
                if bx.shape[0] < self.max_anchors:
                    bx = torch.cat([bx, torch.zeros((50 - bx.shape[0], bx.shape[1]), device=bx.device, dtype=bx.dtype) - 1])
                else:
                    # [bs, 1000]
                    keep = bx[..., 4].topk(k=50)[1]
                    keep = keep.unsqueeze(-1).expand(-1, 6)
                    bx = bx.gather(0, keep)

            preds[i] = bx

        if self.keep_shape:
            preds = torch.tensor(preds)

        return preds

    def nms(self, bx):
        """see also `torchvision.ops.nms`"""

        arg = torch.argsort(bx, dim=-1, descending=True)
        keep = []
        while arg.numel() > 0:
            i = arg[0].item()
            keep.append(i)

            if arg.numel() == 1:
                break

            iou = torchvision.ops.box_iou(bx[i, :4], bx[arg[1:], 4])
            fi = iou < self.iou_thres
            arg = arg[fi]

        return torch.tensor(keep, device=bx.device)


class Area:
    @staticmethod
    def real_areas(box):
        """Area(box) = (x2 - x1) * (y2 - y1)

        Args:
            box(np.array): shape=(N, 4), 4 means xyxy.

        Returns:
            real_areas(np.array): shape=(N, )
        """
        return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

    @staticmethod
    def intersection_areas1D(box1, box2):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

        return (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    @classmethod
    def union_areas1D(cls, box1, box2, inter=None):
        area1 = cls.real_areas(box1)
        area2 = cls.real_areas(box2)
        if inter is None:
            inter = cls.intersection_areas1D(box1, box2)

        return area1 + area2 - inter

    @staticmethod
    def outer_areas1D(box1, box2):
        """outer rectangle area

        Args:
            box1(np.array): shape=(N, 4), 4 means xyxy.
            box2(np.array): shape=(M ,4), 4 means xyxy.

        Returns:
            outer_areas(np.array): shape=(N, M)
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
        return (torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)).clamp(0) * (torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)).clamp(0)


class Iou:
    def __init__(self, alpha=1, eps=1e-6):
        """

        Args:
            alpha:
                alpha iou, https://arxiv.org/abs/2110.13675
                if used, set in 3 which is the default value of paper
            eps:
                note that, do not set too small, 'cause it will
                when run in low computational accuracy mode
                e.g. np.array(1e-12, dtype=np.float16) -> array(0., dtype=float16)
        """
        self.alpha = alpha
        self.eps = eps

    def iou1D(self, box1, box2, inter=None, union=None):
        """box1 and box2 must have the same shape

        Args:
            box1: (N, 4)
            box2: (N, 4)
            inter:
            union:

        Returns:
            iou_mat(np.array): shape=(N, )
        """

        if inter is None:
            inter = Area.intersection_areas1D(box1, box2)

        if union is None:
            union = Area.union_areas1D(box1, box2, inter)

        ori_iou = inter / (union + self.eps)
        return ori_iou ** self.alpha, ori_iou

    def u_iou1D(self, box1, box2):
        area2 = Area.real_areas(box2)
        inter = Area.intersection_areas1D(box1, box2)
        ori_iou = inter / (area2 + self.eps)

        return ori_iou ** self.alpha, ori_iou

    def g_iou1D(self, box1, box2):
        outer = Area.outer_areas1D(box1, box2)
        inter = Area.intersection_areas1D(box1, box2)
        union = Area.union_areas1D(box1, box2)
        iou, ori_iou = self.iou1D(box1, box2, inter=inter, union=union)

        return iou - ((outer - union) / outer) ** self.alpha, ori_iou

    def d_iou1D(self, box1, box2, iou=None, ori_iou=None):
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c = cw ** 2 + ch ** 2 + self.eps
        d = ((b2_x1 - b1_x1 + b2_x2 - b1_x2) ** 2 + (b2_y1 - b1_y1 + b2_y2 - b1_y2) ** 2) / 4

        if iou is None:
            iou, ori_iou = self.iou1D(box1, box2)

        return iou - (d / c) ** self.alpha, ori_iou

    def c_iou1D(self, box1, box2, a=None, v=None):
        iou, ori_iou = self.iou1D(box1, box2)
        d_iou, ori_iou = self.d_iou1D(box1, box2, iou=iou, ori_iou=ori_iou)

        if v is None:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

            v = (4 / torch.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        if a is None:
            with torch.no_grad():
                a = v / (1 - iou + v + self.eps)

        return d_iou - (a * v) ** self.alpha, ori_iou

    def e_iou1D(self, box1, box2):
        """Efficient IOU
        https://arxiv.org/pdf/2101.08158.pdf
        """
        iou, ori_iou = self.iou1D(box1, box2)
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
        rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
        c2 = (cw ** 2 + ch ** 2) + self.eps
        cw2 = cw ** 2 + self.eps
        ch2 = ch ** 2 + self.eps
        # if Focal:
        #     return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(inter / (union + eps), gamma)  # Focal_EIou
        a = (rho2 / c2) ** self.alpha
        b = (rho_w2 / cw2) ** self.alpha
        c = (rho_h2 / ch2) ** self.alpha

        return iou - a - b - c, ori_iou

    def s_iou1D(self, box1, box2):
        """Scylla Iou
        https://arxiv.org/pdf/2205.12740.pdf"""
        iou, ori_iou = self.iou1D(box1, box2)
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
        ct_w = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5  # center width
        ct_h = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5  # center height

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps

        sigma = torch.pow(ct_w ** 2 + ct_h ** 2, 0.5)
        sin_alpha = torch.abs(ct_w) / sigma
        sin_beta = torch.abs(ct_h) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha > threshold, sin_beta, sin_alpha)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - torch.pi / 2)  # 1-2sin(x)^2=cos(2x)

        rho_x = (ct_w / cw) ** 2
        rho_y = (ct_h / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)

        return iou - (0.5 * (distance_cost + shape_cost)) ** self.alpha, ori_iou

    def w_iou1D(self, box1, box2):
        """Wise IoU v1
        https://arxiv.org/pdf/2301.10051v1.pdf"""
        iou, ori_iou = self.iou1D(box1, box2)
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        c2 = (cw ** 2 + ch ** 2) + self.eps

        r = torch.exp(rho2 / c2)

        # if scale:
        #     return getattr(WIoU_Scale, '_scaled_loss')(wise_scale), (1 - iou) * torch.exp((rho2 / c2)), iou  # WIoU v3 https://arxiv.org/abs/2301.10051
        return iou, r, ori_iou

    def w_v3_iou1D(self):
        """Wise IoU v3
        https://arxiv.org/abs/2301.10051"""


class WIoU_Scale:
    """
    monotonous: {
            None: origin v1
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
        momentum: The momentum of running mean
    """
    iou_mean = 1.
    _momentum = 1 - pow(0.5, exp=1 / 7000)
    _is_train = True

    def __init__(self, iou, monotonous=False):
        self.iou = iou
        self.monotonous = monotonous
        self._update(self)

    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1


class ProjectionPooling(nn.Module):
    def __init__(self, horizontal=True):
        super().__init__()
        self.horizontal = horizontal

    def forward(self, x):
        if self.horizontal:
            return x.mean(-2).unsqueeze(-2).expand_as(x)
        else:
            return x.mean(-1).unsqueeze(-1).expand_as(x)
