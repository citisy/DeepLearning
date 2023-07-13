import math
import torch
from torch import nn
from utils.loss import FocalLoss
from utils.torch_utils import initialize_layers
from utils.layers import Conv, ConvInModule
from . import cls_nms, bbox_iou
from ..image_classifier.DarkNet import CspDarkNet, Cache, C3, darknet_config

in_module_config = dict(
    in_ch=3,
    input_size=640,
)

neck_config = dict()

head_config = dict(
    anchors=[  # length of wh
        [(10, 13), (16, 30), (33, 23)],
        [(30, 61), (62, 45), (59, 119)],
        [(116, 90), (156, 198), (373, 326)],
    ],

)


class Model(nn.Module):
    """refer to https://github.com/ultralytics/yolov5"""

    def __init__(
            self, n_classes,
            in_module=None, backbone=None, neck=None, head=None,
            in_module_config=in_module_config, backbone_config=darknet_config,
            neck_config=neck_config, head_config=head_config,
            conf_thres=0.001, nms_thres=0.6, max_det=300
    ):
        super().__init__()
        self.input = in_module(**in_module_config) if in_module else ConvInModule(**in_module_config)
        if backbone is None:
            self.backbone = CspDarkNet(in_ch=self.input.out_channels, conv_config=backbone_config)
        else:
            self.backbone = backbone(**backbone_config)
        self.neck = neck(**neck_config) if neck else Neck(self.backbone.out_channels)
        self.head = head(**head_config) if head else Head(n_classes, self.neck.out_channels, **head_config)

        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.max_det = max_det
        self.input_size = self.input.input_size
        self.grid = None
        self.stride = self.head.stride

        initialize_layers(self)

    def forward(self, x, gt_boxes=None, gt_cls=None):
        features = self.backbone(x)
        features = self.neck(features)

        if isinstance(features, torch.Tensor):
            features = [features]

        preds, loss = self.head(features, gt_boxes, gt_cls, self.input_size)

        if self.training:
            return dict(
                preds=preds,
                loss=loss
            )
        else:
            return self.post_process(preds)

    def post_process(self, preds):
        """

        Args:
            preds: (b, n, 4 + 1 + n_class)

        """
        if self.grid is None:
            self.grid = []
            for i, f in enumerate(preds):
                a, h, w = f.shape[1:4]
                shape = 1, a, h, w, 2  # grid shape
                y, x = torch.arange(h).to(f), torch.arange(w).to(f)
                yv, xv = torch.meshgrid(y, x, indexing='ij')
                grid = torch.stack((xv, yv), 2).expand(shape)
                self.grid.append(grid)

        z = []
        for i, f in enumerate(preds):
            xy, wh, s = f.split((2, 2, f.shape[-1] - 4), -1)
            # to abs box
            xy = (xy + self.grid[i]) * self.stride[i]
            wh = wh * self.stride[i]
            s = s.sigmoid()
            f = torch.cat([xy, wh, s], -1).view(f.shape[0], -1, f.shape[-1])
            z.append(f)

        preds = torch.cat(z, 1)

        result = []
        for i, x in enumerate(preds):
            conf = x[:, 4]
            bboxes = x[:, :4]
            det_cls = x[:, 5:] * x[:, 4:5]  # obj_conf * cls_conf

            keep = conf > self.conf_thres
            bboxes, det_cls = bboxes[keep], det_cls[keep]

            keep, classes = (det_cls > self.conf_thres).nonzero(as_tuple=False).T
            bboxes, scores = bboxes[keep], det_cls[keep, classes]

            if bboxes.numel():
                # (center x, center y, width, height) to (x1, y1, x2, y2)
                _bboxes = bboxes.clone()
                bboxes[:, :2] = _bboxes[:, :2] - _bboxes[:, 2:] / 2
                bboxes[:, 2:] = _bboxes[:, :2] + _bboxes[:, 2:] / 2

                keep = cls_nms(bboxes, scores, classes, self.nms_thres)
                keep = keep[:self.max_det]
                bboxes, classes, scores = bboxes[keep], classes[keep], scores[keep]

            result.append({
                "bboxes": bboxes,
                "confs": scores,
                "classes": classes.to(dtype=torch.int),
            })

        return result


class Neck(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        out_ch = in_ch // 2

        out_channels = []
        layers = [
            Conv(in_ch, out_ch, 1, act=nn.SiLU()),
            Cache(),

            nn.Upsample(None, 2, 'nearest'),
            Concat(1),
            C3(in_ch, out_ch, n=3, shortcut=False),
        ]

        in_ch = out_ch
        out_ch = in_ch // 2
        out_channels.append(out_ch)

        layers += [
            Conv(in_ch, out_ch, 1, act=nn.SiLU()),
            Cache(1),

            nn.Upsample(None, 2, 'nearest'),
            Concat(0),
            C3(in_ch, out_ch, n=3, shortcut=False),
            Cache(0),

            Conv(out_ch, out_ch, 3, 2, act=nn.SiLU()),
            Concat(1),
        ]

        in_ch = out_ch
        out_ch = in_ch * 2
        out_channels.append(out_ch)

        layers += [
            C3(out_ch, out_ch, n=3, shortcut=False),
            Cache(1),

            Conv(out_ch, out_ch, 3, 2, act=nn.SiLU()),
            Concat(2),
        ]

        in_ch = out_ch
        out_ch = in_ch * 2
        out_channels.append(out_ch)

        layers += [
            C3(out_ch, out_ch, n=3, shortcut=False),
            Cache(2)
        ]

        self.conv_list = nn.ModuleList(layers)
        self.out_channels = out_channels

    def forward(self, features):
        x = features.pop(-1)
        for m in self.conv_list:
            if isinstance(m, (Concat, Cache)):
                x, features = m(x, features)
            else:
                x = m(x)

        return features


class Concat(nn.Module):
    def __init__(self, idx, replace=False):
        super().__init__()
        self.idx = idx
        self.replace = replace

    def forward(self, x, features):
        x = torch.cat([x, features[self.idx]], 1)

        if self.replace:
            features[self.idx] = x

        return x, features


class Head(nn.Module):
    def __init__(self, n_classes, in_ches, anchors,
                 stride=(8, 16, 32), balance=[4.0, 1.0, 0.4],
                 cls_pw=1., obj_pw=1., label_smoothing=0., fl_gamma=0., anchor_t=4.,
                 reg_gain=0.05, cls_gain=0.5, obj_gain=1.0, gr=1.,
                 auto_balance=False, sort_obj_iou=False):
        super().__init__()
        self.n_classes = n_classes
        self.output_size = n_classes + 5
        self.n_layers = len(anchors)
        self.n_anchors = len(anchors[0])

        self.grid = None  # init grid
        self.anchor_grid = None  # init anchor grid
        self.stride = torch.tensor(stride)  # image_size / feature_sizes
        self.register_buffer('anchors', torch.tensor(anchors).float() / self.stride.view(-1, 1, 1))  # shape(nl,na,2)

        self.conv_list = nn.ModuleList(nn.Conv2d(x, self.output_size * self.n_anchors, 1) for x in in_ches)  # output conv
        self._initialize_biases()

        self.reg_gain, self.cls_gain, self.obj_gain = reg_gain, cls_gain, obj_gain

        cls_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw]))
        obj_bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw]))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        smooth_bce = lambda eps: (1.0 - 0.5 * eps, 0.5 * eps)
        self.cp, self.cn = smooth_bce(eps=label_smoothing)  # positive, negative BCE targets

        # Focal loss
        if fl_gamma > 0:
            cls_bce_loss, obj_bce_loss = FocalLoss(cls_bce_loss, fl_gamma), FocalLoss(obj_bce_loss, fl_gamma)

        self.balance = balance
        self.base_stride_idx = list(self.stride).index(16) if auto_balance else 0
        self.cls_bce_loss, self.obj_bce_loss = cls_bce_loss, obj_bce_loss

        self.gr = gr
        self.auto_balance = auto_balance
        self.sort_obj_iou = sort_obj_iou
        self.anchor_t = anchor_t

    def _initialize_biases(self, cf=None):
        """refer to https://arxiv.org/abs/1708.02002 section 3.3"""
        for mi, s in zip(self.conv_list, self.stride):  # from
            b = mi.bias.view(self.n_anchors, -1).detach()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (self.n_classes - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, features, gt_boxes=None, gt_cls=None, image_size=None):
        for i, f in enumerate(features):
            f = self.conv_list[i](f)  # conv
            b, _, h, w = f.shape

            # (b, n_anchors * output_size, h, w) -> (b, n_anchors, h, w, output_size)
            f = f.view(b, self.n_anchors, self.output_size, h, w).permute(0, 1, 3, 4, 2).contiguous()
            features[i] = f

        if self.anchor_grid is None:  # only run once
            self.anchor_grid = []
            for i, f in enumerate(features):
                h, w = f.shape[2:4]
                anchor_grid = (self.anchors[i]).view((1, self.n_anchors, 1, 1, 2)).expand(1, self.n_anchors, h, w, 2)
                self.anchor_grid.append(anchor_grid)

        # note that, do not use in place mode, like f[:,:] = f[:,:]
        # it would make training going wrong and loss going nan
        for i, f in enumerate(features):
            xy, wh, _ = f.split((2, 2, self.n_classes + 1), -1)
            xy = 2 * xy.sigmoid() - 0.5
            wh = (2 * wh.sigmoid()) ** 2 * self.anchor_grid[i]
            features[i] = torch.cat([xy, wh, _], -1)

        loss = None

        if self.training:
            loss = self.loss(features, gt_boxes, gt_cls, image_size)
        return features, loss

    def loss(self, features, gt_boxes, gt_cls, image_size):  # predictions, targets
        """

        Args:
            features (List[torch.Tensor]): n_features * (b, n_anchors, h, w, 4 + 1 + n_class)

        Returns:

        """
        device = features[0].device
        cls_loss = torch.zeros(1, device=device)  # class loss
        reg_loss = torch.zeros(1, device=device)  # box loss
        obj_loss = torch.zeros(1, device=device)  # object loss

        feature_sizes = [torch.tensor(_.shape[2:4], device=device) for _ in features]
        targets = []
        for _, (boxes, cls) in enumerate(zip(gt_boxes, gt_cls)):
            # top xyxy to center xywh
            convert_bbox = boxes.clone()
            convert_bbox[:, 0:2] = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
            convert_bbox[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]
            convert_bbox = convert_bbox / image_size
            boxes = convert_bbox

            targets.append(torch.cat([
                boxes,
                cls[:, None],
                torch.full((len(cls), 1), _).to(cls),
            ], 1))

        targets = torch.cat(targets, 0)
        nt = targets.shape[0]
        ai = torch.arange(self.n_anchors).to(targets).view(self.n_anchors, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(self.n_anchors, 1, 1), ai[..., None]), 2)  # (na, nt, 7), 7 gives xywh+cls+bi+ai

        g = 0.5  # bias
        off = torch.tensor([
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],  # j,k,l,m
            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
        ], device=targets.device).float() * g  # offsets

        for i, (pi, anchors, hw) in enumerate(zip(features, self.anchors, feature_sizes)):
            if not nt:
                continue

            t = targets.clone()
            t[..., :4] *= hw[[1, 0, 1, 0]]

            r = t[..., 2:4] / anchors[:, None]  # wh ratio
            keep = torch.max(r, 1 / r).max(2)[0] < self.anchor_t  # compare
            t = t[keep]

            # Offsets
            gxy = t[:, :2]  # grid xy
            gxy_inv = hw[[1, 0]] - gxy  # inverse
            j, k = ((gxy % 1 < g) & (gxy > 1)).T
            l, m = ((gxy_inv % 1 < g) & (gxy_inv > 1)).T

            keep = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[keep]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[keep]

            pos = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=device)  # target obj

            if t.numel():
                gxy, gwh, cba = t.split((2, 2, 3), 1)
                c, b, a = cba.long().T

                gij = (gxy - offsets).long()
                gi, gj = gij.T  # grid indices
                gi, gj = gi.clamp_(0, hw[0] - 1), gj.clamp_(0, hw[1] - 1)

                # reg: xi - i
                gt_reg = torch.cat((gxy - gij, gwh), 1)
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.n_classes), 1)  # target-subset of predictions

                # Regression
                det_reg = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(det_reg, gt_reg, CIoU=True, xywh=True).squeeze()     # 1D

                # box1 = det_reg
                # box2 = gt_reg
                # (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
                # w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
                # b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
                # b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
                # box1 = torch.cat([b1_x1, b1_y1, b1_x2, b1_y2], 1)
                # box2 = torch.cat([b2_x1, b2_y1, b2_x2, b2_y2], 1)
                # iou = torchvision.ops.complete_box_iou(box1, box2)    # 2D
                # iou = iou[range(len(pbox)), range(len(pbox))]

                reg_loss += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(pos.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                pos[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.n_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=device)  # targets
                    t[range(c.shape[0]), c] = self.cp
                    cls_loss += self.cls_bce_loss(pcls, t)  # BCE

            obji = self.obj_bce_loss(pi[..., 4], pos)
            obj_loss += obji * self.balance[i]  # obj loss
            if self.auto_balance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.auto_balance:
            self.balance = [x / self.balance[self.base_stride_idx] for x in self.balance]

        reg_loss *= self.reg_gain
        obj_loss *= self.obj_gain
        cls_loss *= self.cls_gain
        bs = obj_loss.shape[0]  # batch size

        return (reg_loss + obj_loss + cls_loss) * bs
