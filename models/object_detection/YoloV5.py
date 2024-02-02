import math
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import cv2
from utils import os_lib, visualize
from ..losses import FocalLoss, IouLoss
from utils.torch_utils import initialize_layers
from ..layers import ConvInModule, Cache, Concat
from . import cls_nms, Iou
from ..image_classification.CspDarkNet import Backbone, C3, Conv


class Config:
    # default configs, base on yolov5l config
    in_module = dict(
        in_ch=3,
        input_size=640,
    )
    backbone = (64, (3, 6, 9, 3), (1, 2))
    neck = dict(n_c3=3)
    head = dict(
        anchors=[  # length of wh
            [(10, 13), (16, 30), (33, 23)],
            [(30, 61), (62, 45), (59, 119)],
            [(116, 90), (156, 198), (373, 326)],
        ],
        stride=(8, 16, 32)
    )
    loss = dict(
        conf_thres=0.1, nms_thres=0.6, max_det=300
    )

    default_model_multiple = {
        'yolov5x': dict(depth_multiple=1.33, width_multiple=1.25),
        'yolov5l': dict(depth_multiple=1, width_multiple=1),
        'yolov5m': dict(depth_multiple=0.67, width_multiple=0.75),
        'yolov5s': dict(depth_multiple=0.33, width_multiple=0.50),
        'yolov5n': dict(depth_multiple=0.33, width_multiple=0.25),
    }

    @classmethod
    def get(cls, name='yolov5l'):
        return dict(
            in_module_config=cls.in_module,
            **cls.make_config(cls.backbone, cls.neck, **cls.default_model_multiple[name]),
            head_config=cls.head,
            loss_config=cls.loss
        )

    @staticmethod
    def make_config(backbone_config=backbone, neck_config=neck, depth_multiple=1, width_multiple=1):
        """
        Args:
            depth_multiple: model depth multiple
            width_multiple: layer channel multiple

        Usage:
            .. code-block:: python

                from models.object_detection.YoloV5 import make_config, default_model_multiple
                Model(**make_config(**default_model_multiple['yolov5m']))
        """
        compute_width = lambda x: math.ceil(x * width_multiple / 8) * 8  # make sure that it is multiple of 8
        compute_deep = lambda n: max(round(n * depth_multiple), 1) if n > 1 else n  # depth gain

        out_ch, n_conv, cache_block_idx = backbone_config
        out_ch = compute_width(out_ch)
        n_conv_ = [compute_deep(_) for _ in n_conv]
        backbone_config = (out_ch, tuple(n_conv_), cache_block_idx)
        neck_config = neck_config.copy()
        neck_config['n_c3'] = compute_deep(neck_config['n_c3'])

        return dict(backbone_config=backbone_config, neck_config=neck_config)

    @staticmethod
    def auto_anchors(iter_data, img_size, head_config=head, **kwargs):
        """
        Usage:
            .. code-block:: python

                from models.object_detection.YoloV5 import auto_anchors
                head_config = auto_anchors(iter_data, img_size)
                Model(head_config=head_config)
        """
        anchors = AutoAnchor(iter_data, img_size=img_size, **kwargs).run(head_config['anchors'], head_config['stride'])
        head_config = head_config.copy()
        head_config['anchors'] = anchors

        return head_config


class Model(nn.Module):
    """refer to https://github.com/ultralytics/yolov5"""

    def __init__(
            self, n_classes,
            in_module=None, backbone=None, neck=None, head=None,
            in_module_config=Config.in_module, backbone_config=Config.backbone,
            neck_config=Config.neck, head_config=Config.head, loss_config=Config.loss
    ):
        super().__init__()
        self.input = in_module(**in_module_config) if in_module is not None else ConvInModule(**in_module_config)
        if backbone is None:
            self.backbone = Backbone(in_ch=self.input.out_channels, backbone_config=backbone_config)
        else:
            self.backbone = backbone(**backbone_config)
        self.neck = neck(**neck_config) if neck is not None else Neck(self.backbone.out_channels, **neck_config)
        self.head = head(**head_config) if head is not None else Head(n_classes, self.neck.out_channels, **head_config)
        self.make_loss(**loss_config)

        self.input_size = self.input.input_size
        self.grid = None
        self.stride = self.head.stride

        initialize_layers(self)

    def make_loss(self, conf_thres=0.1, nms_thres=0.6, max_det=300):
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.max_det = max_det

    def forward(self, x, gt_boxes=None, gt_cls=None):
        x = self.input(x)
        features = self.backbone(x)
        features = self.neck(features)

        if isinstance(features, torch.Tensor):
            features = [features]

        preds, losses = self.head(features, gt_boxes, gt_cls, self.input_size)

        if self.training:
            return dict(
                preds=preds,
                **losses
            )
        else:
            return self.post_process(preds)

    def post_process(self, preds):
        """

        Args:
            preds: (b, n, 4 + 1 + n_class)

        """
        preds = self.gen_preds(preds)
        result = self.parse_preds(preds)
        return result

    def gen_preds(self, preds):
        if self.grid is None:
            self.grid = []
            for i, f in enumerate(preds):
                a, h, w = f.shape[1:4]
                shape = 1, a, h, w, 2  # grid shape
                y, x = torch.arange(h).to(f), torch.arange(w).to(f)
                yv, xv = torch.meshgrid(y, x, indexing='ij')  # note, low version pytorch, do not add `indexing`
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
        return preds

    def parse_preds(self, preds):
        result = []
        for i, x in enumerate(preds):
            conf = x[:, 4]
            bboxes = x[:, :4]
            det_cls = x[:, 5:] * x[:, 4:5]  # obj_conf * cls_conf

            keep = conf > self.conf_thres
            bboxes, det_cls = bboxes[keep], det_cls[keep]

            keep, classes = (det_cls > self.conf_thres).nonzero(as_tuple=True)
            bboxes, scores = bboxes[keep], det_cls[keep, classes]

            if bboxes.shape[0] > 10 * self.max_det:
                # if num of bboxes is too large, it will raise:
                # RuntimeError: Trying to create tensor with negative dimension
                arg = torch.argsort(det_cls, descending=True)
                keep = arg < 10 * self.max_det
                bboxes, classes, scores = bboxes[keep], classes[keep], scores[keep]

            if bboxes.numel():
                # (center x, center y, width, height) to (x1, y1, x2, y2)
                _bboxes = bboxes.clone()
                bboxes[:, :2] = _bboxes[:, :2] - _bboxes[:, 2:] / 2
                bboxes[:, 2:] = _bboxes[:, :2] + _bboxes[:, 2:] / 2

                keep = cls_nms(bboxes, scores, classes, iou_threshold=self.nms_thres)
                keep = keep[:self.max_det]
                bboxes, classes, scores = bboxes[keep], classes[keep], scores[keep]

            result.append({
                "bboxes": bboxes,
                "confs": scores,
                "classes": classes.to(dtype=torch.int),
            })

        return result


class Model4Triton(Model):
    """for triton server"""

    def forward(self, x, gt_boxes=None, gt_cls=None):
        x = self.pre_process(x)
        x = self.input(x)
        features = self.backbone(x)
        features = self.neck(features)

        if isinstance(features, torch.Tensor):
            features = [features]

        preds, losses = self.head(features, gt_boxes, gt_cls, self.input_size)
        preds = self.post_process(preds)

        return preds

    def pre_process(self, x):
        """for faster infer, use uint8 input and fp16 to process"""
        x = x / 255
        x = x.to(dtype=torch.float16)
        return x

    def post_process(self, preds):
        """for faster infer, only output 500 bboxes"""
        preds = self.gen_preds(preds)
        conf = preds[:, :, 4]
        _, indices = torch.sort(conf, dim=-1, descending=True)
        indices = indices[:, :500].unsqueeze(-1).expand(-1, -1, preds.shape[-1])
        preds = preds.gather(1, indices)
        preds = preds.to(dtype=torch.float16)

        return preds


class Neck(nn.Module):
    def __init__(self, in_ch, n_c3=3):
        super().__init__()
        out_ch = in_ch // 2

        out_channels = []
        layers = [
            Conv(in_ch, out_ch, 1),
            Cache(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(1),
            C3(in_ch, out_ch, n=n_c3, shortcut=False),
        ]

        in_ch = out_ch
        out_ch = in_ch // 2
        out_channels.append(out_ch)

        layers += [
            Conv(in_ch, out_ch, 1),
            Cache(1),

            nn.Upsample(scale_factor=2, mode='nearest'),
            Concat(0),
            C3(in_ch, out_ch, n=n_c3, shortcut=False),
            Cache(0),

            Conv(out_ch, out_ch, 3, 2),
            Concat(1),
        ]

        in_ch = out_ch
        out_ch = in_ch * 2
        out_channels.append(out_ch)

        layers += [
            C3(out_ch, out_ch, n=n_c3, shortcut=False),
            Cache(1),

            Conv(out_ch, out_ch, 3, 2),
            Concat(2),
        ]

        in_ch = out_ch
        out_ch = in_ch * 2
        out_channels.append(out_ch)

        layers += [
            C3(out_ch, out_ch, n=n_c3, shortcut=False),
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


class Head(nn.Module):
    def __init__(self, n_classes, in_ches, anchors,
                 stride=(8, 16, 32), balance=[4.0, 1.0, 0.4],
                 cls_pw=1., obj_pw=1., label_smoothing=0., fl_gamma=0., iou_gamma=0, anchor_t=4.,
                 reg_gain=0.05, cls_gain=0.5, obj_gain=1.0, gr=1., iou_method=Iou().c_iou1D,
                 auto_balance=False, sort_obj_iou=False):
        super().__init__()
        self.n_classes = n_classes
        self.output_size = n_classes + 5
        self.n_layers = len(anchors)  # same to num of features
        self.n_anchors = len(anchors[0])

        self.grid = None  # init grid
        self.anchor_grid = None  # init anchor grid
        self.stride = torch.tensor(stride)  # image_size / feature_sizes
        self.register_buffer('anchors', torch.tensor(anchors).float() / self.stride.view(-1, 1, 1))  # (n_layers, n_anchors, 2)

        self.conv_list = nn.ModuleList(nn.Conv2d(x, self.output_size * self.n_anchors, 1) for x in in_ches)  # output conv
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
        self.iou_loss = IouLoss(iou_method=iou_method, gamma=iou_gamma)

        self.gr = gr
        self.auto_balance = auto_balance
        self.sort_obj_iou = sort_obj_iou
        self.anchor_t = anchor_t

    def initialize_layers(self, cf=None):
        """refer to https://arxiv.org/abs/1708.02002 section 3.3"""
        for mi, s in zip(self.conv_list, self.stride):  # from
            b = mi.bias.view(self.n_anchors, -1).detach()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (self.n_classes - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, features, gt_boxes=None, gt_cls=None, image_size=None):
        for i, f in enumerate(features):
            f = self.conv_list[i](f)
            b, _, h, w = f.shape  # num_grid = h * w

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

        losses = None

        if self.training:
            losses = self.loss(features, gt_boxes, gt_cls, image_size)
        return features, losses

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
                # iou = bbox_iou(det_reg, gt_reg, CIoU=True, xywh=True).squeeze()  # 1D
                # reg_loss += (1.0 - iou).mean()  # iou loss

                box1 = det_reg
                box2 = gt_reg
                (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
                w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
                b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
                b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
                box1 = torch.cat([b1_x1, b1_y1, b1_x2, b1_y2], 1)
                box2 = torch.cat([b2_x1, b2_y1, b2_x2, b2_y2], 1)
                iou_loss, iou = self.iou_loss(box1, box2)
                reg_loss += iou_loss

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

        return {
            'loss.reg': reg_loss,
            'loss.obj': obj_loss,
            'loss.cls': cls_loss,
            'loss': (reg_loss + obj_loss + cls_loss) * bs
        }


class AutoAnchor:
    def __init__(self, iter_data, img_size=640,
                 thr=4.0, min_bpr=0.98, mutation_prob=0.9, sigma=0.1, device=None,
                 verbose=True, stdout_method=print):
        self.stdout_method = stdout_method if verbose else os_lib.FakeIo()
        self.img_size = img_size
        self.thr = thr if thr < 1 else 1 / thr
        self.min_bpr = min_bpr
        self.mutation_prob = mutation_prob
        self.sigma = sigma
        self.device = device

        # Get label wh
        bboxes = [ret['bboxes'] for ret in iter_data]
        bboxes = np.concatenate(bboxes)
        ref_bboxes_wh = bboxes[:, -2:]

        ori_image_wh = []
        for ret in tqdm(iter_data, desc=visualize.TextVisualize.highlight_str('Load auto-anchor data')):
            if 'image_size' in ret:
                ori_image_wh.append(ret['image_size'][:2][::-1])
            else:
                image = ret['image']
                if not isinstance(image, np.ndarray):
                    image = cv2.imread(image)
                ori_image_wh.append(image.shape[:2][::-1])

        ori_image_wh = np.array(ori_image_wh)
        scale_image_wh = img_size * ori_image_wh / ori_image_wh.max(1, keepdims=True)
        scale_image_wh = np.concatenate([np.repeat(scale_image_wh[i][None, :], len(ret['bboxes']), axis=0) for i, ret in enumerate(iter_data)])
        self.scale_image_wh = scale_image_wh
        self.abs_bboxes_wh = scale_image_wh * ref_bboxes_wh
        self.ref_bboxes_wh = ref_bboxes_wh

    def run(self, anchors, stride):
        anchors = np.array(anchors)
        stride = np.array(stride)
        scale = np.random.uniform(0.9, 1.1, size=(self.scale_image_wh.shape[0], 1))  # augment scale
        jitter_abs_bboxes_wh = self.abs_bboxes_wh * scale

        stride = stride.reshape((-1, 1, 1))  # model strides

        bpr, aat = self.pr(anchors.reshape((-1, 2)), self.abs_bboxes_wh)
        self.stdout_method(f'ori bpr = {bpr}, aat = {aat}')

        bpr, aat = self.pr(anchors.reshape((-1, 2)), jitter_abs_bboxes_wh)
        self.stdout_method(f'jitter bpr = {bpr}, aat = {aat}')

        if bpr < self.min_bpr:
            na = anchors.size // 2  # number of anchors
            new_anchors = self.kmean_anchors(n=na, gen=1000)
            new_bpr, new_aat = self.pr(new_anchors, jitter_abs_bboxes_wh)
            self.stdout_method(f'jitter bpr = {new_bpr}, aat = {new_aat}')
            if new_bpr > bpr:  # replace anchors
                new_anchors = new_anchors.reshape(anchors.shape)
                a = anchors.prod(-1).mean(-1)
                da = a[-1] - a[0]  # delta a
                ds = stride[-1] - stride[0]  # delta s
                if np.sign(da) == np.sign(ds):  # same order
                    anchors = new_anchors

        anchors = np.round(anchors)
        self.stdout_method(f'anchors = {anchors.tolist()}')
        return anchors

    def kmean_anchors(self, n=9, gen=1000):
        from scipy.cluster.vq import kmeans

        large_abs_bboxes_wh = self.abs_bboxes_wh[(self.abs_bboxes_wh >= 2.0).any(1)]  # filter > 2 pixels

        try:
            assert n <= len(large_abs_bboxes_wh)  # apply overdetermined constraint
            s = large_abs_bboxes_wh.std(0)  # sigmas for whitening
            anchors = kmeans(large_abs_bboxes_wh / s, n, iter=30)[0] * s  # points
            assert n == len(anchors)  # kmeans may return fewer points than requested if wh is insufficient or too similar
        except Exception:
            anchors = np.sort(np.random.rand(n * 2)).reshape(n, 2) * self.img_size  # random init

        best_score = self.anchor_fitness(anchors, large_abs_bboxes_wh)
        pbar = tqdm(range(gen), desc=visualize.TextVisualize.highlight_str('Attempt to find better anchors'))
        for _ in pbar:
            v = np.ones(anchors.shape)
            while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
                v = (
                        (np.random.random(anchors.shape) < self.mutation_prob)
                        * np.random.randn(*anchors.shape)
                        * self.sigma + 1
                ).clip(0.3, 3.0)
            tmp_anchors = (anchors.copy() * v).clip(min=2.0)
            score = self.anchor_fitness(tmp_anchors, large_abs_bboxes_wh)
            if score > best_score:
                best_score, anchors = score, tmp_anchors.copy()
            pbar.set_postfix({'best_score': best_score})

        anchors = anchors[np.argsort(anchors.prod(1))]  # sort small to large
        return anchors

    def pr(self, anchors, wh):
        x, best = self.metric(anchors, wh)
        aat = (x > self.thr).sum(1).mean()  # anchors above threshold
        bpr = (best > self.thr).mean()  # best possible recall
        return bpr, aat

    def metric(self, anchors, wh):  # compute metrics
        if self.device:
            anchors = torch.from_numpy(anchors).to(self.device)
            wh = torch.from_numpy(wh).to(self.device)
            r = wh[:, None] / anchors[None]
            x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
            best = x.max(-1)[0]
            x = x.cpu().numpy()
            best = best.cpu().numpy()
        else:
            r = wh[:, None] / anchors[None]
            x = np.minimum(r, 1 / r).min(-1)  # ratio metric
            best = x.max(-1)
        return x, best

    def anchor_fitness(self, anchors, wh):  # mutation fitness
        _, best = self.metric(anchors, wh)
        return (best * (best > self.thr)).mean()  # fitness
