from typing import List
import torchvision
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from utils.layers import Conv, Linear, ConvInModule
from . import GetBackbone, cls_nms
from utils.torch_utils import initialize_layers

in_module_config = dict(
    in_ch=3,
    input_size=500,
)

anchor_config = dict(
    # sizes=((8 ** 2, 16 ** 2, 24 ** 2),),
    # ratios=((0.5, 1, 2),),
    sizes=((32, 64, 128, 256, 512),),
    ratios=((0.5, 1.0, 2.0),),
    box_min_len=10
)

rpn_config = dict(
    n_head_conv=1,
    max_bboxes=2000,
    max_backprop_sample=256,
    anchor_config=anchor_config
)

roi_config = dict(
    max_backprop_sample=512,
    pos_iou=0.5,
    neg_iou=0.5
)


class Model(nn.Module):
    """refer to
    paper:
        - [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
    code:
        - https://github.com/rbgirshick/py-faster-rcnn

    See Also `torchvision.models.detection.faster_rcnn`
    """

    def __init__(
            self, n_classes,
            in_module=None, backbone=None, neck=None, head=None,
            in_module_config=in_module_config, backbone_config=dict(),
            neck_config=rpn_config, head_config=roi_config,
            score_thres=0.05, nms_thres=0.5,
    ):
        super().__init__()

        self.score_thres = score_thres
        self.nms_thres = nms_thres

        self.input = in_module(**in_module_config) if in_module else ConvInModule(**in_module_config)

        if backbone is None:
            # note that, if use `torch.cuda.amp.autocast(True)`, it would become slower
            # it is the problem with torchvision.models.mobilenet
            self.backbone = GetBackbone.get_mobilenet_v2()
        elif isinstance(backbone, str):
            self.backbone = GetBackbone.get_one(backbone, backbone_config)
        else:
            self.backbone = backbone

        self.neck = neck(**neck_config) if neck else RPN(self.backbone.out_channels, **neck_config)
        self.head = head(**head_config) if head else RoIHead(self.backbone.out_channels, n_classes, **roi_config)

        initialize_layers(self)

    def forward(self, images, gt_boxes=None, gt_cls=None):
        """
        Arguments:
            images (Tensor):
            gt_boxes (List[Tensor]):
            gt_cls (List[Tensor])

        Returns:
            result (dict[Tensor]):
        """
        image_size = images.shape[-2:]

        features = self.input(images)
        features = self.backbone(features)

        if isinstance(features, torch.Tensor):
            features = [features]

        proposals, neck_loss = self.neck(images, features, gt_boxes, gt_cls)
        det_reg, det_cls, proposals, head_loss = self.head(features, proposals, image_size, gt_boxes, gt_cls)

        if self.training:
            return dict(
                proposals=proposals,
                det_reg=det_reg,
                det_cls=det_cls,
                loss=neck_loss + head_loss
            )
        else:
            return self.post_process(proposals, det_reg, det_cls, image_size)

    def post_process(self, proposals, det_reg, det_cls, image_size):
        det_cls = F.softmax(det_cls, -1)
        sample_nums = [boxes.shape[0] for boxes in proposals]
        det_reg = det_reg.split(sample_nums, 0)
        det_cls = det_cls.split(sample_nums, 0)
        result = []

        for p, r, s in zip(proposals, det_reg, det_cls):
            num_classes = r.shape[1] // 4
            r = r.reshape(-1, num_classes, 4)
            p = p.unsqueeze(1).repeat(1, num_classes, 1)
            detection = bbox2proposal(p, r, weights=(10, 10, 5, 5))  # reduce influence of reg to detection
            classes = torch.arange(num_classes, device=r.device)
            classes = classes.reshape(1, -1).repeat(len(s), 1)

            # note that, filter bg cls before, and then select the detection
            # not select the detection before and then filter bg cls before
            classes = classes[:, :-1]
            detection = detection[:, :-1]
            s = s[:, :-1]

            classes = classes.reshape(-1)
            detection = detection.reshape(-1, 4)
            s = s.reshape(-1)

            keep = s > self.score_thres
            detection, classes, s = detection[keep], classes[keep], s[keep]

            detection = clip_boxes(detection, image_size)
            keep = filter_boxes(detection, 1.)
            detection, classes, s = detection[keep], classes[keep], s[keep]

            if detection.numel():
                # note that, there are two nms strategies:
                # - all detection join nms without split class, it is faster but worse
                # - all detection join nms with split class, it is slower but better
                # it is applied for 2nd strategy
                # keep = torchvision.ops.nms(detection, s, self.nms_thresh)
                keep = cls_nms(detection, s, classes, self.nms_thres)
                detection, classes, s = detection[keep], classes[keep], s[keep]

            result.append({
                "bboxes": detection,
                "classes": classes,
                "confs": s,
            })

        return result


class RPN(nn.Module):
    """See Also `torchvision.models.detection.rpn`"""

    def __init__(
            self, in_ch, n_head_conv=1,
            pos_iou=0.7, neg_iou=0.3,
            nms_thresh=0.7, score_thresh=0.5,
            min_size=1., max_bboxes=2000, max_backprop_sample=256,
            anchor_config=dict()
    ):
        super(RPN, self).__init__()
        self.neg_iou = neg_iou
        self.pos_iou = pos_iou
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = min_size
        self.max_bboxes = max_bboxes
        self.max_backprop_sample = max_backprop_sample

        self.anchor_generator = AnchorGenerator(**anchor_config)
        self.head = RPNHead(in_ch, self.anchor_generator.num_anchors, n_head_conv)

    def gen_proposals(self, anchors, cls, reg, feature_idx, img_size):
        cls = torch.sigmoid(cls)
        box = bbox2proposal(anchors, reg)
        box = clip_boxes(box, img_size)
        keep1 = filter_boxes(box, self.min_size)

        proposals = []
        scores = []

        for i in range(cls.shape[0]):
            keep = keep1[i]
            b = box[i, keep]
            s = cls[i, keep].view(-1)
            idx = feature_idx[keep]

            max_bboxes = min(len(s), self.max_bboxes)
            keep = torch.topk(s, max_bboxes)[1]
            b, s, idx = b[keep], s[keep], idx[keep]

            keep = s > self.score_thresh
            b, s, idx = b[keep], s[keep], idx[keep]

            if b.numel():
                # keep = torchvision.ops.nms(b, s, self.nms_thresh)
                keep = cls_nms(b, s, idx, self.nms_thresh)
                b, s, idx = b[keep], s[keep], idx[keep]

            proposals.append(b)
            scores.append(s)

        return proposals, scores

    def forward(self, images, features, gt_boxes=None, gt_cls=None):
        """
        Arguments:
            images (Tensor)
            features (List[Tensor])
            gt_boxes (List[Tensor])
            gt_cls (List[Tensor])

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per image.
            losses (Tensor): the losses for the model during training. During testing, it is empty.
        """
        cls, reg = self.head(features)
        anchors, feature_info = self.anchor_generator(images, features)

        _cls = []
        _reg = []
        feature_idx = []
        for c, r in zip(cls, reg):
            c = c.view(c.shape[0], -1, 1, c.shape[2], c.shape[3])  # (b, num_anchors * 1, h, w) -> (b, num_anchors, 1, h, w)
            c = c.permute(0, 3, 4, 1, 2).contiguous()  # (b, num_anchors, 1, h, w) -> (b, h, w, num_anchors, 1)
            c = c.view(c.shape[0], -1, 1)  # (b, h, w, num_anchors, 1) -> (b, h * w * num_anchors, 1)

            r = r.view(r.shape[0], -1, 4, r.shape[2], r.shape[3])  # (b, num_anchors * 4, h, w) -> (b, num_anchors, 4, h, w)
            r = r.permute(0, 3, 4, 1, 2).contiguous()  # (b, num_anchors, 4, h, w) -> (b, h, w, num_anchors, 4)
            r = r.view(r.shape[0], -1, 4)  # (b, h, w, num_anchors, 4) -> (b, h * w * num_anchors, 4)

            _cls.append(c)
            _reg.append(r)
            feature_idx.append(c.shape[1])

        cls = torch.cat(_cls, 1)
        reg = torch.cat(_reg, 1)
        feature_idx = torch.cat([torch.full((n,), i).to(cls) for i, n in enumerate(feature_idx)])

        # note that using `detach` method, 'cause cls and reg do not backprop through
        proposals, scores = self.gen_proposals(anchors, cls.detach(), reg.detach(), feature_idx, images[0].shape[-2:])

        loss = None
        if self.training:
            loss = self.loss(reg, anchors, gt_boxes, cls, gt_cls)

        return proposals, loss

    def loss(self, det_reg, det_boxes, gt_boxes, det_cls, gt_cls):
        n_batch = len(det_boxes)
        obj_gt_cls = [torch.ones_like(c).to(c) for c in gt_cls]  # all object is pos sample
        # note that, it is compared with anchors not proposals
        labels, match_gt_boxes, match_gt_cls = divide_sample(det_boxes, gt_boxes, obj_gt_cls, self.neg_iou, self.pos_iou, allow_low_quality_matches=True)

        pos_idx = torch.where(labels == 1)[0]
        neg_idx = torch.where(labels == -1)[0]

        match_gt_cls[neg_idx] = 0
        match_gt_cls = match_gt_cls.to(dtype=torch.float32)

        max_backprop_sample = self.max_backprop_sample * n_batch
        n_pos = min(len(pos_idx), max_backprop_sample)
        n_neg = max_backprop_sample - n_pos

        pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[:n_pos]]
        neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[:n_neg]]
        total_idx = torch.cat([pos_idx, neg_idx])
        total_idx = total_idx[torch.randperm(total_idx.numel(), device=pos_idx.device)]

        # (bs, n_boxes, n) -> (bs * n_boxes, n)
        det_boxes = det_boxes.view(-1, 4)
        det_reg = det_reg.view(-1, 4)
        det_cls = det_cls.view(-1)

        det_reg = det_reg[pos_idx]
        gt_reg = proposal2deltas(det_boxes[pos_idx], match_gt_boxes[pos_idx])

        return od_loss(det_reg, gt_reg, det_cls[total_idx], match_gt_cls[total_idx], total_idx.numel())


class AnchorGenerator(nn.Module):
    def __init__(
            self,
            sizes=((8 ** 2, 16 ** 2, 24 ** 2),),
            ratios=((0.5, 1, 2),),
            box_min_len=10
    ):
        super().__init__()

        self.window_anchors = nn.ParameterList([self.gen_window_anchors(size, ratio) for size, ratio in zip(sizes, ratios)])
        self.window_anchors.requires_grad_(requires_grad=False)

        self.box_min_len = box_min_len
        self.num_anchors = [len(size) * len(ratio) for size, ratio in zip(sizes, ratios)]

        assert not any([n - self.num_anchors[0] for n in self.num_anchors]), \
            'all items in sizes or ratios must be in the same length'

        self.num_anchors = self.num_anchors[0]

    @staticmethod
    def gen_window_anchors(size, ratio):
        size = torch.as_tensor(size)
        ratio = torch.as_tensor(ratio)

        # h_ratio * w_ratio = 1
        # eg, ratio=0.5, h_ratio=0.25, w_ratio=4
        h_ratio = torch.sqrt(ratio)
        w_ratio = 1. / h_ratio

        # (len(size) * len(ratio), )
        # eg, size=64, h_ratio=0.25, hs=16
        hs = (h_ratio[:, None] * size[None, :]).view(-1)
        ws = (w_ratio[:, None] * size[None, :]).view(-1)

        # half of ratio, (len(size) * len(ratio), 4)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def gen_anchors(self, grid_sizes, image_size):
        anchors = []
        feature_info = []

        s = 0

        for i, (grid_size, base_anchors) in enumerate(zip(grid_sizes, self.window_anchors)):
            scale_ratio = image_size / grid_size
            scale_ratio = scale_ratio.to(dtype=torch.int32)

            grid_height, grid_width = grid_size
            stride_height, stride_width = scale_ratio
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            # abs coor of image
            shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)  # (feature_map.size, 4)

            # (num_anchors * feature_map.size, 4)
            shifts = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).view(-1, 4)
            e = s + shifts.shape[0]
            info = torch.tensor((i, stride_height, stride_width), device=device)
            info = torch.repeat_interleave(info.unsqueeze(0), shifts.shape[0], dim=0)
            info = torch.cat([torch.arange(s, e, device=device).unsqueeze(-1), info], dim=-1)
            s = e

            anchors.append(shifts)
            feature_info.append(info)  # (idx of anchor, idx of feature, *scale_ratio)

        # (num_features * num_anchors * feature_map.size, 4)
        anchors = torch.cat(anchors).detach()  # note that, do not know why anchors has grad_fn, so detach it
        feature_info = torch.cat(feature_info)

        return anchors, feature_info

    def forward(self, images, features):
        """

        Args:
            images:
            features:

        Returns:
            anchors: (batch_size, n_boxes, 4)
            feature_info: (n_boxes, 4)
                4 gives (idx of anchor, idx of feature, scale_ratio_w, scale_ratio_h)
        """
        # per pixel of feature map per grid
        grid_sizes = [torch.tensor(feature_map.shape[-2:], device=images.device) for feature_map in features]
        image_size = torch.tensor(images.shape[-2:], device=images.device)

        anchors, feature_info = self.gen_anchors(grid_sizes, image_size)

        # to reduce time, filter in advance
        # keep = filter_boxes(anchors, image_size, self.box_min_len)
        # anchors = anchors[keep]

        anchors = anchors.repeat(images.shape[0], 1, 1)

        return anchors, feature_info


class RPNHead(nn.Module):
    def __init__(self, in_ch, num_anchors, n_conv):
        super().__init__()

        self.num_anchors = num_anchors
        self.conv_seq = nn.Sequential(*[Conv(in_ch, in_ch, 3, is_norm=False) for _ in range(n_conv)])

        # it is 2 cls in paper, if setting 2, use softmax instead after
        self.cls = nn.Conv2d(in_ch, num_anchors, 1)
        self.reg = nn.Conv2d(in_ch, num_anchors * 4, 1)

    def forward(self, features):
        cls, reg = [], []

        for f in features:
            f = self.conv_seq(f)
            cls.append(self.cls(f))
            reg.append(self.reg(f))

        return cls, reg


class RoIHead(nn.Module):
    def __init__(
            self, in_ch, n_classes,
            pooling_size=7, max_backprop_sample=512,
            pos_iou=0.5, neg_iou=0.5,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.output_size = n_classes + 1   # the last cls is bg sample
        self.neg_iou = neg_iou
        self.pos_iou = pos_iou
        self.max_backprop_sample = max_backprop_sample
        self.pooling_size = pooling_size

        self.neck = nn.Sequential(
            nn.Flatten(),
            Linear(in_ch * pooling_size * pooling_size, in_ch),
            Linear(in_ch, in_ch),
        )

        # note that per cls per box reg not all cls per box reg
        self.reg_fc = Linear(in_ch, self.output_size * 4, act=None)
        self.cls_fc = Linear(in_ch, self.output_size, act=None)

    def gen_detections(self, proposals, features, image_size):
        detections = []
        box = torch.cat(proposals, dim=0)
        ids = torch.cat(
            [torch.full_like(b[:, :1], i, layout=torch.strided).to(box) for i, b in enumerate(proposals)],
            dim=0,
        )
        box = torch.cat([ids, box], dim=1)
        for j, feature in enumerate(features):
            box = box.to(dtype=feature.dtype)
            scale_ratio = feature.shape[2] / image_size[0]
            # note that, original faster rcnn use roi_pool
            # and then, it is a very interesting thing that
            # when scaling the box by myself and set `spatial_scale=1`, the training goes failed
            # detection = torchvision.ops.roi_pool(feature, box, self.pooling_size, spatial_scale=scale_ratio)
            detection = torchvision.ops.roi_align(feature, box, self.pooling_size, spatial_scale=scale_ratio, sampling_ratio=2)

            detections.append(detection)

        detections = torch.cat(detections)

        return detections

    def forward(self, features, proposals, image_size, gt_boxes=None, gt_cls=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_size (List[int])
            gt_boxes (List[Tensor])
            gt_cls (List[Tensor])
        """

        if self.training:
            _proposals = []
            _gt_cls = []
            _gt_reg = []
            for p, g, c in zip(proposals, gt_boxes, gt_cls):
                # note that add gt boxes to proposals, make sure that have pos samples
                p = torch.cat([p, g])

                _labels, match_gt_boxes, match_gt_cls = divide_sample([p], [g], [c], self.neg_iou, self.pos_iou)

                pos_idx = torch.where(_labels == 1)[0]
                neg_idx = torch.where(_labels == -1)[0]

                # the last cls is neg sample
                match_gt_cls[neg_idx] = self.n_classes

                n_pos = min(len(pos_idx), self.max_backprop_sample)
                n_neg = self.max_backprop_sample - n_pos

                pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[:n_pos]]
                neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[:n_neg]]
                total_idx = torch.cat([pos_idx, neg_idx])
                total_idx = total_idx[torch.randperm(total_idx.numel(), device=pos_idx.device)]

                _proposals.append(p[total_idx])
                _gt_cls.append(match_gt_cls[total_idx])
                _gt_reg.append(proposal2deltas(p[total_idx], match_gt_boxes[total_idx]))

            proposals = _proposals
            gt_cls = _gt_cls
            gt_reg = _gt_reg

        else:
            gt_cls = None
            gt_reg = None

        x = self.gen_detections(proposals, features, image_size)
        x = self.neck(x)
        cls = self.cls_fc(x)
        reg = self.reg_fc(x)

        loss = None
        if self.training:
            loss = self.loss(reg, gt_reg, cls, gt_cls)

        return reg, cls, proposals, loss

    def loss(self, det_reg, gt_reg, det_cls, gt_cls):
        gt_cls = torch.cat(gt_cls, dim=0).long()
        gt_reg = torch.cat(gt_reg, dim=0)
        pos_idx = gt_cls < self.n_classes
        cls_pos_idx = gt_cls[pos_idx]
        det_reg = det_reg.reshape(len(det_cls), -1, 4)
        pos_det_reg = det_reg[pos_idx, cls_pos_idx]
        pos_gt_reg = gt_reg[pos_idx]

        return od_loss(pos_det_reg, pos_gt_reg, det_cls, gt_cls, len(det_cls))


def filter_boxes(boxes, min_size):
    ws, hs = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
    keep = (ws >= min_size) & (hs >= min_size)

    return keep


def clip_boxes(boxes, image_size):
    """clip"""
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = image_size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=boxes.dim())
    boxes = clipped_boxes.reshape(boxes.shape)

    return boxes


def bbox2proposal(boxes, deltas, weights=(1, 1, 1, 1)):
    widths = boxes[..., 2] - boxes[..., 0] + 1.0
    heights = boxes[..., 3] - boxes[..., 1] + 1.0
    ctr_x = boxes[..., 0] + 0.5 * widths
    ctr_y = boxes[..., 1] + 0.5 * heights

    dx = deltas[..., 0] / weights[0]
    dy = deltas[..., 1] / weights[1]
    dw = deltas[..., 2] / weights[2]
    dh = deltas[..., 3] / weights[3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    proposal = boxes.clone()
    proposal[..., 0] = pred_ctr_x - 0.5 * pred_w
    proposal[..., 1] = pred_ctr_y - 0.5 * pred_h
    proposal[..., 2] = pred_ctr_x + 0.5 * pred_w
    proposal[..., 3] = pred_ctr_y + 0.5 * pred_h

    return proposal


def proposal2deltas(det_boxes, gt_boxes):
    ex_widths = det_boxes[..., 2] - det_boxes[..., 0] + 1.0
    ex_heights = det_boxes[..., 3] - det_boxes[..., 1] + 1.0
    ex_ctr_x = det_boxes[..., 0] + 0.5 * ex_widths
    ex_ctr_y = det_boxes[..., 1] + 0.5 * ex_heights

    gt_widths = gt_boxes[..., 2] - gt_boxes[..., 0] + 1.0
    gt_heights = gt_boxes[..., 3] - gt_boxes[..., 1] + 1.0
    gt_ctr_x = gt_boxes[..., 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[..., 1] + 0.5 * gt_heights

    dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    dw = torch.log(gt_widths / ex_widths)
    dh = torch.log(gt_heights / ex_heights)

    deltas = torch.stack((dx, dy, dw, dh), -1)

    return deltas


def divide_sample(det_boxes, gt_boxes, gt_cls, neg_iou, pos_iou, allow_low_quality_matches=False):
    """divide pred sample to pos and neg"""
    labels = []
    match_gt_boxes = []
    match_gt_cls = []

    for det, gt, cls in zip(det_boxes, gt_boxes, gt_cls):
        device = det.device

        label = torch.zeros((det.shape[0],)).type_as(det)
        gt_idx = torch.zeros((det.shape[0],), dtype=torch.int32, device=device)

        if gt.numel():
            iou_mat = torchvision.ops.box_iou(gt, det)
            iou, gt_idx = iou_mat.max(dim=0)

            label[iou < neg_iou] = -1
            label[iou > pos_iou] = 1

            if allow_low_quality_matches:
                iou, _ = iou_mat.max(dim=1)
                idx = torch.where(iou_mat == iou[:, None])[1]
                label[idx] = 1

        labels.append(label)
        match_gt_boxes.append(gt[gt_idx])  # (n_det, 4)
        match_gt_cls.append(cls[gt_idx])  # (n_det, n_cls)

    labels = torch.cat(labels)
    match_gt_boxes = torch.cat(match_gt_boxes)
    match_gt_cls = torch.cat(match_gt_cls)

    return labels, match_gt_boxes, match_gt_cls


def od_loss(pos_det_reg, pos_gt_reg, det_cls, gt_cls, n_sample, a=0.5):
    reg_loss = F.smooth_l1_loss(
        pos_det_reg,
        pos_gt_reg,
        beta=1 / 9,
        reduction="sum",
    ) / n_sample

    if len(det_cls.shape) == 1:
        cls_loss = F.binary_cross_entropy_with_logits(det_cls, gt_cls)
    else:
        cls_loss = F.cross_entropy(det_cls, gt_cls)

    return a * reg_loss + (1 - a) * cls_loss
