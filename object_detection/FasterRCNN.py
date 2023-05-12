import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from image_classifier.ResNet import ResNet
from utils.layers import Conv, Linear, ConvInModule

# some config examples
in_module_config = dict(
    in_ch=3,
    input_size=500,
)

resnet_config = dict(
    in_module=nn.Module(),
    out_module=nn.Module(),
    conv_config=((3, 64, 2), (4, 128, 2), (6, 256, 2))
)

anchor_config = dict(
    sizes=((8 ** 2, 16 ** 2, 24 ** 2),),
    ratios=((0.5, 1, 2),),
    box_min_len=10
)

rpn_config = dict(
    n_conv=1,
    max_bboxes=2000,
    max_backprop_sample=256,
    box_min_len=10,
    anchor_config=anchor_config
)

roi_config = dict(
    max_backprop_sample=256,
    pos_iou=0.5,
    neg_iou=0.5
)


class FasterRCNN(nn.Module):
    """
    See Also `torchvision.models.detection.faster_rcnn`
    """

    def __init__(
            self,
            output_size=None,
            in_module=None, backbone=None, neck=None, head=None,
            in_module_config=in_module_config,
            backbone_config=resnet_config,
            neck_config=rpn_config,
            head_config=roi_config,
            **kwargs
    ):
        super().__init__()

        self.input = in_module(**in_module_config) if in_module else ConvInModule(**in_module_config)
        self.backbone = backbone(**backbone_config) if backbone else ResNet(**backbone_config).conv_seq
        self.neck = neck(**neck_config) if neck else RPN(256, **neck_config)
        self.head = head(**head_config) if head else ROIHead(256, 7, output_size, **head_config)

        self.output_size = self.head.output_size

    def forward(self, x, gt_boxes=None, gt_cls=None):
        x = self.input(x)
        features = self.backbone(x)

        if isinstance(features, torch.Tensor):
            features = [features]

        proposals, scores, reg, feature_infos, neck_loss = self.neck(x, features, gt_boxes, gt_cls)
        det_reg, det_cls, _feature_infos, head_loss = self.head(proposals, features, feature_infos, gt_boxes, gt_cls)

        _r = []
        _c = []
        for i in range(len(x)):
            idx = _feature_infos == i
            _r.append(det_reg[idx])
            _c.append(det_cls[idx])

        det_reg = _r
        det_cls = _c

        if self.training:
            loss = neck_loss + head_loss
            return det_reg, det_cls, loss

        else:
            return self.post_process(proposals, det_reg, det_cls, x.shape[-2:])

    def post_process(self, proposals, det_reg, det_cls, img_size):
        detections = []
        classes = []
        for p, reg, cls in zip(proposals, det_reg, det_cls):
            cls = torch.argmax(cls, 1)
            idx = cls != self.output_size
            detection = bbox2proposal(p[idx], reg[idx])
            keep = filter_boxes(detection, img_size, 10)

            detections.append(detection[keep])
            classes.append(cls[idx][keep])

        return detections, classes


class RPN(nn.Module):
    """
    See Also `torchvision.models.detection.rpn`
    """

    def __init__(self, in_ch, n_conv=1, max_bboxes=2000, box_min_len=10,
                 filter_iou=.7, pos_iou=0.7, neg_iou=0.3, max_backprop_sample=256,
                 anchor_config=dict()):
        super().__init__()

        self.filter_iou = filter_iou
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.max_bboxes = max_bboxes
        self.box_min_len = box_min_len
        self.max_backprop_sample = max_backprop_sample
        self.anchor_generator = AnchorGenerator(**anchor_config)
        self.head = RPNHead(in_ch, self.anchor_generator.num_anchors, n_conv)

    def gen_proposals(self, anchors, feature_info, cls, reg, img_size):
        box = bbox2proposal(anchors, reg[:, feature_info[:, 0]])  # (bs, n_boxes, 4)
        keep1 = filter_boxes(box, img_size, self.box_min_len)

        # proposals = torch.zeros((reg.shape[0], self.max_bboxes, 4)).type_as(reg)
        proposals = []
        feature_infos = []

        for i in range(cls.shape[0]):
            k = keep1[i]
            b = box[i][k]
            s = cls[i, feature_info[k, 0], 1].view(-1)

            keep = torchvision.ops.nms(b, s, self.filter_iou)
            keep = keep[:self.max_bboxes]

            # proposals[i, :b.shape[0]] = b[keep]
            proposals.append(b[keep])
            feature_infos.append(feature_info[keep])

        return proposals, feature_infos

    def forward(self, images, features, gt_boxes=None, gt_cls=None):
        anchors, feature_info = self.anchor_generator(images, features)
        cls, reg = self.head(features)
        # note that using `detach` method, 'cause cls and reg do not backprop through
        proposals, feature_infos = self.gen_proposals(anchors, feature_info, cls.detach(), reg.detach(), images[0].shape[-2:])

        loss = None

        if self.training:
            det_cls, det_reg = [], []
            for box, c, r, feature_info in zip(proposals, cls, reg, feature_infos):
                det_cls.append(c[:feature_info.shape[0]])
                det_reg.append(r[:feature_info.shape[0]])

            rpn_cls = []
            for c in gt_cls:
                rpn_cls.append(torch.ones_like(c).type_as(c))  # all object is pos sample

            labels, match_gt_boxes, match_gt_cls = divide_sample(proposals, gt_boxes, rpn_cls, self.neg_iou, self.pos_iou)
            det_boxes = torch.cat(proposals)  # (bs * n_boxes, 4)
            det_reg = torch.cat(det_reg)  # (bs * n_boxes, 4)
            det_cls = torch.cat(det_cls)  # (bs * n_boxes, 2)
            gt_reg = proposal2deltas(det_boxes, match_gt_boxes)

            pos_idx = torch.where(labels == 1)[0]
            neg_idx = torch.where(labels == -1)[0]

            match_gt_cls[neg_idx] = 0

            n_pos = min(len(pos_idx), self.max_backprop_sample)
            n_neg = self.max_backprop_sample - n_pos

            pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[:n_pos]]
            neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[:n_neg]]
            total_idx = torch.cat([pos_idx, neg_idx])

            loss = od_loss(det_reg[pos_idx], gt_reg[pos_idx], det_cls[total_idx], match_gt_cls[total_idx], total_idx.numel())

        return proposals, cls, reg, feature_infos, loss


class RPNHead(nn.Module):
    def __init__(self, in_ch, num_anchors, n_conv):
        super().__init__()

        self.num_anchors = num_anchors
        self.conv_seq = nn.Sequential(*[Conv(in_ch, in_ch, 3, is_bn=False) for _ in range(n_conv)])
        self.cls = nn.Conv2d(in_ch, num_anchors * 2, 1)
        self.reg = nn.Conv2d(in_ch, num_anchors * 4, 1)

    def forward(self, features):
        """

        Args:
            features:

        Returns:
            cls: (batch_size, num_features * num_anchors * feature_map.size, 2)
            reg: (batch_size, num_features * num_anchors * feature_map.size, 4)

        """
        cls, reg = [], []

        for f in features:
            f = self.conv_seq(f)
            cls.append(self.cls(f).unsqueeze(1))
            reg.append(self.reg(f).unsqueeze(1))

        cls = torch.cat(cls, dim=1)  # (b, num_features, 2*num_anchors, h, w)
        cls = cls.permute(0, 1, 3, 4, 2).contiguous().view(cls.shape[0], -1, 2)  # (b, num_features * num_anchors * feature_map.size, 2)

        reg = torch.cat(reg, dim=1)  # (b, num_features, 4*num_anchors, h, w)
        reg = reg.permute(0, 1, 3, 4, 2).contiguous().view(cls.shape[0], -1, 4)  # (b, num_features * num_anchors * feature_map.size, 4)

        return cls, reg


class AnchorGenerator(nn.Module):
    def __init__(
            self,
            sizes=((8 ** 2, 16 ** 2, 24 ** 2),),
            ratios=((0.5, 1, 2),),
            box_min_len=10
    ):
        super().__init__()

        self.window_anchors = nn.ParameterList([self.gen_window_anchors(size, ratio) for size, ratio in zip(sizes, ratios)])

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
        w_ratio = 1 / h_ratio

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
            shifts = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            e = s + shifts.shape[0]
            info = torch.tensor((i, stride_height, stride_width), device=device)
            info = torch.repeat_interleave(info.unsqueeze(0), shifts.shape[0], dim=0)
            info = torch.cat([torch.arange(s, e, device=device).unsqueeze(-1), info], dim=-1)
            s = e

            anchors.append(shifts)
            feature_info.append(info)  # (idx of anchor, idx of feature, *scale_ratio)

        # (num_features * num_anchors * feature_map.size, 4)
        anchors = torch.cat(anchors)
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
        keep = filter_boxes(anchors, image_size, self.box_min_len)
        anchors = anchors[keep]
        feature_info = feature_info[keep]
        anchors = torch.repeat_interleave(anchors.unsqueeze(0), images.shape[0], dim=0)

        return anchors, feature_info


class ROIHead(nn.Module):
    def __init__(self, in_ch, pooling_size, output_size,
                 max_backprop_sample=256, pos_iou=0.5, neg_iou=0.5):
        super().__init__()

        self.pooling_size = pooling_size
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.max_backprop_sample = max_backprop_sample
        self.output_size = output_size

        # self.pool = ROIPooling(pooling_size)

        self.neck = nn.Sequential(
            nn.Flatten(),
            Linear(in_ch * pooling_size * pooling_size, in_ch),
            Linear(in_ch, in_ch),
        )

        self.cls_fc = nn.Linear(in_ch, output_size + 1)  # the last cls is background
        self.reg_fc = nn.Linear(in_ch, 4)

    def gen_detections(self, proposals, features, feature_infos):
        detections = []
        _feature_infos = []
        for j, feature in enumerate(features):
            box = []
            for i in range(len(proposals)):
                feature_info = feature_infos[i]
                proposal = proposals[i]
                feature_idx = torch.where(feature_info[:, 1] == j)[0]
                scale_ratio = feature_info[feature_idx][:, -2:]

                # proposal = proposal[feature_idx]
                proposal[..., :2] /= scale_ratio
                proposal[..., -2:] /= scale_ratio
                proposal = proposal.to(dtype=feature.dtype)

                box.append(proposal)
                _feature_infos.append(torch.zeros((proposal.shape[0],), dtype=torch.int32, device=proposal.device) + i)

            detection = torchvision.ops.roi_pool(feature, box, self.pooling_size)
            detections.append(detection)

        detections = torch.cat(detections)
        _feature_infos = torch.cat(_feature_infos)

        return detections, _feature_infos

    def forward(self, proposals, features, feature_infos, gt_boxes=None, gt_cls=None):
        if self.training:
            _proposals = []
            _feature_infos = []
            _gt_cls = []
            pos_gt_boxes = []
            pos_det_boxes = []
            shift_idx = []
            total = 0
            for p, g, c, f in zip(proposals, gt_boxes, gt_cls, feature_infos):
                # add gt boxes to proposals, make sure that have pos samples
                p = torch.cat([p, g])

                _f = torch.repeat_interleave(f[-1].unsqueeze(0), len(g), dim=0)
                _f[:, 0] = -1
                f = torch.cat([f, _f])

                labels, match_gt_boxes, match_gt_cls = divide_sample([p], [g], [c], self.neg_iou, self.pos_iou)

                pos_idx = torch.where(labels == 1)[0]
                neg_idx = torch.where(labels == -1)[0]

                match_gt_cls[neg_idx] = self.output_size

                n_pos = min(len(pos_idx), self.max_backprop_sample)
                n_neg = self.max_backprop_sample - n_pos

                pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[:n_pos]]
                neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=neg_idx.device)[:n_neg]]
                total_idx = torch.cat([pos_idx, neg_idx])

                _proposals.append(p[total_idx])
                _feature_infos.append(f[total_idx])
                _gt_cls.append(match_gt_cls[total_idx])
                pos_gt_boxes.append(match_gt_boxes[pos_idx])
                pos_det_boxes.append(p[pos_idx])
                shift_idx.append(list(range(total, len(pos_idx) + total)))
                total += len(total_idx)

            proposals = _proposals
            feature_infos = _feature_infos
            gt_cls = _gt_cls

        detections, _feature_infos = self.gen_detections(proposals, features, feature_infos)
        detections = self.neck(detections)
        cls = self.cls_fc(detections)
        reg = self.reg_fc(detections)

        loss = None
        if self.training:
            pos_det_reg = torch.cat([reg[idx] for idx in shift_idx])
            pos_gt_boxes = torch.cat(pos_gt_boxes)
            pos_det_boxes = torch.cat(pos_det_boxes)
            gt_cls = torch.cat(gt_cls)

            pos_gt_reg = proposal2deltas(pos_det_boxes, pos_gt_boxes)

            loss = od_loss(pos_det_reg, pos_gt_reg, cls, gt_cls, len(cls))

        return reg, cls, _feature_infos, loss


class ROIPooling(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        y = []
        for bx in x:
            bx = bx.expand_dim(0)
            y.append(F.adaptive_avg_pool2d(bx, self.output_size))

        y = torch.cat(y)

        return y


def filter_boxes(boxes, image_size, min_size):
    """clip"""
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = image_size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=boxes.dim())
    boxes = clipped_boxes.reshape(boxes.shape)

    ws, hs = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
    keep = (ws >= min_size) & (hs >= min_size)

    return keep


def bbox2proposal(boxes, deltas):
    widths = boxes[..., 2] - boxes[..., 0] + 1.0
    heights = boxes[..., 3] - boxes[..., 1] + 1.0
    ctr_x = boxes[..., 0] + 0.5 * widths
    ctr_y = boxes[..., 1] + 0.5 * heights

    dx = deltas[..., 0]
    dy = deltas[..., 1]
    dw = deltas[..., 2]
    dh = deltas[..., 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    proposal = boxes.clone()
    proposal[..., 0] = pred_ctr_x - 0.5 * pred_w
    proposal[..., 1] = pred_ctr_y - 0.5 * pred_h
    proposal[..., 2] = pred_ctr_x + 0.5 * pred_w
    proposal[..., 3] = pred_ctr_y + 0.5 * pred_h

    proposal = proposal.clamp(min=0)

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


def divide_sample(det_boxes, gt_boxes, gt_cls, neg_iou, pos_iou):
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

        labels.append(label)
        match_gt_boxes.append(gt[gt_idx])
        match_gt_cls.append(cls[gt_idx])

    labels = torch.cat(labels)
    match_gt_boxes = torch.cat(match_gt_boxes)
    match_gt_cls = torch.cat(match_gt_cls)

    return labels, match_gt_boxes, match_gt_cls


def od_loss(pos_det_reg, pos_gt_reg, det_cls, gt_cls, n_sample):
    reg_loss = F.smooth_l1_loss(
        pos_det_reg,
        pos_gt_reg,
        beta=1 / 9,
        reduction="sum",
    ) / n_sample

    cls_loss = F.cross_entropy(det_cls, gt_cls)

    return cls_loss + reg_loss


Model = FasterRCNN
