import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from image_classifier.ResNet import ResNet
from utils.layers import Conv, Linear, ConvInModule, OutModule


class FasterRCNN(nn.Module):
    """
    See Also `torchvision.models.detection.faster_rcnn`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, backbone=None,
            rpn_config=dict(),
            **kwargs
    ):
        super().__init__()

        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=input_size)

        if backbone is None:
            backbone = ResNet(
                in_module=nn.Module(),
                out_module=nn.Module(),
                conv_config=((3, 64, 2), (4, 128, 2), (6, 256, 2))
            ).conv_seq

        self.input = in_module
        self.backbone = backbone
        self.neck = RPN(256, 256, **rpn_config)
        self.head = ROIHead(256, 7, output_size)

    def forward(self, x, gt_boxes=None, gt_cls=None):
        x = self.input(x)
        features = self.backbone(x)

        if isinstance(features, torch.Tensor):
            features = [features]

        proposals, scores, feature_infos = self.neck(x, features)
        detections, det_cls, _feature_infos = self.head(proposals, features, feature_infos)

        if self.training:
            img_size = x.shape[-1]

            real_det_boxes, real_det_cls = [], []
            for box, cls, feature_info in zip(proposals, scores, feature_infos):
                real_det_boxes.append(box[:feature_info.shape[0]] / img_size)
                real_det_cls.append(cls[:feature_info.shape[0]])

            tmp_cls = []
            for cls in gt_cls:
                tmp_cls.append(torch.ones_like(cls).type_as(cls))

            neck_loss = self.neck.loss(real_det_boxes, real_det_cls, gt_boxes, tmp_cls)

            real_det_boxes, real_det_cls = [], []
            for i in range(x.shape[0]):
                feature_idx = torch.where(_feature_infos == i)
                real_det_boxes.append(detections[feature_idx])
                real_det_cls.append(det_cls[feature_idx])

            head_loss = self.neck.loss(real_det_boxes, real_det_cls, gt_boxes, gt_cls)
            loss = neck_loss + head_loss

            return detections, det_cls, loss

        else:
            return detections, det_cls


class RPN(nn.Module):
    """
    See Also `torchvision.models.detection.rpn`
    """

    def __init__(self, in_ch, n_conv, max_anchors=3000,
                 filter_iou=.7, pos_iou=0.7, neg_iou=0.3,
                 anchor_config=dict()):
        super().__init__()

        self.filter_iou = filter_iou
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.max_anchors = max_anchors
        self.anchor_generator = AnchorGenerator(**anchor_config)
        self.head = RPNHead(in_ch, self.anchor_generator.num_anchors, n_conv)

    def gen_proposals(self, anchors, feature_info, cls, reg):
        feature_idx = feature_info[:, 0]
        deltas = reg[:, feature_idx]
        box = self.bbox_transform_inv(anchors, deltas)
        score = cls[:, feature_idx, 1].squeeze(-1)

        proposals = torch.zeros((reg.shape[0], self.max_anchors, 4)).type_as(reg)
        feature_infos = []

        for i in range(cls.shape[0]):
            b = box[i]
            s = score[i]

            keep = torchvision.ops.nms(b, s, self.filter_iou)
            keep = keep[:self.max_anchors]

            b = b[keep]

            proposals[i, :b.shape[0]] = b
            feature_infos.append(feature_info[keep])

        return proposals, feature_infos

    @staticmethod
    def bbox_transform_inv(boxes, deltas):
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

        pred_boxes = boxes.clone()
        pred_boxes[..., 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[..., 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[..., 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[..., 3] = pred_ctr_y + 0.5 * pred_h

        pred_boxes = pred_boxes.clamp(min=0)

        return pred_boxes

    def forward(self, images, features):
        anchors, feature_info = self.anchor_generator(images, features)
        cls, reg = self.head(features)
        proposals, feature_infos = self.gen_proposals(anchors, feature_info, cls, reg)

        return proposals, cls, feature_infos

    def divide_sample(self, det_boxes, gt_boxes, gt_cls):
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

                label[iou < self.neg_iou] = -1
                label[iou > self.pos_iou] = 1

            labels.append(label)
            match_gt_boxes.append(gt[gt_idx])
            match_gt_cls.append(cls[gt_idx])

        labels = torch.cat(labels)
        match_gt_boxes = torch.cat(match_gt_boxes)
        match_gt_cls = torch.cat(match_gt_cls)

        return labels, match_gt_boxes, match_gt_cls

    def loss(self, det_boxes, det_cls, gt_boxes, gt_cls):
        labels, match_gt_boxes, match_gt_cls = self.divide_sample(det_boxes, gt_boxes, gt_cls)

        det_boxes = torch.cat(det_boxes)
        det_cls = torch.cat(det_cls)

        pos_idx = torch.where(labels == 1)[0]
        neg_idx = torch.where(labels == -1)[0]
        total_idx = torch.cat([pos_idx, neg_idx])

        reg_loss = F.smooth_l1_loss(
            det_boxes[pos_idx],
            match_gt_boxes[pos_idx],
            beta=1 / 9,
            reduction="sum",
        ) / total_idx.numel()

        cls_loss = F.cross_entropy(det_cls[total_idx], match_gt_cls[total_idx])

        return cls_loss + reg_loss


class RPNHead(nn.Module):
    def __init__(self, in_ch, num_anchors, n_conv):
        super().__init__()

        self.num_anchors = num_anchors

        layers = [Conv(in_ch, in_ch, 3, is_bn=False) for _ in range(n_conv)]
        self.conv_seq = nn.Sequential(*layers)

        self.cls = nn.Conv2d(in_ch, num_anchors * 2, 1)
        self.reg = nn.Conv2d(in_ch, num_anchors * 4, 1)

    def forward(self, features):
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
            min_size=50
    ):
        super().__init__()

        self.window_anchors = nn.ParameterList([self.gen_window_anchors(size, ratio) for size, ratio in zip(sizes, ratios)])

        self.min_size = min_size
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
            feature_info.append(info)   # (idx of anchor, idx of feature, *scale_ratio)

        # (num_features * num_anchors * feature_map.size, 4)
        anchors = torch.cat(anchors)
        feature_info = torch.cat(feature_info)

        return anchors, feature_info

    def filter_anchors(self, boxes, image_size, feature_info):
        """clip"""
        boxes_x = boxes[..., 0::2]
        boxes_y = boxes[..., 1::2]
        height, width = image_size

        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)

        clipped_boxes = torch.stack((boxes_x, boxes_y), dim=boxes.dim())
        boxes = clipped_boxes.reshape(boxes.shape)

        ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
        keep = torch.where((ws >= self.min_size) & (hs >= self.min_size))[0]

        return boxes[keep], feature_info[keep]

    def forward(self, images, features):
        # per pixel of feature map per grid
        grid_sizes = [torch.tensor(feature_map.shape[-2:], device=images.device) for feature_map in features]
        image_size = torch.tensor(images.shape[-2:], device=images.device)

        anchors, feature_info = self.gen_anchors(grid_sizes, image_size)
        anchors, feature_info = self.filter_anchors(anchors, image_size, feature_info)
        anchors = torch.repeat_interleave(anchors.unsqueeze(0), images.shape[0], dim=0)

        return anchors, feature_info


class ROIHead(nn.Module):
    def __init__(self, in_ch, pooling_size, output_size):
        super().__init__()

        self.pooling_size = pooling_size

        # self.pool = ROIPooling(pooling_size)

        self.neck = nn.Sequential(
            nn.Flatten(),
            Linear(in_ch * pooling_size * pooling_size, in_ch),
            Linear(in_ch, in_ch),
        )

        self.cls_fc = nn.Linear(in_ch, output_size)
        self.reg_fc = nn.Linear(in_ch, 4)

    def gen_detections(self, proposals, features, feature_infos):
        detections = []
        _feature_infos = []
        for j, feature in enumerate(features):
            box = []
            for i in range(proposals.shape[0]):
                feature_info = feature_infos[i]
                proposal = proposals[i]
                feature_idx = torch.where(feature_info[:, 1] == j)[0]
                scale_ratio = feature_info[feature_idx][:, -2:]

                proposal = proposal[feature_idx]
                proposal[..., :2] /= scale_ratio
                proposal[..., -2:] /= scale_ratio
                proposal = proposal.to(dtype=feature.dtype)

                box.append(proposal)
                _feature_infos.append(torch.zeros((proposal.shape[0], ), dtype=torch.int32, device=proposal.device) + i)

            detection = torchvision.ops.roi_pool(feature, box, self.pooling_size)

            detections.append(detection)

        detections = torch.cat(detections)
        _feature_infos = torch.cat(_feature_infos)

        return detections, _feature_infos

    def forward(self, proposals, features, feature_infos):
        detections, _feature_infos = self.gen_detections(proposals, features, feature_infos)
        detections = self.neck(detections)
        cls = self.cls_fc(detections)
        reg = self.reg_fc(detections)

        return reg, cls, _feature_infos


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


Model = FasterRCNN
