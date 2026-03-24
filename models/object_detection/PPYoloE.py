from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import cls_nms


class Model(nn.Module):
    """convert paddle to torch, only refactor a little code
    References
        https://github.com/PaddlePaddle/PaddleYOLO

    """
    conf_thres = 0.5
    nms_thres = 0.6

    def __init__(self, n_classes=80):
        super().__init__()
        self.backbone = CSPResNet()
        self.neck = CustomCSPPAN()
        self.yolo_head = PPYOLOEHead(num_classes=n_classes)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.fit(*args, **kwargs)
        else:
            return self.inference(*args, **kwargs)

    def fit(self, x, gt_boxes=None, gt_cls=None):
        neck_feats = self.process(x)
        losses = self.yolo_head.loss(neck_feats, gt_boxes, gt_cls)
        return losses

    def inference(self, x):
        neck_feats = self.process(x)
        yolo_head_outs = self.yolo_head(neck_feats)
        post_outs = self.post_process(yolo_head_outs)

        return post_outs

    def process(self, x):
        body_feats = self.backbone(x)
        neck_feats = self.neck(body_feats)
        return neck_feats

    def post_process(self, head_outs):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor
        pred_scores = pred_scores.permute([0, 2, 1])  # (b, n, n_classes), no confidence
        result = []
        for bboxes, scores in zip(pred_bboxes, pred_scores):
            keep, classes = (scores > self.conf_thres).nonzero(as_tuple=True)
            bboxes, scores = bboxes[keep], scores[keep, classes]
            if bboxes.numel():
                keep = cls_nms(bboxes, scores, classes, iou_threshold=self.nms_thres)
                bboxes, classes, scores = bboxes[keep], classes[keep], scores[keep]
            result.append({
                "bboxes": bboxes,
                "confs": scores,
                "classes": classes.to(dtype=torch.int),
            })
        return result


class Model4Export(Model):
    """for exporting to onnx, torchscript, etc."""

    def inference(self, x, **kwargs):
        x = self.pre_process(x)
        neck_feats = self.process(x)
        yolo_head_outs = self.yolo_head(neck_feats)
        preds = self.post_process(yolo_head_outs)
        return preds

    def pre_process(self, x):
        """for faster infer, use uint8 input and bfp16 to process"""
        x = x.to(dtype=torch.bfloat16)
        x = x / 255
        return x

    def post_process(self, head_outs):
        """for faster infer, only output 500 bboxes"""
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor
        pred_scores = pred_scores.permute([0, 2, 1])
        conf, _ = pred_scores.max(-1)
        _, indices = torch.sort(conf, dim=-1, descending=True)
        # keep the output performance same as yolov5
        preds = torch.cat([
            pred_bboxes,
            torch.ones_like(conf).unsqueeze(-1),  # no conf
            pred_scores
        ], dim=-1)
        indices = indices[:, :500].unsqueeze(-1).expand(-1, -1, preds.shape[-1])
        preds = preds.gather(1, indices)
        preds = preds.to(dtype=torch.float16)

        return preds


class CSPResNet(nn.Module):
    def __init__(
            self,
            layers=[3, 6, 6, 3],
            channels=[64, 128, 256, 512, 1024],
            return_idx=[1, 2, 3],
            width_mult=1.0,
            depth_mult=1.0,
            use_alpha=False,
            **kwargs
    ):
        super().__init__()
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]
        act = nn.SiLU()

        mid_ch = channels[0] // 2
        self.stem = nn.Sequential(
            OrderedDict({
                'conv1': ConvBNLayer(3, mid_ch, 3, stride=2, padding=1, act=act),
                'conv2': ConvBNLayer(mid_ch, mid_ch, 3, stride=1, padding=1, act=act),
                'conv3': ConvBNLayer(mid_ch, channels[0], 3, stride=1, padding=1, act=act)
            })
        )

        n = len(channels) - 1
        stages = OrderedDict()
        for i in range(n):
            stages[str(i)] = CSPResStage(
                BasicBlock,
                channels[i],
                channels[i + 1],
                layers[i],
                2,
                act=act,
                use_alpha=use_alpha
            )
        self.stages = nn.Sequential(stages)
        self.return_idx = return_idx

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs


class ConvBNLayer(nn.Module):
    def __init__(
            self,
            ch_in,
            ch_out,
            filter_size=3,
            stride=1,
            groups=1,
            padding=0,
            act=None
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )

        self.bn = nn.BatchNorm2d(ch_out)
        self.act = act or nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class CSPResStage(nn.Module):
    def __init__(
            self,
            block_fn,
            ch_in,
            ch_out,
            n,
            stride,
            act=None,
            use_alpha=False
    ):
        super().__init__()

        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(
                ch_in, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2,
                ch_mid // 2,
                act=act,
                shortcut=True,
                use_alpha=use_alpha
            ) for i in range(n)
        ])
        self.attn = EffectiveSELayer(ch_mid)
        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], dim=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act=None, alpha=False):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(ch_in, ch_out, 1, stride=1, padding=0, act=None)
        self.act = act
        if alpha:
            self.alpha = nn.Parameter(torch.ones((1,)))
        else:
            self.alpha = 1

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.alpha * self.conv2(x)
        y = self.act(y)
        return y


class CustomCSPPAN(nn.Module):
    def __init__(
            self,
            in_channels=[256, 512, 1024],
            out_channels=[768, 384, 192],
            stage_num=1,
            block_num=3,
            drop_block=False,
            block_size=3,
            keep_prob=0.9,
            spp=True,
            width_mult=1.0,
            depth_mult=1.0,
            use_alpha=False,
            use_trans=False,
            eval_size=[640, 640]
    ):

        super().__init__()
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        act = nn.SiLU()
        self.num_blocks = len(in_channels)

        self.hidden_dim = in_channels[-1]
        in_channels = in_channels[::-1]

        self.use_trans = use_trans
        self.eval_size = eval_size

        fpn_stages = []
        fpn_routes = []
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2

            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_module(
                    str(j),
                    CSPStage(
                        ch_in if j == 0 else ch_out,
                        ch_out,
                        block_num,
                        act=act,
                        spp=(spp and i == 0),
                        use_alpha=use_alpha
                    )
                )

            if drop_block:
                stage.add_module('drop', DropBlock(block_size, keep_prob))

            fpn_stages.append(stage)

            if i < self.num_blocks - 1:
                fpn_routes.append(
                    ConvBNLayer(
                        ch_in=ch_out,
                        ch_out=ch_out // 2,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        act=act))

            ch_pre = ch_out

        self.fpn_stages = nn.ModuleList(fpn_stages)
        self.fpn_routes = nn.ModuleList(fpn_routes)

        pan_stages = []
        pan_routes = []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(
                ConvBNLayer(
                    ch_in=out_channels[i + 1],
                    ch_out=out_channels[i + 1],
                    filter_size=3,
                    stride=2,
                    padding=1,
                    act=act
                )
            )

            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_module(
                    str(j),
                    CSPStage(
                        ch_in if j == 0 else ch_out,
                        ch_out,
                        block_num,
                        act=act,
                        spp=False,
                        use_alpha=use_alpha
                    )
                )
            if drop_block:
                stage.add_module('drop', DropBlock(block_size, keep_prob))

            pan_stages.append(stage)

        self.pan_stages = nn.ModuleList(pan_stages[::-1])
        self.pan_routes = nn.ModuleList(pan_routes[::-1])

    def build_2d_sincos_position_embedding(
            self,
            w,
            h,
            embed_dim=1024,
            temperature=10000.,
    ):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        pos_emb = torch.cat(
            [
                torch.sin(out_w), torch.cos(out_w), torch.sin(out_h),
                torch.cos(out_h)
            ],
            dim=1)[None, :, :]

        return pos_emb

    def forward(self, blocks):
        if self.use_trans:
            last_feat = blocks[-1]
            n, c, h, w = last_feat.shape

            # flatten [B, C, H, W] to [B, HxW, C]
            src_flatten = last_feat.flatten(2).permute(0, 2, 1)
            if self.eval_size is not None and not self.training:
                pos_embed = self.pos_embed
            else:
                pos_embed = self.build_2d_sincos_position_embedding(w=w, h=h, embed_dim=self.hidden_dim)

            memory = self.encoder(src_flatten, pos_embed=pos_embed)
            last_feat_encode = memory.permute(0, 2, 1).reshape([n, c, h, w])
            blocks[-1] = last_feat_encode

        blocks = blocks[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.cat([route, block], dim=1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(route, scale_factor=2.)

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = torch.cat([route, block], dim=1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)

        return pan_feats[::-1]


class BasicBlock(nn.Module):
    def __init__(
            self,
            ch_in,
            ch_out,
            act=None,
            shortcut=True,
            use_alpha=False
    ):
        super().__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act, alpha=use_alpha)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return torch.add(x, y)
        else:
            return y


class CSPStage(nn.Module):
    def __init__(
            self,
            ch_in,
            ch_out,
            n,
            act=None,
            spp=False,
            use_alpha=False
    ):
        super().__init__()

        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            self.convs.add_module(
                str(i),
                BasicBlock(
                    next_ch_in,
                    ch_mid,
                    act=act,
                    shortcut=False,
                    use_alpha=use_alpha
                )
            )
            if i == (n - 1) // 2 and spp:
                self.convs.add_module('spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = torch.cat([y1, y2], dim=1)
        y = self.conv3(y)
        return y


class SPP(nn.Module):
    def __init__(
            self,
            ch_in,
            ch_out,
            k,
            pool_size,
            act=None,
    ):
        super().__init__()
        self.pool = nn.ModuleList()
        for i, size in enumerate(pool_size):
            self.pool.add_module(
                'pool{}'.format(i),
                nn.MaxPool2d(
                    kernel_size=size,
                    stride=1,
                    padding=size // 2,
                )
            )
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, dim=1)

        y = self.conv(y)
        return y


class DropBlock(nn.Module):
    def __init__(self, block_size, keep_prob, name=None):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (float): keep probability
            name (str): layer name
        """
        super().__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size ** 2)
            shape = x.shape[2:]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = (torch.rand(x.shape) < gamma).to(x.dtype)
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2,
            )
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y


class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act=None):
        super().__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

    def forward(self, feat, avg_feat):
        weight = F.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


class PPYOLOEHead(nn.Module):
    def __init__(
            self,
            in_channels=[768, 384, 192],
            num_classes=80,
            fpn_strides=(32, 16, 8),
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            reg_max=16,
            reg_range=None,
            use_shared_conv=True,
            eval_size=[640, 640],
            loss_weight={
                'class': 1.0,
                'iou': 2.5,
                'dfl': 0.5,
            },
            use_varifocal_loss=True,
            static_assigner_epoch=4,
    ):
        super().__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        if reg_range:
            self.sm_use = True
            self.reg_range = reg_range
        else:
            self.sm_use = False
            self.reg_range = (0, reg_max + 1)
        self.reg_channels = self.reg_range[1] - self.reg_range[0]
        self.use_shared_conv = use_shared_conv

        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        act = nn.SiLU()
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act))
            self.stem_reg.append(ESEAttn(in_c, act=act))
        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(nn.Conv2d(in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(nn.Conv2d(in_c, 4 * self.reg_channels, 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2d(self.reg_channels, 1, 1, bias=False)

        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_size = eval_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = ATSSAssigner()
        self.assigner = TaskAlignedAssigner_CR()

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w).to(feats[i]) + self.grid_cell_offset
            shift_y = torch.arange(end=h).to(feats[i]) + self.grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], dim=-1)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full((h * w, 1), stride).to(feats[i]))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    def loss(self, feats, gt_boxes, gt_cls):
        anchors, anchor_points, num_anchors_list, stride_tensor = self.generate_anchors_for_grid_cell(
            feats, self.fpn_strides, self.grid_cell_scale, self.grid_cell_offset
        )

        anchors = anchors.to(feats[0].device)
        anchor_points = anchor_points.to(feats[0].device)
        stride_tensor = stride_tensor.to(feats[0].device)
        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).permute([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).permute([0, 2, 1]))
        cls_score_list = torch.concat(cls_score_list, dim=1)
        reg_distri_list = torch.concat(reg_distri_list, dim=1)

        return self.get_loss(
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor,
            gt_boxes, gt_cls
        )

    def generate_anchors_for_grid_cell(
            self, feats,
            fpn_strides,
            grid_cell_size=5.0,
            grid_cell_offset=0.5,
    ):
        r"""
        Like ATSS, generate anchors based on grid size.
        Args:
            feats (List[Tensor]): shape[s, (b, c, h, w)]
            fpn_strides (tuple|list): shape[s], stride for each scale feature
            grid_cell_size (float): anchor size
            grid_cell_offset (float): The range is between 0 and 1.
        Returns:
            anchors (Tensor): shape[l, 4], "xmin, ymin, xmax, ymax" format.
            anchor_points (Tensor): shape[l, 2], "x, y" format.
            num_anchors_list (List[int]): shape[s], contains [s_1, s_2, ...].
            stride_tensor (Tensor): shape[l, 1], contains the stride for each scale.
        """
        assert len(feats) == len(fpn_strides)
        dtype = torch.float32
        anchors = []
        anchor_points = []
        num_anchors_list = []
        stride_tensor = []
        for feat, stride in zip(feats, fpn_strides):
            _, _, h, w = feat.shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (torch.arange(end=w) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor = torch.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                dim=-1).to(dtype)
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype)

            anchors.append(anchor.reshape([-1, 4]))
            anchor_points.append(anchor_point.reshape([-1, 2]))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=dtype))
        anchors = torch.concat(anchors)
        anchors.requires_grad = False
        anchor_points = torch.concat(anchor_points)
        anchor_points.requires_grad = False
        stride_tensor = torch.concat(stride_tensor)
        stride_tensor.requires_grad = False
        return anchors, anchor_points, num_anchors_list, stride_tensor

    def get_loss(self, pred_scores, pred_distri, anchors, anchor_points, num_anchors_list, stride_tensor, gt_boxes, gt_cls):
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels, gt_bboxes, _, pad_gt_mask = self.pad_gt(gt_cls, gt_boxes)

        alpha_l = -1

        if self.sm_use:
            # only used in smalldet of PPYOLOE-SOD model
            assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                stride_tensor,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes
            )
        else:
            if not hasattr(self, "assigned_labels"):
                assigned_labels, assigned_bboxes, assigned_scores = self.assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes
                )

            else:
                # only used in distill
                assigned_labels = self.assigned_labels
                assigned_bboxes = self.assigned_bboxes
                assigned_scores = self.assigned_scores

        # rescale bbox
        assigned_bboxes = assigned_bboxes / stride_tensor

        assign_out_dict = self.get_loss_from_assign(
            pred_scores, pred_distri, pred_bboxes, anchor_points_s,
            assigned_labels, assigned_bboxes, assigned_scores, alpha_l
        )

        loss = assign_out_dict

        return loss

    def pad_gt(self, gt_labels, gt_bboxes, gt_scores=None):
        r""" Pad 0 in gt_labels and gt_bboxes.
        Args:
            gt_labels (Tensor|List[Tensor], int64): Label of gt_bboxes,
                shape is [B, n, 1] or [[n_1, 1], [n_2, 1], ...], here n = sum(n_i)
            gt_bboxes (Tensor|List[Tensor], float32): Ground truth bboxes,
                shape is [B, n, 4] or [[n_1, 4], [n_2, 4], ...], here n = sum(n_i)
            gt_scores (Tensor|List[Tensor]|None, float32): Score of gt_bboxes,
                shape is [B, n, 1] or [[n_1, 4], [n_2, 4], ...], here n = sum(n_i)
        Returns:
            pad_gt_labels (Tensor, int64): shape[B, n, 1]
            pad_gt_bboxes (Tensor, float32): shape[B, n, 4]
            pad_gt_scores (Tensor, float32): shape[B, n, 1]
            pad_gt_mask (Tensor, float32): shape[B, n, 1], 1 means bbox, 0 means no bbox
        """
        num_max_boxes = max([len(a) for a in gt_bboxes])
        batch_size = len(gt_bboxes)
        # pad label and bbox
        pad_gt_labels = torch.zeros([batch_size, num_max_boxes, 1]).to(gt_labels[0])
        pad_gt_bboxes = torch.zeros([batch_size, num_max_boxes, 4]).to(gt_bboxes[0])
        pad_gt_scores = torch.zeros([batch_size, num_max_boxes, 1]).to(gt_bboxes[0])
        pad_gt_mask = torch.zeros([batch_size, num_max_boxes, 1]).to(gt_bboxes[0])
        for i, (label, bbox) in enumerate(zip(gt_labels, gt_bboxes)):
            if len(label) > 0 and len(bbox) > 0:
                pad_gt_labels[i, :len(label), 0] = label
                pad_gt_bboxes[i, :len(bbox)] = bbox
                pad_gt_mask[i, :len(bbox)] = 1.
                if gt_scores is not None:
                    pad_gt_scores[i, :len(gt_scores[i])] = gt_scores[i]
        if gt_scores is None:
            pad_gt_scores = pad_gt_mask.clone()
        return pad_gt_labels, pad_gt_bboxes, pad_gt_scores, pad_gt_mask

    def get_loss_from_assign(
            self, pred_scores, pred_distri, pred_bboxes,
            anchor_points_s, assigned_labels, assigned_bboxes,
            assigned_scores, alpha_l
    ):
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels, self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores, one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        assigned_scores_sum = assigned_scores.sum()
        assigned_scores_sum = torch.clip(assigned_scores_sum, min=1.)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = self._bbox_loss(
            pred_distri, pred_bboxes, anchor_points_s,
            assigned_labels, assigned_bboxes, assigned_scores,
            assigned_scores_sum
        )
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        out_dict = {
            'loss': loss,
            'loss.cls': loss_cls,
            'loss.iou': loss_iou,
            'loss.dfl': loss_dfl,
            'loss.l1': loss_l1,
        }
        return out_dict

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        # note, unsafe to autocast
        with torch.amp.autocast(score.device.type, enabled=False):
            score = score.float()
            label = label.float()
            loss = F.binary_cross_entropy(score, label, weight=weight, reduction='sum')
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        with torch.no_grad():
            weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        # note, unsafe to autocast
        with torch.amp.autocast(pred_score.device.type, enabled=False):
            pred_score = pred_score.float()
            gt_score = gt_score.float()
            loss = F.binary_cross_entropy(pred_score, gt_score, weight=weight, reduction='sum')
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        def get_static_shape(tensor):
            shape = tensor.shape
            return shape

        _, l, _ = get_static_shape(pred_dist)
        pred_dist = F.softmax(pred_dist.reshape([-1, l, 4, self.reg_channels]))
        pred_dist = self.proj_conv(pred_dist.permute([0, 3, 1, 2])).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox_decode_fake(self, pred_dist):
        def get_static_shape(tensor):
            shape = tensor.shape
            return shape

        _, l, _ = get_static_shape(pred_dist)
        pred_dist_dfl = F.softmax(pred_dist.reshape([-1, l, 4, self.reg_channels]))
        pred_dist = self.proj_conv(pred_dist_dfl.permute([0, 3, 1, 2])).squeeze(1)
        return pred_dist, pred_dist_dfl

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.concat([lt, rb], -1).clip(self.reg_range[0], self.reg_range[1] - 1 - 0.01)

    def _df_loss(self, pred_dist, target, lower_bound=0):
        target_left = target.floor().to(torch.int64)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float32) - target
        weight_right = 1 - weight_left
        pred_dist = pred_dist.reshape(-1, pred_dist.shape[-1])
        target_left = (target_left - lower_bound).flatten()
        target_right = (target_right - lower_bound).flatten()
        weight_left = weight_left.flatten()
        weight_right = weight_right.flatten()
        loss_left = F.cross_entropy(
            pred_dist,
            target_left,
            reduction='none'
        ) * weight_left
        loss_right = F.cross_entropy(
            pred_dist,
            target_right,
            reduction='none'
        ) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)

        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.to(torch.int32).unsqueeze(-1).tile([1, 1, 4]).to(torch.bool)
            pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos, assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).to(torch.int32).tile([1, 1, self.reg_channels * 4]).to(torch.bool)
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).reshape([-1, 4, self.reg_channels])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = torch.masked_select(assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos, assigned_ltrb_pos, self.reg_range[0]) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = torch.zeros([1], device=pred_dist.device)
            loss_iou = torch.zeros([1], device=pred_dist.device)
            loss_dfl = pred_dist.sum() * 0.
        return loss_l1, loss_iou, loss_dfl

    def forward(self, feats):
        anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            _, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape([-1, 4, self.reg_channels, l]).permute([0, 2, 3, 1])
            if self.use_shared_conv:
                reg_dist = self.proj_conv(F.softmax(reg_dist, dim=1)).squeeze(1)
            else:
                reg_dist = F.softmax(reg_dist, dim=1)
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([-1, self.num_classes, l]))
            reg_dist_list.append(reg_dist)

        cls_score_list = torch.cat(cls_score_list, dim=-1)
        if self.use_shared_conv:
            reg_dist_list = torch.cat(reg_dist_list, dim=1)
        else:
            reg_dist_list = torch.cat(reg_dist_list, dim=2)
            reg_dist_list = self.proj_conv(reg_dist_list).squeeze(1)

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor


class ATSSAssigner(nn.Module):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection
     via Adaptive Training Sample Selection
    """

    def __init__(
            self,
            topk=9,
            num_classes=80,
            force_gt_matching=False,
            eps=1e-9,
            sm_use=False
    ):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps
        self.sm_use = sm_use

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list, pad_gt_mask):
        gt2anchor_distances_list = torch.split(gt2anchor_distances, num_anchors_list, dim=-1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(gt2anchor_distances_list, num_anchors_index):
            num_anchors = distances.shape[-1]
            _, topk_idxs = torch.topk(distances, self.topk, dim=-1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(dim=-2).to(gt2anchor_distances.dtype)
            is_in_topk_list.append(is_in_topk * pad_gt_mask)
        is_in_topk_list = torch.concat(is_in_topk_list, dim=-1)
        topk_idxs_list = torch.concat(topk_idxs_list, dim=-1)
        return is_in_topk_list, topk_idxs_list

    def forward(
            self,
            anchor_bboxes,
            num_anchors_list,
            gt_labels,
            gt_bboxes,
            pad_gt_mask,
            bg_index,
            gt_scores=None,
            pred_bboxes=None
    ):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            anchor_bboxes (Tensor, float32): pre-defined anchors, shape(L, 4),
                    "xmin, xmax, ymin, ymax" format
            num_anchors_list (List): num of anchors in each level
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes,
                    shape(B, n, 1), if None, then it will initialize with one_hot label
            pred_bboxes (Tensor, float32, optional): predicted bounding boxes, shape(B, L, 4)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C), if pred_bboxes is not None, then output ious
        """
        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3

        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index, dtype=torch.int32)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros([batch_size, num_anchors, self.num_classes])
            return assigned_labels, assigned_bboxes, assigned_scores

        # 1. compute iou between gt and anchor bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes.reshape([-1, 4]), anchor_bboxes)
        ious = ious.reshape([batch_size, -1, num_anchors])

        # 2. compute center distance between all anchors and gt, [B, n, L]
        gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)
        anchor_centers = bbox_center(anchor_bboxes)
        gt2anchor_distances = (gt_centers - anchor_centers.unsqueeze(0)).norm(2, dim=-1).reshape([batch_size, -1, num_anchors])

        # 3. on each pyramid level, selecting topk closest candidates
        # based on the center distance, [B, n, L]
        is_in_topk, topk_idxs = self._gather_topk_pyramid(gt2anchor_distances, num_anchors_list, pad_gt_mask)

        # 4. get corresponding iou for the these candidates, and compute the
        # mean and std, 5. set mean + std as the iou threshold
        iou_candidates = ious * is_in_topk
        iou_threshold = torch.index_sample(
            iou_candidates.flatten(end_dim=-2),
            topk_idxs.flatten(end_dim=-2)
        )
        iou_threshold = iou_threshold.reshape([batch_size, num_max_boxes, -1])
        iou_threshold = iou_threshold.mean(dim=-1, keepdim=True) + iou_threshold.std(dim=-1, keepdim=True)
        is_in_topk = torch.where(iou_candidates > iou_threshold, is_in_topk, torch.zeros_like(is_in_topk))

        # 6. check the positive sample's center in gt, [B, n, L]
        if self.sm_use:
            is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes, sm_use=True)
        else:
            is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # 7. if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).to(torch.int32).tile([1, num_max_boxes, 1]).to(torch.bool)
            if self.sm_use:
                is_max_iou = compute_max_iou_anchor(ious * mask_positive)
            else:
                is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        # 8. make sure every gt_bbox matches the anchor
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).tile([1, num_max_boxes, 1])
            mask_positive = torch.where(mask_max_iou, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        assigned_gt_index = mask_positive.argmax(dim=-2)

        # assigned target
        batch_ind = torch.arange(end=batch_size, dtype=gt_labels.dtype, device=gt_labels.device).unsqueeze(-1)
        assigned_gt_index = assigned_gt_index + (batch_ind * num_max_boxes).to(torch.int64)
        assigned_labels = torch.gather(gt_labels.flatten(), 0, assigned_gt_index.flatten())
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(
            mask_positive_sum > 0, assigned_labels,
            torch.full_like(assigned_labels, bg_index)
        )

        assigned_bboxes = torch.gather(gt_bboxes.reshape([-1, 4]), 0, assigned_gt_index.flatten())
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_scores = F.one_hot(assigned_labels, self.num_classes + 1)
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(assigned_scores, -1, torch.tensor(ind))
        if pred_bboxes is not None:
            # assigned iou
            ious = batch_iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            ious, _ = ious.max(dim=-2)
            ious = ious.unsqueeze(-1)
            assigned_scores *= ious
        elif gt_scores is not None:
            gather_scores = torch.gather(gt_scores.flatten(), 0, assigned_gt_index.flatten())
            gather_scores = gather_scores.reshape([batch_size, num_anchors])
            gather_scores = torch.where(mask_positive_sum > 0, gather_scores, torch.zeros_like(gather_scores))
            assigned_scores *= gather_scores.unsqueeze(-1)

        return assigned_labels, assigned_bboxes, assigned_scores


def batch_distance2bbox(points, distance, max_shapes=None):
    """Decode distance prediction to bounding box for batch.
    Args:
        points (Tensor): [B, ..., 2], "xy" format
        distance (Tensor): [B, ..., 4], "ltrb" format
        max_shapes (Tensor): [B, 2], "h,w" format, Shape of the image.
    Returns:
        Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    lt, rb = torch.split(distance, 2, -1)
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = torch.concat([x1y1, x2y2], -1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))
    return out_bbox


def iou_similarity(box1, box2, eps=1e-10):
    """Calculate iou of box1 and box2

    Args:
        box1 (Tensor): box with the shape [M1, 4]
        box2 (Tensor): box with the shape [M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [M1, M2]
    """
    box1 = box1.unsqueeze(1)  # [M1, 4] -> [M1, 1, 4]
    box2 = box2.unsqueeze(0)  # [M2, 4] -> [1, M2, 4]
    px1y1, px2y2 = box1[:, :, 0:2], box1[:, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, 0:2], box2[:, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def bbox_center(boxes):
    """Get bbox centers from boxes.
    Args:
        boxes (Tensor): boxes with shape (..., 4), "xmin, ymin, xmax, ymax" format.
    Returns:
        Tensor: boxes centers with shape (..., 2), "cx, cy" format.
    """
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return torch.stack([boxes_cx, boxes_cy], dim=-1)


def batch_iou_similarity(box1, box2, eps=1e-9):
    """Calculate iou of box1 and box2 in batch

    Args:
        box1 (Tensor): box with the shape [N, M1, 4]
        box2 (Tensor): box with the shape [N, M2, 4]

    Return:
        iou (Tensor): iou between box1 and box2 with the shape [N, M1, M2]
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def compute_max_iou_gt(ious):
    r"""
    For each GT, find the anchor with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = ious.shape[-1]
    max_iou_index = ious.argmax(dim=-1)
    is_max_iou = F.one_hot(max_iou_index, num_anchors)
    return is_max_iou.to(ious.dtype)


def compute_max_iou_anchor(ious):
    r"""
    For each anchor, find the GT with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(dim=-2)
    is_max_iou = F.one_hot(max_iou_index, num_max_boxes).permute([0, 2, 1])
    return is_max_iou.to(ious.dtype)


def check_points_inside_bboxes(
        points,
        bboxes,
        center_radius_tensor=None,
        eps=1e-9,
        sm_use=False
):
    r"""
    Args:
        points (Tensor, float32): shape[L, 2], "xy" format, L: num_anchors
        bboxes (Tensor, float32): shape[B, n, 4], "xmin, ymin, xmax, ymax" format
        center_radius_tensor (Tensor, float32): shape [L, 1]. Default: None.
        eps (float): Default: 1e-9
    Returns:
        is_in_bboxes (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    points = points[None, None]
    x, y = points.chunk(2, dim=-1)
    xmin, ymin, xmax, ymax = bboxes.unsqueeze(2).chunk(4, dim=-1)
    # check whether `points` is in `bboxes`
    l = x - xmin
    t = y - ymin
    r = xmax - x
    b = ymax - y
    delta_ltrb = torch.concat([l, t, r, b], dim=-1)
    d, _ = delta_ltrb.min(dim=-1)
    is_in_bboxes = (d > eps)
    if center_radius_tensor is not None:
        # check whether `points` is in `center_radius`
        center_radius_tensor = center_radius_tensor.unsqueeze([0, 1])
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        l = x - (cx - center_radius_tensor)
        t = y - (cy - center_radius_tensor)
        r = (cx + center_radius_tensor) - x
        b = (cy + center_radius_tensor) - y
        delta_ltrb_c = torch.concat([l, t, r, b], dim=-1)
        is_in_center = (delta_ltrb_c.min(dim=-1) > eps)
        if sm_use:
            return is_in_bboxes.to(bboxes.dtype), is_in_center.to(bboxes.dtype)
        else:
            return (torch.logical_and(is_in_bboxes, is_in_center),
                    torch.logical_or(is_in_bboxes, is_in_center))

    return is_in_bboxes.to(bboxes.dtype)


def gather_topk_anchors(metrics, topk, largest=True, topk_mask=None, eps=1e-9):
    r"""
    Args:
        metrics (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
        topk (int): The number of top elements to look for along the axis.
        largest (bool) : largest is a flag, if set to true,
            algorithm will sort by descending order, otherwise sort by
            ascending order. Default: True
        topk_mask (Tensor, float32): shape[B, n, 1], ignore bbox mask,
            Default: None
        eps (float): Default: 1e-9
    Returns:
        is_in_topk (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > eps).astype(metrics.dtype)
    is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(dim=-2).to(metrics.dtype)
    return is_in_topk * topk_mask


class TaskAlignedAssigner_CR(nn.Module):
    """TOOD: Task-aligned One-stage Object Detection with Center R
    """

    def __init__(
            self,
            topk=13,
            alpha=1.0,
            beta=6.0,
            center_radius=None,
            eps=1e-9
    ):
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.center_radius = center_radius
        self.eps = eps

    def forward(
            self,
            pred_scores,
            pred_bboxes,
            anchor_points,
            stride_tensor,
            gt_labels,
            gt_bboxes,
            pad_gt_mask,
            bg_index,
            gt_scores=None
    ):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            pred_scores (Tensor, float32): predicted class probability, shape(B, L, C)
            pred_bboxes (Tensor, float32): predicted bounding boxes, shape(B, L, 4)
            anchor_points (Tensor, float32): pre-defined anchors, shape(L, 2), "cxcy" format
            stride_tensor (Tensor, float32): stride of feature map, shape(L, 1)
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes, shape(B, n, 1)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C)
        """
        assert pred_scores.ndim == pred_bboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full((batch_size, num_anchors), bg_index, dtype=torch.int32)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros([batch_size, num_anchors, num_classes])
            return assigned_labels, assigned_bboxes, assigned_scores

        # compute iou between gt and pred bbox, [B, n, L]
        ious = batch_iou_similarity(gt_bboxes, pred_bboxes)
        # gather pred bboxes class score
        pred_scores = pred_scores.permute([0, 2, 1])
        batch_ind = torch.arange(end=batch_size, dtype=gt_labels.dtype, device=gt_labels.device).unsqueeze(-1)
        gt_labels_ind = torch.stack(
            [batch_ind.tile([1, num_max_boxes]), gt_labels.squeeze(-1)],
            dim=-1
        )
        bbox_cls_scores = pred_scores[gt_labels_ind[:, :, 0], gt_labels_ind[:, :, 1]]  # compute alignment metrics, [B, n, L]
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta) * pad_gt_mask

        # select positive sample, [B, n, L]
        if self.center_radius is None:
            # check the positive sample's center in gt, [B, n, L]
            is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes, sm_use=True)
            # select topk largest alignment metrics pred bbox as candidates
            # for each gt, [B, n, L]
            mask_positive = gather_topk_anchors(alignment_metrics, self.topk, topk_mask=pad_gt_mask) * is_in_gts
        else:
            is_in_gts, is_in_center = check_points_inside_bboxes(
                anchor_points,
                gt_bboxes,
                stride_tensor * self.center_radius,
                sm_use=True
            )
            is_in_gts *= pad_gt_mask
            is_in_center *= pad_gt_mask
            candidate_metrics = torch.where(
                is_in_gts.sum(-1, keepdim=True) == 0,
                alignment_metrics + is_in_center,
                alignment_metrics
            )
            mask_positive = gather_topk_anchors(
                candidate_metrics, self.topk,
                topk_mask=pad_gt_mask
            ) * (is_in_center > 0) | (is_in_gts > 0).to(torch.float32)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected, [B, n, L]
        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile([1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious * mask_positive)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        assigned_gt_index = mask_positive.argmax(dim=-2)

        # assigned target
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = torch.gather(gt_labels.flatten(), 0, assigned_gt_index.flatten())
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(
            mask_positive_sum > 0, assigned_labels,
            torch.full_like(assigned_labels, bg_index)
        )

        assigned_bboxes = torch.gather(gt_bboxes.reshape([-1, 4]), 0, assigned_gt_index.flatten()[:, None].expand(-1, 4))
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_scores = F.one_hot(assigned_labels, num_classes + 1)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(assigned_scores, -1, torch.tensor(ind, device=assigned_scores.device))
        # rescale alignment metrics
        alignment_metrics *= mask_positive
        max_metrics_per_instance, _ = alignment_metrics.max(dim=-1, keepdim=True)
        max_ious_per_instance, _ = (ious * mask_positive).max(dim=-1, keepdim=True)
        alignment_metrics = alignment_metrics / (max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics, _ = alignment_metrics.max(-2)
        alignment_metrics = alignment_metrics.unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        return assigned_labels, assigned_bboxes, assigned_scores


class GIoULoss(nn.Module):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1., eps=1e-10, reduction='none'):
        super().__init__()
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xkis1 = torch.maximum(x1, x1g)
        ykis1 = torch.maximum(y1, y1g)
        xkis2 = torch.minimum(x2, x2g)
        ykis2 = torch.minimum(y2, y2g)
        # w_inter = (xkis2 - xkis1).clip(0, 1e25)
        # h_inter = (ykis2 - ykis1).clip(0, 1e25)
        w_inter = torch.maximum(xkis2 - xkis1, torch.tensor(0.))
        h_inter = torch.maximum(ykis2 - ykis1, torch.tensor(0.))
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union

        return iou, overlap, union

    def forward(self, pbox, gbox, iou_weight=1., loc_reweight=None):
        x1, y1, x2, y2 = torch.split(pbox, split_size_or_sections=1, dim=-1)
        x1g, y1g, x2g, y2g = torch.split(gbox, split_size_or_sections=1, dim=-1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = torch.minimum(x1, x1g)
        yc1 = torch.minimum(y1, y1g)
        xc2 = torch.maximum(x2, x2g)
        yc2 = torch.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = torch.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == 'none':
            loss = giou
        elif self.reduction == 'sum':
            loss = torch.sum(giou * iou_weight)
        else:
            loss = torch.mean(giou * iou_weight)
        return loss * self.loss_weight
