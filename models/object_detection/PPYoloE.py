from collections import OrderedDict

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

    def __init__(self):
        super().__init__()
        self.backbone = CSPResNet()
        self.neck = CustomCSPPAN()
        self.yolo_head = PPYOLOEHead()

    def forward(self, x):
        body_feats = self.backbone(x)
        neck_feats = self.neck(body_feats)

        yolo_head_outs = self.yolo_head(neck_feats)
        post_outs = self.parse_preds(yolo_head_outs)

        return post_outs

    def parse_preds(self, head_outs):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = self.batch_distance2bbox(anchor_points, pred_dist)
        pred_bboxes *= stride_tensor
        pred_scores = pred_scores.permute([0, 2, 1])
        result = []
        for bboxes, scores in zip(pred_bboxes, pred_scores):
            keep, classes = (scores > self.conf_thres).nonzero(as_tuple=True)
            bboxes, scores = bboxes[keep], scores[keep, classes]
            keep = cls_nms(bboxes, scores, classes, iou_threshold=self.nms_thres)
            bboxes, classes, scores = bboxes[keep], classes[keep], scores[keep]
            result.append({
                "bboxes": bboxes,
                "confs": scores,
                "classes": classes.to(dtype=torch.int),
            })
        return result

    def batch_distance2bbox(self, points, distance, max_shapes=None):
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
        out_bbox = torch.cat([x1y1, x2y2], -1)
        if max_shapes is not None:
            max_shapes = max_shapes.flip(-1).tile([1, 2])
            delta_dim = out_bbox.ndim - max_shapes.ndim
            for _ in range(delta_dim):
                max_shapes.unsqueeze_(1)
            out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
            out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))
        return out_bbox


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

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(torch.float32)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float32))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

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
