import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils import converter, math_utils, torch_utils
from .. import bundles
from ..image_classification import PPLCNetV3, PPHGNet
from ..layers import Conv, ConvT, Cache


class Config(bundles.Config):
    student_backbone = dict(
        name='models.image_classification.PPLCNetV3.Backbone',
        **PPLCNetV3.Config.det_backbone
    )

    teacher_backbone = dict(
        name='models.image_classification.PPHGNet.Backbone',
        **PPHGNet.Config.det_small_backbone
    )

    student_neck = dict(
        name='models.object_detection.PPOCRv4_det.RSEFPN',
        in_ches=[12, 18, 42, 360],
        out_ch=96,
        shortcut=True
    )

    teacher_neck = dict(
        name='models.object_detection.PPOCRv4_det.LKPAN',
        in_ches=[256, 512, 768, 1024],
        out_ch=256,
        intracl=True
    )

    student_head = dict(
        name='models.object_detection.PPOCRv4_det.DBHead',
        k=50,
        fix_nan=True
    )

    teacher_head = dict(
        name='models.object_detection.PPOCRv4_det.PFHeadLocal',
        k=50,
        mode='large',
        fix_nan=True
    )

    student_loss = dict(
        main_loss_type='DiceLoss',
        alpha=5,
        beta=10,
        ohem_ratio=3
    )

    teacher_loss = dict(
        main_loss_type='DiceLoss',
        alpha=5,
        beta=10,
        ohem_ratio=3
    )

    default_model = 'student'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            'student': dict(
                backbone_config=cls.student_backbone,
                neck_config=cls.student_neck,
                head_config=cls.student_head,
                loss_config=cls.student_loss
            ),

            'teacher': dict(
                backbone_config=cls.teacher_backbone,
                neck_config=cls.teacher_neck,
                head_config=cls.teacher_head,
                loss_config=cls.teacher_loss
            )
        }


class WeightConverter:

    @staticmethod
    def _convert(state_dict):
        info = []
        for k in state_dict.keys():
            if (
                    k.endswith('fc1.weight') or k.endswith('fc2.weight')
                    or k.endswith('fc.weight') or k.endswith('qkv.weight')
                    or k.endswith('proj.weight')
            ):
                info.append(('w', 'l'))
            elif k.endswith('._mean'):
                info.append(('nm', 'n'))
            elif k.endswith('._variance'):
                info.append(('nv', 'n'))
            else:
                info.append(('', ''))

        key_types, value_types = math_utils.transpose(info)
        state_dict = torch_utils.Converter.tensors_from_paddle_to_torch(state_dict, key_types, value_types)
        return state_dict

    head_convert_dict = {
        'head.{0}.conv1': 'head.{0}.blocks.0.conv',
        'head.{0}.conv_bn1': 'head.{0}.blocks.0.norm',
        'head.{0}.conv2': 'head.{0}.blocks.1.conv',
        'head.{0}.conv_bn2': 'head.{0}.blocks.1.norm',
        'head.{0}.conv3': 'head.{0}.blocks.3.conv',
    }

    teacher_neck_convert_dict = {
        'neck.incl{0}.bn': 'neck.incl{0}.conv1x1_return_channel.norm',
        'neck.incl{0}.conv1x1_return_channel': 'neck.incl{0}.conv1x1_return_channel.conv',
    }

    @classmethod
    def from_student(cls, state_dict):
        # https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar
        # it seems that, the lost the weight of se_block, but so confused that it won't affect the results
        state_dict = cls._convert(state_dict)

        convert_dict = {
            **PPLCNetV3.WeightConverter.backbone_convert_dict,
            **cls.head_convert_dict
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        return state_dict

    @classmethod
    def from_teacher(cls, state_dict):
        state_dict = cls._convert(state_dict)

        convert_dict = {
            **cls.teacher_neck_convert_dict,
            **PPHGNet.WeightConverter.backbone_convert_dict,
            **cls.head_convert_dict,
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        return state_dict


class Model(nn.Module):
    def __init__(
            self, backbone_config=Config.student_backbone, neck_config=Config.student_neck, head_config=Config.student_head,
            loss_config=Config.student_loss,
            **kwargs
    ):
        super().__init__()
        self.__dict__.update(kwargs)

        backbone_config = dict(backbone_config)
        neck_config = dict(neck_config)
        head_config = dict(head_config)
        loss_config = dict(loss_config or {})

        backbone_name = backbone_config.pop('name')
        self.backbone = converter.DataInsConvert.str_to_instance(backbone_name)(**backbone_config)
        neck_name = neck_config.pop('name')
        self.neck = converter.DataInsConvert.str_to_instance(neck_name)(**neck_config)
        head_name = head_config.pop('name')
        self.head = converter.DataInsConvert.str_to_instance(head_name)(in_ch=self.neck.out_channels, **head_config)
        self.criterion = DBLoss(**loss_config)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.fit(*args, **kwargs)
        else:
            return self.inference(*args, **kwargs)

    def fit(self, x, label_list=()):
        outputs = self.process(x)
        return self.loss(outputs, label_list)

    def inference(self, x, **kwargs):
        outputs = self.process(x)
        return self.post_process(outputs['preds'])

    def process(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(x)

    def loss(self, x, label_list=()):
        return self.criterion(x, label_list)

    thresh = 0.3
    min_size = 3
    unclip_ratio = 1.5
    box_thresh = 0.6
    max_candidates = 1000

    def post_process(self, preds):
        preds = preds.cpu().numpy()
        masks = (preds > self.thresh).astype(np.uint8)
        results = []
        for mask, pred in zip(masks, preds):
            mask = mask[0]
            pred = pred[0]
            outs = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = outs[0]
            segmentations = []
            for contour in contours[:self.max_candidates]:
                points, sside = self.get_mini_boxes(contour)
                if sside < self.min_size:
                    continue
                points = np.array(points)
                score = self.box_score_fast(pred, points.reshape(-1, 2))

                if self.box_thresh > score:
                    continue

                points = self.unclip(points).reshape(-1, 1, 2)
                points, sside = self.get_mini_boxes(points)
                if sside < self.min_size + 2:
                    continue
                segmentations.append(points)
            segmentations = np.array(segmentations).astype(int)
            results.append(dict(
                segmentations=segmentations,
                mask=mask
            ))
        return results

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def unclip(self, box):
        from shapely.geometry import Polygon  # pip install shapely
        import pyclipper  # pip install pyclipper

        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded


class Model4Export(Model):
    """for exporting to onnx, torchscript, etc."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        self.register_buffer('mean', mean, persistent=False)
        self.register_buffer('std', std, persistent=False)

    def forward(self, x):
        x = self.pre_process(x)
        outputs = self.process(x)
        preds = outputs['preds']
        preds = preds.to(torch.float16)
        return preds

    def pre_process(self, x):
        """for faster infer, use uint8 input and fp32 to output"""
        x = x.to(dtype=torch.float32)   # cannot use fp16
        x = x / 255
        x = (x - self.mean) / self.std
        return x


class RSEFPN(nn.Module):
    def __init__(self, in_ches, out_ch, shortcut=True):
        super().__init__()
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()

        for in_ch in in_ches:
            self.ins_conv.append(
                ResBlock(in_ch, out_ch, 1, shortcut=shortcut)
            )
            self.inp_conv.append(
                ResBlock(out_ch, out_ch // 4, 3, shortcut=shortcut)
            )

        self.out_channels = out_ch

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest")  # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")  # 1/4

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, shortcut=True):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=int(kernel_size // 2), bias=False)
        self.se_block = PPLCNetV3.SEBlock(out_ch)
        self.shortcut = shortcut

    def forward(self, ins):
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_ch, k=50, return_f=False, aux_in_channels=0, fix_nan=False, **kwargs):
        super().__init__()
        self.k = k
        self.binarize = Head(in_ch, return_f=return_f, fix_nan=fix_nan)
        self.thresh = Head(in_ch, fix_nan=fix_nan)
        self.aux_in_channels = aux_in_channels

        if aux_in_channels > 0:
            self._aux_upsample_scale = {
                "aux_p4": 4,  # 1/16 -> 1/4
                "aux_p3": 2,  # 1/8  -> 1/4
                "aux_p2": 1,  # 1/4  -> 1/4
            }
            self.aux_binarize_p4 = Head(aux_in_channels, fix_nan=fix_nan)
            self.aux_thresh_p4 = Head(aux_in_channels, fix_nan=fix_nan)
            self.aux_binarize_p3 = Head(aux_in_channels, fix_nan=fix_nan)
            self.aux_thresh_p3 = Head(aux_in_channels, fix_nan=fix_nan)
            self.aux_binarize_p2 = Head(aux_in_channels, fix_nan=fix_nan)
            self.aux_thresh_p2 = Head(aux_in_channels, fix_nan=fix_nan)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        if isinstance(x, dict):
            fuse = x["fuse"]
            aux_feats = {k: x[k] for k in ("aux_p4", "aux_p3", "aux_p2") if k in x}
        else:
            fuse = x
            aux_feats = {}

        shrink_maps, _ = self.binarize(fuse)
        if self.training:
            threshold_maps, _ = self.thresh(fuse)
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            y = torch.cat([shrink_maps, threshold_maps, binary_maps], dim=1)
            outputs = {
                "preds": y,
            }
            if self.aux_in_channels > 0 and aux_feats:
                for key, feat in aux_feats.items():
                    scale = self._aux_upsample_scale[key]
                    if scale > 1:
                        feat = F.interpolate(feat, scale_factor=scale, mode="bilinear", align_corners=False)
                    level = key[4:]
                    aux_binarize = getattr(self, "aux_binarize_" + level)
                    aux_thresh_head = getattr(self, "aux_thresh_" + level)
                    aux_shrink, _ = aux_binarize(feat)
                    aux_thresh, _ = aux_thresh_head(feat)
                    aux_binary = self.step_function(aux_shrink, aux_thresh)
                    outputs["aux_maps_" + level] = torch.cat(
                        [aux_shrink, aux_thresh, aux_binary], dim=1
                    )
        else:
            y = shrink_maps
            outputs = {
                "preds": y,
            }

        return outputs


class Head(nn.Module):
    def __init__(self, in_ch, return_f=False, fix_nan=False):
        super().__init__()
        self.fix_nan = fix_nan
        self.blocks = nn.ModuleList([
            Conv(in_ch, in_ch // 4, 3, bias=False, mode='cna'),
            ConvT(in_ch // 4, in_ch // 4, 2, 2, mode='cna'),
            Cache() if return_f else nn.Identity(),
            ConvT(in_ch // 4, 1, 2, 2, mode='ca', act=nn.Sigmoid())
        ])

    def forward(self, x):
        features = []
        for i, m in enumerate(self.blocks):
            if isinstance(m, Cache):
                x, features = m(x, features)
            else:
                x = m(x)
                if self.fix_nan and self.training and i in (0, 1):
                    x = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        return x, features


class LKPAN(nn.Module):
    def __init__(self, in_ches, out_ch, intracl=True):
        super().__init__()
        self.out_channels = out_ch
        self.intracl = intracl

        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        # pan head
        self.pan_head_conv = nn.ModuleList()
        self.pan_lat_conv = nn.ModuleList()

        for i in range(len(in_ches)):
            self.ins_conv.append(
                nn.Conv2d(in_ches[i], out_ch, 1, bias=False)
            )

            self.inp_conv.append(
                nn.Conv2d(out_ch, out_ch // 4, 9, padding=4, bias=False)
            )

            if i > 0:
                self.pan_head_conv.append(
                    nn.Conv2d(out_ch // 4, out_ch // 4, 3, padding=1, stride=2, bias=False)
                )

            self.pan_lat_conv.append(
                nn.Conv2d(out_ch // 4, out_ch // 4, 9, padding=4, bias=False)
            )

        if self.intracl:
            self.incl1 = IntraCLBlock(out_ch // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(out_ch // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(out_ch // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(out_ch // 4, reduce_factor=2)

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest")  # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")  # 1/4

        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)

        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)

        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)

        if self.intracl:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class IntraCLBlock(nn.Module):
    def __init__(self, in_ch=96, reduce_factor=4):
        super().__init__()
        hidden_ch = in_ch // reduce_factor

        self.conv1x1_reduce_channel = nn.Conv2d(
            in_ch,
            hidden_ch,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.v_layer_7x1 = nn.Conv2d(
            hidden_ch,
            hidden_ch,
            kernel_size=(7, 1),
            stride=(1, 1),
            padding=(3, 0)
        )
        self.v_layer_5x1 = nn.Conv2d(
            hidden_ch,
            hidden_ch,
            kernel_size=(5, 1),
            stride=(1, 1),
            padding=(2, 0)
        )
        self.v_layer_3x1 = nn.Conv2d(
            hidden_ch,
            hidden_ch,
            kernel_size=(3, 1),
            stride=(1, 1),
            padding=(1, 0)
        )

        self.q_layer_1x7 = nn.Conv2d(
            hidden_ch,
            hidden_ch,
            kernel_size=(1, 7),
            stride=(1, 1),
            padding=(0, 3)
        )
        self.q_layer_1x5 = nn.Conv2d(
            hidden_ch,
            hidden_ch,
            kernel_size=(1, 5),
            stride=(1, 1),
            padding=(0, 2)
        )
        self.q_layer_1x3 = nn.Conv2d(
            hidden_ch,
            hidden_ch,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1)
        )

        # base
        self.c_layer_7x7 = nn.Conv2d(
            hidden_ch,
            hidden_ch,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=(3, 3)
        )
        self.c_layer_5x5 = nn.Conv2d(
            hidden_ch,
            hidden_ch,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2)
        )
        self.c_layer_3x3 = nn.Conv2d(
            hidden_ch,
            hidden_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.conv1x1_return_channel = Conv(hidden_ch, in_ch, 1, 1, mode='cna')

    def forward(self, x):
        x_new = self.conv1x1_reduce_channel(x)

        x_7_c = self.c_layer_7x7(x_new)
        x_7_v = self.v_layer_7x1(x_new)
        x_7_q = self.q_layer_1x7(x_new)
        x_7 = x_7_c + x_7_v + x_7_q

        x_5_c = self.c_layer_5x5(x_7)
        x_5_v = self.v_layer_5x1(x_7)
        x_5_q = self.q_layer_1x5(x_7)
        x_5 = x_5_c + x_5_v + x_5_q

        x_3_c = self.c_layer_3x3(x_5)
        x_3_v = self.v_layer_3x1(x_5)
        x_3_q = self.q_layer_1x3(x_5)
        x_3 = x_3_c + x_3_v + x_3_q

        x_relation = self.conv1x1_return_channel(x_3)
        return x + x_relation


class PFHeadLocal(DBHead):
    def __init__(self, in_ch, k=50, mode='large', **kwargs):
        super().__init__(in_ch, k, return_f=True, **kwargs)
        self.mode = mode
        self.up_conv = nn.Upsample(scale_factor=2, mode="nearest")
        if self.mode == "large":
            self.cbn_layer = LocalModule(in_ch // 4, in_ch // 4)
        elif self.mode == "small":
            self.cbn_layer = LocalModule(in_ch // 4, in_ch // 8)
        else:
            raise ValueError(f"mode can only be one of ['large', 'small'], but received {mode}")

    def forward(self, x):
        shrink_maps, features = self.binarize(x)
        f = features[0]
        base_maps = shrink_maps
        cbn_maps = self.cbn_layer(self.up_conv(f), shrink_maps)
        cbn_maps = F.sigmoid(cbn_maps)

        if self.training:
            # why only training steps use multi maps
            threshold_maps, _ = self.thresh(x)
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            y = torch.cat([cbn_maps, threshold_maps, binary_maps], dim=1)
            outputs = {
                "preds": y,
                "distance_maps": cbn_maps,
                "cbn_maps": binary_maps
            }
        else:
            y = 0.5 * (base_maps + cbn_maps)
            outputs = {
                "preds": y,
            }

        return outputs


class LocalModule(nn.Module):
    def __init__(self, in_c, mid_c):
        super().__init__()
        self.last_3 = Conv(in_c + 1, mid_c, 3, 1, 1, bias=False, mode='cna')
        self.last_1 = nn.Conv2d(mid_c, 1, 1, 1, 0)

    def forward(self, x, init_map):
        outf = torch.cat([init_map, x], dim=1)
        # last Conv
        out = self.last_1(self.last_3(outf))
        return out


class DBLoss(nn.Module):
    """Differentiable Binarization (DB) Loss Function"""

    def __init__(
            self,
            main_loss_type="DiceLoss",
            alpha=5,
            beta=10,
            ohem_ratio=3,
            eps=1e-6,
            aux_weight_p4=0.0,
            aux_weight_p3=0.0,
            aux_weight_p2=0.0,
            focal_alpha=0.25,
            focal_gamma=2.0,
            dice_weight=1.0,
            focal_weight=1.0,
            **kwargs,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.aux_weight_p4 = aux_weight_p4
        self.aux_weight_p3 = aux_weight_p3
        self.aux_weight_p2 = aux_weight_p2

        self.l1_loss = MaskL1Loss(eps=eps)
        if main_loss_type == "DiceFocalLoss":
            self.bce_loss = DiceFocalLoss(
                dice_weight=dice_weight,
                focal_weight=focal_weight,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                eps=eps
            )
            self.dice_loss = self.bce_loss
        else:
            self.dice_loss = DiceLoss(eps=eps)
            self.bce_loss = BalanceLoss(
                loss_type=main_loss_type,
                negative_ratio=ohem_ratio,
                eps=eps
            )

    def forward(self, outputs, label_list):
        predict_maps = outputs["preds"]
        (
            label_threshold_map,
            label_threshold_mask,
            label_shrink_map,
            label_shrink_mask,
        ) = label_list
        shrink_maps = predict_maps[:, 0, :, :]
        threshold_maps = predict_maps[:, 1, :, :]
        binary_maps = predict_maps[:, 2, :, :]

        loss_shrink_maps = self.bce_loss(
            shrink_maps, label_shrink_map, label_shrink_mask
        )
        loss_threshold_maps = self.l1_loss(
            threshold_maps, label_threshold_map, label_threshold_mask
        )
        loss_binary_maps = self.dice_loss(
            binary_maps, label_shrink_map, label_shrink_mask
        )
        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_threshold_maps = self.beta * loss_threshold_maps

        # CBN loss
        if "cbn_maps" in outputs.keys():
            cbn_maps = outputs["cbn_maps"]
            cbn_loss = self.bce_loss(
                cbn_maps[:, 0, :, :], label_shrink_map, label_shrink_mask
            )
        else:
            cbn_loss = torch.tensor([0.0]).to(loss_shrink_maps)

        loss_all = loss_shrink_maps + loss_threshold_maps + loss_binary_maps + cbn_loss
        losses = {
            "loss": loss_all,
            "loss.shrink_maps": loss_shrink_maps,
            "loss.threshold_maps": loss_threshold_maps,
            "loss.binary_maps": loss_binary_maps,
            "loss.cbn": cbn_loss,
        }

        for aux_key, aux_w in [
            ("aux_maps_p4", self.aux_weight_p4),
            ("aux_maps_p3", self.aux_weight_p3),
            ("aux_maps_p2", self.aux_weight_p2),
        ]:
            if aux_w > 0 and aux_key in outputs:
                aux_maps = outputs[aux_key]
                aux_shrink = aux_maps[:, 0, :, :]
                aux_threshold = aux_maps[:, 1, :, :]
                aux_binary = aux_maps[:, 2, :, :]
                l_shrink = self.alpha * self.bce_loss(
                    aux_shrink, label_shrink_map, label_shrink_mask
                )
                l_threshold = self.beta * self.l1_loss(
                    aux_threshold, label_threshold_map, label_threshold_mask
                )
                l_binary = self.dice_loss(
                    aux_binary, label_shrink_map, label_shrink_mask
                )
                aux_loss = l_shrink + l_threshold + l_binary
                losses["loss." + aux_key] = aux_loss
                losses["loss"] = losses["loss"] + aux_w * aux_loss

        return losses


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, preds, gt, mask, weights=None):
        if weights is not None:
            mask = weights * mask
        intersection = torch.sum(preds * gt * mask)

        union = torch.sum(preds * mask) + torch.sum(gt * mask) + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, preds, gt, mask):
        loss = (torch.abs(preds - gt) * mask).sum() / (mask.sum() + self.eps)
        loss = torch.mean(loss)
        return loss


class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, preds, gt, mask):
        preds = preds.clamp(self.eps, 1.0 - self.eps)
        logit = torch.log(preds / (1.0 - preds))
        bce = F.binary_cross_entropy_with_logits(logit, gt, reduction='none')
        p = torch.sigmoid(logit)
        p_t = p * gt + (1 - p) * (1 - gt)
        alpha_t = self.alpha * gt + (1 - self.alpha) * (1 - gt)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return (loss * mask).sum() / (mask.sum() + self.eps)


class DiceFocalLoss(nn.Module):
    def __init__(
            self,
            dice_weight=1.0,
            focal_weight=1.0,
            focal_alpha=0.25,
            focal_gamma=2.0,
            eps=1e-6,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(eps=eps)
        self.focal_loss = MaskedFocalLoss(alpha=focal_alpha, gamma=focal_gamma, eps=eps)

    def forward(self, preds, gt, mask=None, weights=None):
        loss_dice = self.dice_loss(preds, gt, mask, weights=weights)
        loss_focal = self.focal_loss(preds, gt, mask)
        return self.dice_weight * loss_dice + self.focal_weight * loss_focal


class BalanceLoss(nn.Module):
    """The BalanceLoss for Differentiable Binarization text detection"""

    loss_fn_mapping = {
        "CrossEntropy": nn.CrossEntropyLoss,
        "Euclidean": nn.MSELoss,
        "DiceLoss": DiceLoss,
        "BCELoss": nn.BCELoss,
        "MaskL1Loss": MaskL1Loss,
    }

    def __init__(
            self,
            loss_type="DiceLoss",
            negative_ratio=3,
            eps=1e-6,
            **loss_kwargs,
    ):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

        self.criterion = self.loss_fn_mapping[loss_type](**loss_kwargs)

    def forward(self, preds, gt, mask=None):
        loss = self.criterion(preds, gt, mask=mask)

        positive = gt * mask
        negative = (1 - gt) * mask

        positive_count = int(positive.sum())
        negative_count = int(min(negative.sum(), positive_count * self.negative_ratio))
        positive_loss = positive * loss
        negative_loss = negative * loss
        negative_loss = torch.reshape(negative_loss, shape=[-1])
        if negative_count > 0:
            sort_loss, _ = negative_loss.sort(descending=True)
            negative_loss = sort_loss[:negative_count]
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)
        else:
            balance_loss = positive_loss.sum() / (positive_count + self.eps)

        return balance_loss
