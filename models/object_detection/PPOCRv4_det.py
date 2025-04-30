import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils import math_utils, torch_utils
from .. import bundles
from ..image_classification import PPLCNetV3
from ..layers import Conv, ConvT


class Config(bundles.Config):
    student_backbone = dict(
        name='models.image_classification.PPLCNetV3.Backbone',
        **PPLCNetV3.Config.det_backbone
    )

    neck = dict(
        in_ches=[12, 18, 42, 360],
        out_ch=96
    )

    head = dict(
        k=50
    )

    default_model = 'student'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            'student': dict(
                backbone_config=cls.student_backbone,
                neck_config=cls.neck,
                head_config=cls.head
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
        'head.binarize.conv1': 'head.binarize.0.conv',
        'head.binarize.conv_bn1': 'head.binarize.0.norm',
        'head.binarize.conv2': 'head.binarize.1.conv',
        'head.binarize.conv_bn2': 'head.binarize.1.norm',
        'head.binarize.conv3': 'head.binarize.2.conv',
        'head.thresh.conv1': 'head.thresh.0.conv',
        'head.thresh.conv_bn1': 'head.thresh.0.norm',
        'head.thresh.conv2': 'head.thresh.1.conv',
        'head.thresh.conv_bn2': 'head.thresh.1.norm',
        'head.thresh.conv3': 'head.thresh.2.conv'
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
            '{0}.bn': '{0}.norm',
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        return state_dict


class Model(nn.Module):
    def __init__(
            self, backbone_config=Config.student_backbone, neck_config=Config.neck, head_config=Config.head,
            **kwargs
    ):
        super().__init__()
        self.__dict__.update(kwargs)

        self.backbone = PPLCNetV3.Backbone(**backbone_config)
        self.neck = RSEFPN(**neck_config)
        self.head = DBHead(in_ch=self.neck.out_channels, **head_config)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        pred = self.head(x)

        if self.training:
            raise NotImplementedError
        else:
            return self.post_process(pred)

    thresh = 0.3
    min_size = 3
    unclip_ratio = 2.0
    box_thresh = 0.7

    def post_process(self, preds):
        preds = preds.cpu().numpy()
        masks = (preds > self.thresh).astype(np.uint8)
        bboxes = []
        for mask, pred in zip(masks, preds):
            mask = mask[0]
            pred = pred[0]
            outs = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = outs[0]
            _bboxes = []
            for contour in contours:
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
                _bboxes.append(points)
            _bboxes = np.array(_bboxes).astype(int)
            bboxes.append(_bboxes)
        return bboxes

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
        from shapely.geometry import Polygon
        import pyclipper

        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded


class RSEFPN(nn.Module):
    def __init__(self, in_ches, out_ch):
        super().__init__()
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()

        for in_ch in in_ches:
            self.ins_conv.append(
                ResBlock(in_ch, out_ch, 1)
            )
            self.inp_conv.append(
                ResBlock(out_ch, out_ch // 4, 3)
            )

        self.out_channels = out_ch

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(in5, scale_factor=2, mode="nearest")  # 1/16
        out3 = in3 + F.upsample(out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.upsample(out3, scale_factor=2, mode="nearest")  # 1/4

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest")
        p4 = F.upsample(p4, scale_factor=4, mode="nearest")
        p3 = F.upsample(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=int(kernel_size // 2), bias=False)
        self.se_block = PPLCNetV3.SEBlock(out_ch)

    def forward(self, ins):
        x = self.in_conv(ins)
        out = x + self.se_block(x)
        return out


class DBHead(nn.Module):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_ch, k=50):
        super().__init__()
        self.k = k
        self.binarize = Head(in_ch)
        self.thresh = Head(in_ch)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        shrink_maps = self.binarize(x)
        if not self.training:
            return shrink_maps

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = torch.cat([shrink_maps, threshold_maps, binary_maps], dim=1)
        return y


class Head(nn.Sequential):
    def __init__(self, in_ch):
        super().__init__(
            Conv(in_ch, in_ch // 4, 3, bias=False, mode='cna'),
            ConvT(in_ch // 4, in_ch // 4, 2, 2, mode='cna'),
            ConvT(in_ch // 4, 1, 2, 2, mode='ca', act=nn.Sigmoid())
        )
