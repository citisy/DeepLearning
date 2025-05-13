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
        out_ch=96
    )

    teacher_neck = dict(
        name='models.object_detection.PPOCRv4_det.LKPAN',
        in_ches=[256, 512, 768, 1024],
        out_ch=256
    )

    student_head = dict(
        name='models.object_detection.PPOCRv4_det.DBHead',
        k=50
    )

    teacher_head = dict(
        name='models.object_detection.PPOCRv4_det.PFHeadLocal',
        k=50,
    )

    default_model = 'student'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            'student': dict(
                backbone_config=cls.student_backbone,
                neck_config=cls.student_neck,
                head_config=cls.student_head
            ),

            'teacher': dict(
                backbone_config=cls.teacher_backbone,
                neck_config=cls.teacher_neck,
                head_config=cls.teacher_head
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
        'head.binarize.conv1': 'head.binarize.blocks.0.conv',
        'head.binarize.conv_bn1': 'head.binarize.blocks.0.norm',
        'head.binarize.conv2': 'head.binarize.blocks.1.conv',
        'head.binarize.conv_bn2': 'head.binarize.blocks.1.norm',
        'head.binarize.conv3': 'head.binarize.blocks.3.conv',
        'head.thresh.conv1': 'head.thresh.blocks.0.conv',
        'head.thresh.conv_bn1': 'head.thresh.blocks.0.norm',
        'head.thresh.conv2': 'head.thresh.blocks.1.conv',
        'head.thresh.conv_bn2': 'head.thresh.blocks.1.norm',
        'head.thresh.conv3': 'head.thresh.blocks.3.conv'
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
            'neck.incl{0}.bn': 'neck.incl{0}.conv1x1_return_channel.norm',
            'neck.incl{0}.conv1x1_return_channel': 'neck.incl{0}.conv1x1_return_channel.conv',

            **PPHGNet.WeightConverter.backbone_convert_dict,
            **cls.head_convert_dict,
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        return state_dict


class Model(nn.Module):
    def __init__(
            self, backbone_config=Config.student_backbone, neck_config=Config.student_neck, head_config=Config.student_head,
            **kwargs
    ):
        super().__init__()
        self.__dict__.update(kwargs)

        backbone_name = backbone_config.pop('name')
        self.backbone = converter.DataInsConvert.str_to_instance(backbone_name)(**backbone_config)
        neck_name = neck_config.pop('name')
        self.neck = converter.DataInsConvert.str_to_instance(neck_name)(**neck_config)
        head_name = head_config.pop('name')
        self.head = converter.DataInsConvert.str_to_instance(head_name)(in_ch=self.neck.out_channels, **head_config)
        self.criterion = DBLoss()

    def forward(self, x, label_list=()):
        x = self.backbone(x)
        x = self.neck(x)
        outputs = self.head(x)

        if self.training:
            return self.loss(outputs, label_list)
        else:
            return self.post_process(outputs['preds'])

    def loss(self, x, label_list=()):
        return self.criterion(x, label_list)

    thresh = 0.3
    min_size = 3
    unclip_ratio = 2.0
    box_thresh = 0.7

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
        from shapely.geometry import Polygon    # pip install shapely
        import pyclipper  # pip install pyclipper

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

    def __init__(self, in_ch, k=50, return_f=False):
        super().__init__()
        self.k = k
        self.binarize = Head(in_ch, return_f=return_f)
        self.thresh = Head(in_ch)

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        shrink_maps, _ = self.binarize(x)
        if self.training:
            # why only use in training steps?
            threshold_maps, _ = self.thresh(x)
            binary_maps = self.step_function(shrink_maps, threshold_maps)
            y = torch.cat([shrink_maps, threshold_maps, binary_maps], dim=1)
            outputs = {
                "preds": y,
            }
        else:
            y = shrink_maps
            outputs = {
                "preds": y,
            }

        return outputs


class Head(nn.Module):
    def __init__(self, in_ch, return_f=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv(in_ch, in_ch // 4, 3, bias=False, mode='cna'),
            ConvT(in_ch // 4, in_ch // 4, 2, 2, mode='cna'),
            Cache() if return_f else nn.Identity(),
            ConvT(in_ch // 4, 1, 2, 2, mode='ca', act=nn.Sigmoid())
        ])

    def forward(self, x):
        features = []
        for m in self.blocks:
            if isinstance(m, Cache):
                x, features = m(x, features)
            else:
                x = m(x)

        return x, features


class LKPAN(nn.Module):
    def __init__(self, in_ches, out_ch):
        super().__init__()
        self.out_channels = out_ch

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
    def __init__(self, in_ch, k=50):
        super().__init__(in_ch, k, return_f=True)
        self.up_conv = nn.Upsample(scale_factor=2, mode="nearest")
        self.cbn_layer = LocalModule(in_ch // 4, in_ch // 4)

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
            **kwargs,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
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
