import torch
import torch.nn.functional as F
from torch import nn

from utils import torch_utils
from .. import bundles
from ..image_classification import PPLCNetV4
from . import PPOCRv4_det


class Config(bundles.Config):
    tiny_backbone = dict(
        name='models.image_classification.PPLCNetV4.Backbone',
        **PPLCNetV4.Config.det_tiny_backbone
    )
    small_backbone = dict(
        name='models.image_classification.PPLCNetV4.Backbone',
        **PPLCNetV4.Config.det_small_backbone
    )
    medium_backbone = dict(
        name='models.image_classification.PPLCNetV4.Backbone',
        **PPLCNetV4.Config.det_medium_backbone
    )

    tiny_neck = dict(
        name='models.object_detection.PPOCRv6_det.RepLKFPN',
        in_ches=[32, 48, 64, 160],
        out_ch=64,
        dilated_kernel_size=5,
        shortcut=True
    )

    small_neck = dict(
        name='models.object_detection.PPOCRv6_det.RepLKFPN',
        in_ches=[48, 96, 192, 384],
        out_ch=96,
        dilated_kernel_size=7,
        shortcut=True
    )

    medium_neck = dict(
        name='models.object_detection.PPOCRv6_det.RepLKPAN',
        in_ches=[128, 256, 512, 896],
        out_ch=256,
        intracl=True
    )

    tiny_head = dict(
        name='models.object_detection.PPOCRv4_det.DBHead',
        k=50,
        fix_nan=True,
        aux_in_channels=64
    )
    small_head = dict(
        name='models.object_detection.PPOCRv4_det.DBHead',
        k=50,
        fix_nan=True,
        aux_in_channels=96
    )
    medium_head = dict(
        name='models.object_detection.PPOCRv4_det.DBHead',
        k=50,
        fix_nan=True,
        aux_in_channels=256
    )

    loss = dict(
        main_loss_type='DiceFocalLoss',
        alpha=5,
        beta=10,
        focal_alpha=0.25,
        focal_gamma=2.5,
        aux_weight_p4=0.2,
        aux_weight_p3=0.3,
        aux_weight_p2=0.4
    )

    default_model = 'medium'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            'tiny': dict(
                backbone_config=cls.tiny_backbone,
                neck_config=cls.tiny_neck,
                head_config=cls.tiny_head,
                loss_config=cls.loss,
                box_thresh=0.4
            ),
            'small': dict(
                backbone_config=cls.small_backbone,
                neck_config=cls.small_neck,
                head_config=cls.small_head,
                loss_config=cls.loss
            ),
            'medium': dict(
                backbone_config=cls.medium_backbone,
                neck_config=cls.medium_neck,
                head_config=cls.medium_head,
                loss_config=cls.loss
            )
        }


class WeightConverter(PPOCRv4_det.WeightConverter):
    neck_convert_dict = {
        'neck.incl{0}.bn': 'neck.incl{0}.conv1x1_return_channel.norm',
        'neck.incl{0}.conv1x1_return_channel': 'neck.incl{0}.conv1x1_return_channel.conv',
        'neck.ins_conv.{0}.se_block.conv1': 'neck.ins_conv.{0}.se_block.ex.0.conv',
        'neck.ins_conv.{0}.se_block.conv2': 'neck.ins_conv.{0}.se_block.ex.1.conv',
    }

    @classmethod
    def from_paddle(cls, state_dict):
        """
        tiny: https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_tiny_det_pretrained.pdparams
        small: https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_det_pretrained.pdparams
        medium: https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_det_pretrained.pdparams
        """
        state_dict = cls._convert(state_dict)
        convert_dict = {
            **cls.neck_convert_dict,
            **PPLCNetV4.WeightConverter.backbone_convert_dict,
            **cls.head_convert_dict,
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


class Model(PPOCRv4_det.Model):
    thresh = 0.2
    box_thresh = 0.45
    unclip_ratio = 1.4
    max_candidates = 3000

    def __init__(
            self,
            backbone_config=Config.medium_backbone,
            neck_config=Config.medium_neck,
            head_config=Config.medium_head,
            loss_config=Config.loss,
            **kwargs
    ):
        super().__init__(
            backbone_config=backbone_config,
            neck_config=neck_config,
            head_config=head_config,
            loss_config=loss_config,
            **kwargs
        )


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


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block from UniRepLKNet.
    Training uses multiple parallel dilated depthwise convolutions;
    inference can merge them into a single large-kernel depthwise conv.
    """

    def __init__(self, channels, kernel_size=9, deploy=False):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.is_repped = deploy

        if kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        else:
            raise ValueError(
                "DilatedReparamBlock requires kernel_size in [5,7,9,11,13], "
                f"but got {kernel_size}"
            )

        if not self.is_repped:
            self.lk_origin = nn.Conv2d(
                channels, channels, kernel_size,
                stride=1, padding=kernel_size // 2, groups=channels, bias=False
            )
            self.origin_bn = nn.BatchNorm2d(channels)

            for k, r in zip(self.kernel_sizes, self.dilates):
                equiv_ks = r * (k - 1) + 1
                conv = nn.Conv2d(
                    channels, channels, k,
                    stride=1, padding=equiv_ks // 2, dilation=r, groups=channels, bias=False
                )
                bn = nn.BatchNorm2d(channels)
                setattr(self, f"dil_conv_k{k}_{r}", conv)
                setattr(self, f"dil_bn_k{k}_{r}", bn)
        else:
            self.lk_origin = nn.Conv2d(
                channels, channels, kernel_size,
                stride=1, padding=kernel_size // 2, groups=channels, bias=True
            )

    def forward(self, x):
        if self.is_repped:
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, f"dil_conv_k{k}_{r}")
            bn = getattr(self, f"dil_bn_k{k}_{r}")
            out = out + bn(conv(x))
        return out

    @staticmethod
    def _fuse_bn(conv, bn):
        kernel = conv.weight
        gamma = bn.weight
        beta = bn.bias
        std = torch.sqrt(bn.running_var + bn.eps)
        fused_weight = kernel * (gamma / std).reshape(-1, 1, 1, 1)
        fused_bias = beta - bn.running_mean * gamma / std
        return fused_weight, fused_bias

    @staticmethod
    def _convert_dilated_to_nondilated(kernel, dilate_rate):
        if dilate_rate == 1:
            return kernel
        identity = torch.ones(1, 1, 1, 1, dtype=kernel.dtype, device=kernel.device)
        result_list = []
        for i in range(kernel.shape[0]):
            k_i = kernel[i:i + 1]
            dilated = F.conv_transpose2d(k_i, identity, stride=dilate_rate)
            result_list.append(dilated)
        return torch.cat(result_list, dim=0)

    @staticmethod
    def _merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
        large_k = large_kernel.shape[2]
        dilated_k = dilated_kernel.shape[2]
        equiv_ks = dilated_r * (dilated_k - 1) + 1
        equiv_kernel = DilatedReparamBlock._convert_dilated_to_nondilated(dilated_kernel, dilated_r)
        rows_to_pad = large_k // 2 - equiv_ks // 2
        if rows_to_pad > 0:
            merged = large_kernel + F.pad(equiv_kernel, [rows_to_pad] * 4)
        else:
            merged = large_kernel + equiv_kernel
        return merged

    @torch.no_grad()
    def rep(self):
        if self.is_repped:
            return
        origin_k, origin_b = self._fuse_bn(self.lk_origin, self.origin_bn)
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, f"dil_conv_k{k}_{r}")
            bn = getattr(self, f"dil_bn_k{k}_{r}")
            branch_k, branch_b = self._fuse_bn(conv, bn)
            origin_k = self._merge_dilated_into_large_kernel(origin_k, branch_k, r)
            origin_b = origin_b + branch_b

        merged_conv = nn.Conv2d(
            self.channels, self.channels, self.kernel_size,
            stride=1, padding=self.kernel_size // 2, groups=self.channels, bias=True
        )
        merged_conv.weight.copy_(origin_k)
        merged_conv.bias.copy_(origin_b)
        self.lk_origin = merged_conv
        self.is_repped = True

        delattr(self, "origin_bn")
        for k, r in zip(self.kernel_sizes, self.dilates):
            delattr(self, f"dil_conv_k{k}_{r}")
            delattr(self, f"dil_bn_k{k}_{r}")


class DilatedReparamConv(nn.Module):
    """Depthwise DilatedReparamBlock + 1x1 pointwise convolution."""

    def __init__(self, in_ch, out_ch, kernel_size=9, deploy=False):
        super().__init__()
        self.is_repped = False
        self.dw = DilatedReparamBlock(in_ch, kernel_size=kernel_size, deploy=deploy)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        if not self.is_repped:
            x = self.bn(x)
        return x

    @torch.no_grad()
    def rep(self):
        if self.is_repped:
            return
        self.dw.rep()
        conv, bn = self.pw, self.bn
        std = torch.sqrt(bn.running_var + bn.eps)
        scale = bn.weight / std
        w = conv.weight * scale[:, None, None, None]
        b = bn.bias - bn.running_mean * scale
        fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size)
        fused.weight.copy_(w)
        fused.bias.copy_(b)
        self.pw = fused
        del self.bn
        self.is_repped = True


class RepLKFPN(nn.Module):
    """RSEFPN with 3x3 inp_conv replaced by DilatedReparamBlock + PW + SE."""

    def __init__(self, in_ches, out_ch, shortcut=True, dilated_kernel_size=7, intracl=False):
        super().__init__()
        self.out_channels = out_ch
        self.is_repped = False
        self.shortcut = shortcut
        self.intracl = intracl

        self.ins_conv = nn.ModuleList()
        self.inp_conv_dw = nn.ModuleList()
        self.inp_conv_pw = nn.ModuleList()
        self.inp_conv_se = nn.ModuleList()

        for in_ch in in_ches:
            self.ins_conv.append(
                PPOCRv4_det.ResBlock(in_ch, out_ch, 1, shortcut=shortcut)
            )
            self.inp_conv_dw.append(
                DilatedReparamBlock(out_ch, kernel_size=dilated_kernel_size)
            )
            self.inp_conv_pw.append(
                nn.Conv2d(out_ch, out_ch // 4, 1, bias=False)
            )
            self.inp_conv_se.append(PPLCNetV4.SELayer(out_ch // 4))

        if self.intracl:
            self.incl1 = PPOCRv4_det.IntraCLBlock(out_ch // 4, reduce_factor=2)
            self.incl2 = PPOCRv4_det.IntraCLBlock(out_ch // 4, reduce_factor=2)
            self.incl3 = PPOCRv4_det.IntraCLBlock(out_ch // 4, reduce_factor=2)
            self.incl4 = PPOCRv4_det.IntraCLBlock(out_ch // 4, reduce_factor=2)

    def _inp_forward(self, x, idx):
        x = self.inp_conv_dw[idx](x)
        x = self.inp_conv_pw[idx](x)
        if self.shortcut:
            x = x + self.inp_conv_se[idx](x)
        else:
            x = self.inp_conv_se[idx](x)
        return x

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest")  # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")  # 1/4

        p5 = self._inp_forward(in5, 3)
        p4 = self._inp_forward(out4, 2)
        p3 = self._inp_forward(out3, 1)
        p2 = self._inp_forward(out2, 0)

        if self.intracl:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        if self.training:
            return {"fuse": fuse, "aux_p4": out4, "aux_p3": out3, "aux_p2": out2}
        return fuse

    def rep(self):
        if self.is_repped:
            return
        for i in range(len(self.inp_conv_dw)):
            self.inp_conv_dw[i].rep()
        self.is_repped = True


class RepLKPAN(nn.Module):
    """LKPAN with 9x9 convs replaced by DilatedReparamConv."""

    def __init__(self, in_ches, out_ch, intracl=True):
        super().__init__()
        self.out_channels = out_ch
        self.is_repped = False
        self.intracl = intracl

        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        self.pan_head_conv = nn.ModuleList()
        self.pan_lat_conv = nn.ModuleList()

        for i in range(len(in_ches)):
            self.ins_conv.append(
                nn.Conv2d(in_ches[i], out_ch, 1, bias=False)
            )
            self.inp_conv.append(
                DilatedReparamConv(out_ch, out_ch // 4, kernel_size=9)
            )
            if i > 0:
                self.pan_head_conv.append(
                    nn.Conv2d(out_ch // 4, out_ch // 4, 3, padding=1, stride=2, bias=False)
                )
            self.pan_lat_conv.append(
                DilatedReparamConv(out_ch // 4, out_ch // 4, kernel_size=9)
            )

        if self.intracl:
            self.incl1 = PPOCRv4_det.IntraCLBlock(out_ch // 4, reduce_factor=2)
            self.incl2 = PPOCRv4_det.IntraCLBlock(out_ch // 4, reduce_factor=2)
            self.incl3 = PPOCRv4_det.IntraCLBlock(out_ch // 4, reduce_factor=2)
            self.incl4 = PPOCRv4_det.IntraCLBlock(out_ch // 4, reduce_factor=2)

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
        if self.training:
            return {"fuse": fuse, "aux_p4": out4, "aux_p3": out3, "aux_p2": out2}
        return fuse

    def rep(self):
        if self.is_repped:
            return
        for i in range(len(self.inp_conv)):
            self.inp_conv[i].rep()
        for i in range(len(self.pan_lat_conv)):
            self.pan_lat_conv[i].rep()
        self.is_repped = True
