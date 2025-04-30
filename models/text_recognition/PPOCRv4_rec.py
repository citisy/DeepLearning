import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from ..image_classification import PPLCNetV3, PPHGNet
from utils import math_utils, torch_utils, converter
from . import BaseTextRecModel
from .. import activations, bundles
from ..layers import Conv
from ..text_pretrain.transformers import TransformerSequential


class Config(bundles.Config):
    student_backbone = dict(
        name='models.image_classification.PPLCNetV3.Backbone',
        **PPLCNetV3.Config.rec_backbone
    )

    teacher_backbone = dict(
        name='models.image_classification.PPHGNet.Backbone',
        **PPHGNet.Config.rec_small_backbone
    )

    neck = dict(
        depth=2,
        out_ch=120,
        hidden_ch=120,
        kernel_size=[1, 3],
    )
    head = dict(
        out_ch=6625
    )

    default_model = 'student'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            # infer
            'student': dict(
                backbone_config=cls.student_backbone,
                neck_config=cls.neck,
                head_config=cls.head
            ),

            # server infer
            'teacher': dict(
                backbone_config=cls.teacher_backbone,
                neck_config=cls.neck,
                head_config=cls.head
            )
        }


class WeightConverter:

    @staticmethod
    def pre_convert(state_dict):
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

    @staticmethod
    def post_convert(state_dict):
        # note, the official checkpoint is not correct, so we need to fix it
        state_dict['neck.encoder.conv4.conv.weight'] = torch.repeat_interleave(state_dict['neck.encoder.conv4.conv.weight'], 3, 2)
        return state_dict

    neck_convert_dict = {
        'head.ctc_encoder': 'neck',
        'head.ctc_encoder.encoder.svtr_block.{0}.norm1': 'neck.encoder.svtr_block.{0}.attn_res.norm',
        'head.ctc_encoder.encoder.svtr_block.{0}.mixer.qkv': 'neck.encoder.svtr_block.{0}.attn_res.fn.to_qkv',
        'head.ctc_encoder.encoder.svtr_block.{0}.mixer.proj': 'neck.encoder.svtr_block.{0}.attn_res.fn.to_out.linear',
        'head.ctc_encoder.encoder.svtr_block.{0}.norm2': 'neck.encoder.svtr_block.{0}.ff_res.norm',
        'head.ctc_encoder.encoder.svtr_block.{0}.mlp.fc1': 'neck.encoder.svtr_block.{0}.ff_res.fn.0.linear',
        'head.ctc_encoder.encoder.svtr_block.{0}.mlp.fc2': 'neck.encoder.svtr_block.{0}.ff_res.fn.1.linear',
    }

    head_convert_dict = {
        'head.ctc_head.fc': 'head.fc'
    }

    @classmethod
    def from_student(cls, state_dict):
        # https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar
        state_dict = cls.pre_convert(state_dict)

        convert_dict = {
            **PPLCNetV3.WeightConverter.backbone_convert_dict,
            **cls.neck_convert_dict,
            **cls.head_convert_dict
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        state_dict = cls.post_convert(state_dict)
        return state_dict

    @classmethod
    def from_teacher(cls, state_dict):
        # https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_server_train.tar
        state_dict = cls.pre_convert(state_dict)

        convert_dict = {
            **PPHGNet.WeightConverter.backbone_convert_dict,
            **cls.neck_convert_dict,
            **cls.head_convert_dict
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        state_dict = cls.post_convert(state_dict)
        return state_dict


class Model(BaseTextRecModel):
    def __init__(
            self,
            backbone_config=Config.student_backbone, neck_config=Config.neck, head_config=Config.head,
            **kwargs
    ):
        backbone_name = backbone_config.pop('name')
        backbone = converter.DataInsConvert.str_to_instance(backbone_name)(**backbone_config)
        neck = SequenceEncoder(in_ch=backbone.out_channels, **neck_config)
        head = CTCHead(in_ch=neck.out_channels, **head_config)
        super().__init__(
            out_features=head.out_channels,
            backbone=backbone,
            neck=neck,
            head=head,
            **kwargs
        )

    def post_process(self, x):
        probs, preds = torch.max(x, -1)
        words = []
        for b in range(x.shape[0]):
            pred = preds[b]
            diff = torch.diff(pred)
            diff = torch.cat([torch.tensor([-1]).to(diff), diff])
            pred = pred[diff != 0]
            pred = pred[pred != 0]
            chars = [self.id2char[int(i)] for i in pred]
            words.append(''.join(chars))

        return {'pred': words}

    def loss(self, pred_label, true_label):
        raise NotImplemented


class SequenceEncoder(nn.Module):
    def __init__(self, in_ch, **kwargs):
        super(SequenceEncoder, self).__init__()
        self.encoder = EncoderWithSVTR(in_ch, **kwargs)
        self.encoder_reshape = Rearrange('b c 1 w -> b w c')
        self.out_channels = self.encoder.out_channels

    def forward(self, x):
        x = self.encoder(x)
        x = self.encoder_reshape(x)
        return x


class EncoderWithSVTR(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch=64,  # XS
            depth=2,
            hidden_ch=120,
            num_heads=8,
            mlp_ratio=2.0,
            drop_prob=0.1,
            kernel_size=3,
            **kwargs
    ):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 8, kernel_size, bias=False, mode='cna', norm_fn=nn.BatchNorm2d, act=activations.Swish())
        self.conv2 = Conv(in_ch // 8, hidden_ch, 1, bias=False, mode='cna', norm_fn=nn.BatchNorm2d, act=activations.Swish())

        self.svtr_block = TransformerSequential(
            hidden_ch,
            num_heads,
            int(hidden_ch * mlp_ratio),
            drop_prob=drop_prob,
            norm_first=True,
            fn_kwargs=dict(
                separate=False
            ),
            norm_kwargs=dict(
                eps=1e-05
            ),
            num_blocks=depth
        )
        self.norm = nn.LayerNorm(hidden_ch, eps=1e-6)
        self.conv3 = Conv(hidden_ch, in_ch, 1, mode='cna', bias=False, norm_fn=nn.BatchNorm2d, act=activations.Swish())
        # last conv-nxn, the input is concat of input tensor and conv3 output tensor
        self.conv4 = Conv(2 * in_ch, in_ch // 8, 3, mode='cna', bias=False, norm_fn=nn.BatchNorm2d, act=activations.Swish())

        self.conv1x1 = Conv(in_ch // 8, out_ch, 1, mode='cna', bias=False, norm_fn=nn.BatchNorm2d, act=activations.Swish())
        self.out_channels = out_ch

    def forward(self, x):
        # for short cut
        h = x
        # reduce dim
        x = self.conv1(x)
        x = self.conv2(x)
        # SVTR global block
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        x = self.svtr_block(x)

        x = self.norm(x)
        # last stage
        x = x.reshape([-1, H, W, C]).permute(0, 3, 1, 2)
        x = self.conv3(x)
        x = torch.cat((h, x), dim=1)
        x = self.conv1x1(self.conv4(x))

        return x


class CTCHead(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch=6625,
            mid_ch=None,
            **kwargs
    ):
        super().__init__()
        if mid_ch is None:
            self.fc = nn.Linear(in_ch, out_ch)
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, mid_ch),
                nn.Linear(mid_ch, out_ch)
            )

        self.out_channels = out_ch
        self.mid_channels = mid_ch

    def forward(self, x):
        predicts = self.fc(x)
        return predicts
