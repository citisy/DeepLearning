import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from ..layers import Conv, Linear, ConvInModule, OutModule
from . import BaseTextRecModel
from data_parse.nl_data_parse.pre_process import decoder

CONV_MIX = 0
GLOBAL_MIX = 1
LOCAL_MIX = 2

in_module_config = dict(out_ch=64, drop_prob=0, n_layer=2)
backbone_config = dict(in_ch_list=(64, 128, 256), num_heads_list=(2, 4, 8), mixer_list=None, n_layers=(3, 6, 3))


class Model(BaseTextRecModel):
    """refer to: [SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)"""

    def __init__(self, input_size, in_ch=3, neck_out_features=192, max_seq_len=25, out_features=None,
                 in_module_config=in_module_config, backbone_config=backbone_config, **kwargs):
        in_module = Embedding(in_ch, input_size, **in_module_config)
        backbone = Backbone(in_module.output_size, neck_out_features, max_seq_len, **backbone_config)
        super().__init__(
            in_module=in_module,
            backbone=backbone,
            neck=Rearrange('b c h w -> (h w) b c'),
            head=Head(neck_out_features, out_features + 1),
            input_size=input_size,
            in_ch=in_ch,
            neck_out_features=neck_out_features,
            max_seq_len=max_seq_len,
            out_features=out_features,
            **kwargs
        )

    def post_process(self, x):
        x = x.permute(1, 0, 2)
        preds, probs = decoder.beam_search(x, beam_size=10)
        words = []
        for b in range(x.shape[0]):
            seq = {}
            for pred, prob in zip(preds[b], probs[b]):
                # note that, filter the duplicate chars can raise the accuracy obviously,
                # but it would filter the right result while there are duplicate chars in the true labels
                diff = torch.diff(pred)
                diff = torch.cat([torch.tensor([-1]).to(diff), diff])
                pred = pred[diff != 0]
                pred = pred[pred != 0]
                pred = tuple(pred)
                seq[pred] = torch.log(torch.exp(prob) + (torch.exp(seq[pred]) if pred in seq else 0))

            chars = max(seq.items(), key=lambda x: x[1])[0]
            chars = [self.id2char[int(c)] for c in chars]
            words.append(''.join(chars))

        return {'pred': words}


class Embedding(nn.Module):
    def __init__(self, in_ch, input_size, out_ch=64, drop_prob=0, n_layer=2):
        super().__init__()
        if n_layer == 2:
            layers = [
                Conv(in_ch, out_ch // 2, 3, s=2, act=nn.GELU()),
                Conv(out_ch // 2, out_ch, 3, s=2, act=nn.GELU())
            ]
        elif n_layer == 3:
            layers = [
                Conv(in_ch, out_ch // 4, 3, s=2, act=nn.GELU()),
                Conv(out_ch // 4, out_ch // 2, 3, s=2, act=nn.GELU()),
                Conv(out_ch // 2, out_ch, 3, s=2, act=nn.GELU())
            ]
        else:
            raise

        layers.append(Rearrange('b c h w -> b (h w) c'))
        self.patch_embed = nn.Sequential(*layers)

        self.output_size = (input_size[0] // (2 ** n_layer), input_size[1] // (2 ** n_layer))
        self.pos_embed = nn.Parameter(torch.zeros([1, self.output_size[0] * self.output_size[1], out_ch], dtype=torch.float32), requires_grad=True)
        truncated_normal_(self.pos_embed)

        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.drop(x)
        return x


class Backbone(nn.Sequential):
    def __init__(self, input_size, out_ch, max_seq_len, in_ch_list=(64, 128, 256), num_heads_list=(2, 4, 8), mixer_list=None, n_layers=(3, 6, 3)):
        mixer_list = mixer_list or [LOCAL_MIX] * 6 + [GLOBAL_MIX] * 6
        w, h = input_size
        layers = []
        n = 0
        dpr = np.linspace(0, 0.1, sum(n_layers))
        for i, n_layer in enumerate(n_layers):
            for j in range(n_layer):
                layers.append(MixingBlock(in_ch_list[i], (w, h), num_heads_list[i], mixer_list[n + j], drop_path_prob=dpr[n + j]))
            n += n_layer

            if i < len(n_layers) - 1:  # not the last layer
                layers.append(Merging(in_ch_list[i], in_ch_list[i + 1], (w, h)))
                h //= 2

        layers.append(Rearrange('b (h w) c -> b c h w', h=h, w=w))
        layers.append(Combing(in_ch_list[-1], out_ch, max_seq_len))

        super().__init__(*layers)

    def initialize_layers(self):
        # note that, it is very useful to raise the score
        self.apply(self._initialize_layers)

    def _initialize_layers(self, m):
        if isinstance(m, nn.Linear):
            truncated_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class MixingBlock(nn.Module):
    def __init__(self, in_ch, input_size, num_heads, mixer, scale_ratio=4., drop_prob=0., drop_path_prob=0.):
        super().__init__()
        hidden_ch = int(in_ch * scale_ratio)
        self.drop_path = DropPath(drop_path_prob)
        self.mixer = Mixing(in_ch, input_size, num_heads, mixer)
        self.norm1 = nn.LayerNorm(in_ch)
        self.mlp = nn.Sequential(
            Linear(in_ch, hidden_ch, mode='lad', act=nn.GELU(), drop_prob=drop_prob),
            Linear(hidden_ch, in_ch, mode='ld', drop_prob=drop_prob)
        )
        self.norm2 = nn.LayerNorm(in_ch)

    def forward(self, x):
        x = self.norm1(x + self.drop_path(self.mixer(x)))
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


class DropPath(nn.Module):

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = torch.tensor(1 - self.drop_prob).to(device=x.device)
        shape = (x.size()[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype).to(device=x.device)
        random_tensor = torch.floor(random_tensor)
        output = torch.div(x, keep_prob) * random_tensor
        return output


class Mixing(nn.Module):
    def __init__(self, in_ch, input_size, num_heads, mixer=GLOBAL_MIX, local_k=(11, 7), drop_prob=0.):
        super().__init__()
        self.mixer = mixer
        w, h = input_size

        if mixer == CONV_MIX:
            self.mixing = nn.Sequential(
                Rearrange('b (h w) c -> b c h w', h=h, w=w),
                Conv(in_ch, in_ch, 3, 1, groups=num_heads, mode='c'),
                Rearrange('b c h w -> b (h w) c'),
            )
        else:  # attention layer
            self.num_heads = num_heads
            head_dim = in_ch // num_heads
            self.scale = head_dim ** -0.5

            self.qkv = nn.Linear(in_ch, in_ch * 3, bias=True)
            self.attn_drop = nn.Dropout(drop_prob)

            self.proj = nn.Linear(in_ch, in_ch)
            self.proj_drop = nn.Dropout(drop_prob)

            if mixer == LOCAL_MIX:
                wk, hk = local_k
                mask = torch.full([h * w, h + hk - 1, w + wk - 1], -float('inf'), dtype=torch.float32)
                # only focus on wk * hk areas
                for j in range(h):
                    for i in range(w):
                        mask[j * w + i, j:j + hk, i:i + wk] = 0.
                mask = torch.flatten(mask[:, hk // 2:h + hk // 2, wk // 2:w + wk // 2], 1)
                self.mask = nn.Parameter(mask[None, None], requires_grad=False)

    def forward(self, x):
        if self.mixer == CONV_MIX:
            x = self.mixing(x)
        else:
            _, n, c = x.size()
            qkv = self.qkv(x)
            qkv = qkv.reshape((-1, n, 3, self.num_heads, c // self.num_heads))
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
            attn = (q.matmul(k.permute(0, 1, 3, 2)))

            if self.mixer == LOCAL_MIX:
                attn += self.mask

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)

            x = (attn.matmul(v)).permute(0, 2, 1, 3).reshape((-1, n, c))
            x = self.proj(x)
            x = self.proj_drop(x)

        return x


class Merging(nn.Module):
    def __init__(self, in_ch, out_ch, input_size, is_pool=False):
        super().__init__()
        self.is_pool = is_pool
        w, h = input_size
        self.c = Rearrange('b (h w) c -> b c h w', h=h, w=w)
        if is_pool:
            self.avg_pool = nn.AvgPool2d(kernel_size=(3, 5), stride=(2, 1), padding=(1, 2))
            self.max_pool = nn.MaxPool2d(kernel_size=(3, 5), stride=(2, 1), padding=(1, 2))
            self.proj = nn.Linear(in_ch, out_ch)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=(2, 1), padding=1)

        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        x = self.c(x)
        if self.is_pool:
            x1 = self.avg_pool(x)
            x2 = self.max_pool(x)
            x = (x1 + x2) * 0.5
            x = self.proj(x.flatten(2).permute(0, 2, 1))
        else:
            x = self.conv(x)
            x = x.flatten(2).permute(0, 2, 1)

        x = self.norm(x)
        return x


class Combing(nn.Sequential):
    def __init__(self, in_ch, out_ch, out_w, drop_prob=0.):
        super().__init__(
            nn.AdaptiveAvgPool2d((1, out_w)),
            Conv(in_ch, out_ch, 1, p=0, mode='cad', act=nn.Hardswish(), drop_prob=drop_prob)
        )


class Head(nn.Sequential):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        super().__init__(nn.Linear(self.in_features, self.out_features))

    def initialize_layers(self):
        self.apply(self._initialize_layers)

    def _initialize_layers(self, m):
        if isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(self.in_features * 1.0)
            nn.init.uniform_(m.weight, -stdv, stdv)
            nn.init.uniform_(m.bias, -stdv, stdv)


def truncated_normal_(tensor, mean=0, std=0.02):
    with torch.no_grad():
        size = tensor.size()
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor
