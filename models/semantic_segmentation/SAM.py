from functools import partial
from typing import List, Type, Optional, Tuple

import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops.boxes import batched_nms

from utils import cv_utils, torch_utils
from .. import bundles
from ..attentions import CrossAttention3D, CrossAttention2D
from ..embeddings import PatchEmbedding
from ..layers import Conv, ConvT, Linear
from ..normalizations import LayerNorm2d
from ..text_pretrain.transformers import TransformerBlock, PositionWiseFeedForward


class Config(bundles.Config):
    vit_h_image_encoder = dict(
        embed_dim=1280,
        depth=32,
        n_heads=16,
        attend_layers=(7, 15, 23, 31),
    )

    vit_l_image_encoder = dict(
        embed_dim=1024,
        depth=24,
        n_heads=16,
        attend_layers=(5, 11, 17, 23),
    )

    vit_b_image_encoder = dict(
        embed_dim=768,
        depth=12,
        n_heads=12,
        attend_layers=(2, 5, 8, 11),
    )

    default_model = 'vit_h'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            'vit_h': dict(
                image_encoder_config=cls.vit_h_image_encoder
            ),

            'vit_l': dict(
                image_encoder_config=cls.vit_h_image_encoder
            ),

            'vit_b': dict(
                image_encoder_config=cls.vit_h_image_encoder
            ),
        }


class WeightConverter:
    @classmethod
    def from_official(cls, state_dict):
        """convert weights from official model to my own model

        Usage:
            .. code-block:: python

                state_dict = torch.load(pretrained_model, map_location=self.device)['state_dict']
                state_dict = WeightConverter.from_official(state_dict)
                Model(...).load_state_dict(state_dict)
        """
        convert_dict = {
            'image_encoder.blocks.{0}.attn.qkv': 'image_encoder.blocks.{0}.attn_res.fn.to_qkv',
            'image_encoder.blocks.{0}.attn.proj': 'image_encoder.blocks.{0}.attn_res.fn.to_out.linear',
            'image_encoder.blocks.{0}.mlp.lin1': 'image_encoder.blocks.{0}.ff_res.fn.0.linear',
            'image_encoder.blocks.{0}.mlp.lin2': 'image_encoder.blocks.{0}.ff_res.fn.1.linear',
            'image_encoder.blocks.{0}.attn.rel_pos_h': 'image_encoder.blocks.{0}.attn_res.fn.attend.rel_pos_h',
            'image_encoder.blocks.{0}.attn.rel_pos_w': 'image_encoder.blocks.{0}.attn_res.fn.attend.rel_pos_w',
            'image_encoder.blocks.{0}.norm1': 'image_encoder.blocks.{0}.attn_res.norm',
            'image_encoder.blocks.{0}.norm2': 'image_encoder.blocks.{0}.ff_res.norm',

            'image_encoder.patch_embed.proj': 'image_encoder.patch_embed.fn.0',
            'image_encoder.neck.0': 'image_encoder.neck.1.conv',
            'image_encoder.neck.1': 'image_encoder.neck.1.norm',
            'image_encoder.neck.2': 'image_encoder.neck.2.conv',
            'image_encoder.neck.3': 'image_encoder.neck.2.norm',

            'prompt_encoder.mask_downscaling.0': 'prompt_encoder.mask_downscaling.0.conv',
            'prompt_encoder.mask_downscaling.1': 'prompt_encoder.mask_downscaling.0.norm',
            'prompt_encoder.mask_downscaling.3': 'prompt_encoder.mask_downscaling.1.conv',
            'prompt_encoder.mask_downscaling.4': 'prompt_encoder.mask_downscaling.1.norm',
            'prompt_encoder.mask_downscaling.6': 'prompt_encoder.mask_downscaling.2.conv',
            'prompt_encoder.pe_layer.positional_encoding_gaussian_matrix': 'prompt_encoder.pe_layer.weights',

            'mask_decoder.transformer.{0}.q_proj': 'mask_decoder.transformer.{0}.to_qkv.0',
            'mask_decoder.transformer.{0}.k_proj': 'mask_decoder.transformer.{0}.to_qkv.1',
            'mask_decoder.transformer.{0}.v_proj': 'mask_decoder.transformer.{0}.to_qkv.2',
            'mask_decoder.transformer.{0}.out_proj': 'mask_decoder.transformer.{0}.to_out.linear',
            'mask_decoder.transformer.layers.{0}.mlp.lin1': 'mask_decoder.transformer.layers.{0}.ff.0.linear',
            'mask_decoder.transformer.layers.{0}.mlp.lin2': 'mask_decoder.transformer.layers.{0}.ff.1.linear',

            'mask_decoder.output_upscaling.0': 'mask_decoder.output_upscaling.0.conv',
            'mask_decoder.output_upscaling.1': 'mask_decoder.output_upscaling.0.norm',
            'mask_decoder.output_upscaling.3': 'mask_decoder.output_upscaling.1.conv',

            'mask_decoder.output_hypernetworks_mlps.{0}.layers.{1}.': 'mask_decoder.output_hypernetworks_mlps.{0}.{1}.linear.',
            'mask_decoder.iou_prediction_head.layers.{0}': 'mask_decoder.iou_prediction_head.{0}.linear',

        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        return state_dict


class Model(nn.Module):
    """
    https://github.com/facebookresearch/segment-anything
    https://arxiv.org/pdf/2304.02643"""

    n_grids = (31, 31)
    points_per_batch = 64
    pred_iou_thresh = 0.88

    mask_threshold: float = 0.0
    stability_score_offset: float = 1.0
    stability_score_thresh: float = 0.95
    box_nms_thresh: float = 0.7

    def __init__(
            self, in_ch=3, input_size=1024, prompt_embed_dim=256,
            image_encoder_config=dict(), prompt_encoder_config=dict(),
            mask_decoder_config=dict(), post_config=dict(),
            **kwargs
    ):
        super().__init__()
        self.__dict__.update(post_config)

        self.input_size = input_size

        self.image_encoder = ImageEncoder(
            input_size=input_size,
            in_ch=in_ch,
            norm_fn=partial(nn.LayerNorm, eps=1e-6),
            out_ch=prompt_embed_dim,
            **image_encoder_config
        )

        image_embedding_size = input_size // self.image_encoder.patch_size
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            image_size=(input_size, input_size),
            **prompt_encoder_config
        )

        self.mask_decoder = MaskDecoder(prompt_embed_dim, **mask_decoder_config)

        self.grid_points = self.make_grid_points()

    def forward(self, x, label_masks=None, **kwargs):
        if self.training:
            raise NotImplemented
        else:
            return self.post_process(x, **kwargs)

    def post_process(self, x, points=None, in_labels=None, effective_areas=None, multimask_output=True, **kwargs):
        """

        Args:
            x:
            points:
                shape: (b, n, 2), 2 gives (x, y), n gives num of grid points
                if none, use average grid points
            in_labels:
                shape: (b, 1), falls in [-1, 0, 1]
                if none, gives in_labels=1
            effective_areas:
                shape: (b, 4), 4 gives (x1, y1, x2, y2)
                if none, use the whole area of images
            multimask_output:
                if True, return all the classes mask
                if False, only return the first mask, it also means that cannot separate overlapping objects
            **kwargs:

        """
        b, c, h, w = x.shape
        features = self.image_encoder(x)

        if points is None:
            if effective_areas is None:
                effective_areas = [(0, w, 0, h)] * b
            points = self.make_points(effective_areas)

        label_masks = []
        for f, p in zip(features, points):
            label_mask = self.post_process_one_image(f[None], p, in_labels, multimask_output)
            label_masks.append(label_mask)

        return torch.stack(label_masks)

    def post_process_one_image(self, features, points, in_labels=None, multimask_output=True):
        masks = []
        boxes = []
        iou_predictions = []
        for i in range(0, len(points), self.points_per_batch):
            in_points = points[i: i + self.points_per_batch]

            in_points = in_points.to(features)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(in_points[:, None, :], in_labels)

            batch_masks, batch_iou_predictions = self.mask_decoder(
                image_embeddings=features,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                image_pe=self.prompt_encoder.dense_pe,
                multimask_output=multimask_output
            )

            batch_masks = F.interpolate(
                batch_masks,
                (self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )

            batch_masks = batch_masks.flatten(0, 1)
            batch_iou_predictions = batch_iou_predictions.flatten(0, 1)

            keep = batch_iou_predictions > self.pred_iou_thresh
            batch_masks, batch_iou_predictions = batch_masks[keep], batch_iou_predictions[keep]

            stability_score = self.calculate_stability_score(batch_masks)
            keep = stability_score >= self.stability_score_thresh
            batch_masks, batch_iou_predictions = batch_masks[keep], batch_iou_predictions[keep]

            batch_masks = batch_masks > self.mask_threshold
            batch_boxes = self.batched_mask_to_box(batch_masks)

            masks.append(batch_masks)
            boxes.append(batch_boxes)
            iou_predictions.append(batch_iou_predictions)

        masks = torch.cat(masks)
        boxes = torch.cat(boxes)
        iou_predictions = torch.cat(iou_predictions)

        keep = batched_nms(
            boxes.float(),
            iou_predictions,
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )

        masks, boxes, iou_predictions = masks[keep], boxes[keep], iou_predictions[keep]
        label_mask = torch.zeros(*masks.shape[1:], dtype=torch.int, device=masks.device)
        for i, mask in enumerate(masks):
            label_mask[mask] = i + 1

        return label_mask

    def make_grid_points(self):
        # n_points = (n_grids[0] + 1) * (n_grids[1] + 1)
        offset_x, offset_y = map(lambda x: 1 / (2 * (x + 1)), self.n_grids)
        points = cv_utils.GridBox.grids_to_points(
            (offset_x, 1 - offset_x, offset_y, 1 - offset_y),
            self.n_grids,
        )
        return points

    def make_points(self, bboxes):
        points = []
        for box in bboxes:
            l, t, r, d = box
            dx, dy = r - l, d - t
            in_points = np.stack([
                l + self.grid_points[:, 0] * dx,
                t + self.grid_points[:, 1] * dy
            ], axis=1)
            points.append(in_points)

        return points

    def calculate_stability_score(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Computes the stability score for a batch of masks. The stability
        score is the IoU between the binary masks obtained by thresholding
        the predicted mask logits at high and low values.
        """
        # One mask is always contained inside the other.
        # Save memory by preventing unnecessary cast to torch.int64
        intersections = (
            (masks > (self.mask_threshold + self.stability_score_offset))
            .sum(-1, dtype=torch.int16)
            .sum(-1, dtype=torch.int32)
        )
        unions = (
            (masks > (self.mask_threshold - self.stability_score_offset))
            .sum(-1, dtype=torch.int16)
            .sum(-1, dtype=torch.int32)
        )
        return intersections / unions

    def batched_mask_to_box(self, masks: torch.Tensor) -> torch.Tensor:
        """copy from https://github.com/facebookresearch/segment-anything

        Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
        an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.

        Args:
            masks (torch.Tensor): (..., h, w)
        Returns:
            box (torch.Tensor): (..., 4)
        """
        # torch.max below raises an error on empty inputs, just skip in this case
        if torch.numel(masks) == 0:
            return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

        # Normalize shape to CxHxW
        shape = masks.shape
        h, w = shape[-2:]
        if len(shape) > 2:
            masks = masks.flatten(0, -3)
        else:
            masks = masks.unsqueeze(0)

        # Get top and bottom edges
        in_height, _ = torch.max(masks, dim=-1)
        in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
        bottom_edges, _ = torch.max(in_height_coords, dim=-1)
        in_height_coords = in_height_coords + h * (~in_height)
        top_edges, _ = torch.min(in_height_coords, dim=-1)

        # Get left and right edges
        in_width, _ = torch.max(masks, dim=-2)
        in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
        right_edges, _ = torch.max(in_width_coords, dim=-1)
        in_width_coords = in_width_coords + w * (~in_width)
        left_edges, _ = torch.min(in_width_coords, dim=-1)

        # If the mask is empty the right edge will be to the left of the left edge.
        # Replace these boxes with [0, 0, 0, 0]
        empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
        out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
        out = out * (~empty_filter).unsqueeze(-1)

        # Return to original shape
        if len(shape) > 2:
            out = out.reshape(*shape[:-2], 4)
        else:
            out = out[0]

        return out


class Model4Export(Model):
    """for exporting to onnx, torchscript, etc."""
    n_grids = (7, 7)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mean = torch.tensor([123.675, 116.28, 103.53])[None, :, None, None]
        std = torch.tensor([58.395, 57.12, 57.375])[None, :, None, None]
        self.register_buffer('mean', mean, persistent=False)
        self.register_buffer('std', std, persistent=False)

        layers = torch_utils.ModuleManager.get_module_by_key(self, include=[Rearrange])
        for current_m, name, full_name in layers:
            old = getattr(current_m, name)
            setattr(current_m, name, torch.jit.script(old))

    def forward(self, x, points, in_labels=None, multimask_output=True):
        x = self.pre_process(x)
        features = self.image_encoder(x)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(points[:, None, :], in_labels)

        batch_masks, batch_iou_predictions = self.mask_decoder(
            image_embeddings=features,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            image_pe=self.prompt_encoder.dense_pe,
            multimask_output=multimask_output
        )

        batch_masks = F.interpolate(
            batch_masks,
            (self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )

        batch_masks = batch_masks.flatten(0, 1)
        batch_iou_predictions = batch_iou_predictions.flatten(0, 1)

        keep = batch_iou_predictions > self.pred_iou_thresh
        batch_masks, batch_iou_predictions = batch_masks[keep], batch_iou_predictions[keep]

        stability_score = self.calculate_stability_score(batch_masks)
        keep = stability_score >= self.stability_score_thresh
        batch_masks, batch_iou_predictions = batch_masks[keep], batch_iou_predictions[keep]

        batch_masks = batch_masks > self.mask_threshold
        return batch_masks, batch_iou_predictions

    def pre_process(self, x):
        """for faster infer, use uint8 input and fp32 to output"""
        x = x.to(dtype=torch.float32)  # cannot use fp16
        x = (x - self.mean) / self.std
        return x


class ImageEncoder(nn.Module):
    """
    This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py
    """

    def __init__(
            self,
            input_size: int = 1024,
            patch_size: int = 16,
            embed_dim: int = 1280,
            depth: int = 32,
            n_heads: int = 16,
            ff_ratio: int = 4,
            in_ch: int = 3,
            out_ch: int = 256,
            qkv_bias: bool = True,
            norm_fn: Type[nn.Module] = nn.LayerNorm,
            use_abs_pos: bool = True,
            window_size: int = 14,
            attend_layers: Tuple[int, ...] = (7, 15, 23, 31),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.patch_embed = PatchEmbedding(
            embed_dim, patch_size, in_ch=in_ch, bias=True, out_ndim=4
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, input_size // patch_size, input_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            window_size_ = window_size if i not in attend_layers else 0
            block = TransformerBlock(
                embed_dim, n_heads, embed_dim * ff_ratio,
                norm_first=True,
                norm_fn=norm_fn,
                attention_fn=WindowsAttention,
                fn_kwargs=dict(
                    use_conv=False,
                    separate=False,
                    window_size=window_size_,
                    qkv_fn_kwargs=dict(
                        bias=qkv_bias,
                    ),
                    out_fn_kwargs=dict(
                        bias=qkv_bias,
                    ),
                ),
                attend_fn=RelScaleAttend,
                attend_fn_kwargs=dict(
                    image_size=(input_size // patch_size, input_size // patch_size) if window_size_ == 0 else (window_size_, window_size_),
                    n_heads=n_heads,
                    head_dim=embed_dim // n_heads,
                )
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            Rearrange('b h w c -> b c h w'),

            Conv(
                embed_dim, out_ch, 1, bias=False,
                mode='cn',
                norm=LayerNorm2d(out_ch)
            ),
            Conv(
                out_ch, out_ch, 3, bias=False,
                mode='cn',
                norm=LayerNorm2d(out_ch)
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.neck(x)

        return x


class PromptEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            image_embedding_size: Tuple[int, int],
            image_size: Tuple[int, int],
            mask_in_ch: int = 16,
            act_fn: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = RandomPositionEmbedding(embed_dim // 2, size=image_embedding_size)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        self.point_embeddings = nn.ModuleList([nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)])
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            Conv(1, mask_in_ch // 4, 2, 2, mode='cna', norm=LayerNorm2d(mask_in_ch // 4), act=act_fn()),
            Conv(mask_in_ch // 4, mask_in_ch, 2, 2, mode='cna', norm=LayerNorm2d(mask_in_ch), act=act_fn()),
            Conv(mask_in_ch, embed_dim, 1, mode='c'),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)
        self.register_buffer('_padding_embedding', torch.zeros((1, embed_dim)), persistent=False)

    @property
    def dense_pe(self):
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
            self,
            points: torch.Tensor,
            labels: torch.Tensor,
            pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        point_embedding = self.pe_layer.forward_with_coords(points, self.image_size)

        if labels is not None:
            point_embedding[labels == -1] = 0.0
            point_embedding[labels == -1] += self.not_a_point_embed.weight
            for i in range(2):
                point_embedding[labels == i] += self.point_embeddings[i].weight

        else:
            point_embedding += self.point_embeddings[1].weight

        if pad:
            # note, suit for exporting
            padding_embedding = torch.repeat_interleave(self._padding_embedding[None], int(point_embedding.shape[0]), dim=0)
            point_embedding = torch.cat([point_embedding, padding_embedding], dim=1)

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
            self,
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            boxes: Optional[torch.Tensor],
            masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def forward(self, points=None, in_labels=None, boxes=None, masks=None) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = []

        if points is not None:
            point_embeddings = self._embed_points(points, in_labels, pad=(boxes is None))
            sparse_embeddings.append(point_embeddings)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings.append(box_embeddings)

        # note, suit for exporting
        if len(sparse_embeddings) == 0:
            sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self.device)
        elif len(sparse_embeddings) == 1:
            sparse_embeddings = sparse_embeddings[0]
        else:
            sparse_embeddings = torch.cat(sparse_embeddings, dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class RandomPositionEmbedding(nn.Module):
    """3-d embedding
    emb_{i} = sin{2pi * (2x - 1) * scale * r}
    emb_{2i} = cos{2pi * (2x - 1) * scale * r}
    where, x is seq vec, r ~ N(0,1)
    """

    def __init__(self, num_pos_feats=64, scale=None, size=None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.size = size
        self.register_buffer("weights", scale * torch.randn((2, num_pos_feats)))
        self.register_buffer("coords", self.make_coords(size), persistent=False)

    def make_coords(self, size):
        h, w = size
        grid = torch.ones((h, w))
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        coords = torch.stack([x_embed, y_embed], dim=-1)
        return coords

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.weights
        coords = 2 * torch.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        # note, suit for exporting
        if size[0] == self.size[0] and size[1] == self.size[1]:
            coords = self.coords
        else:
            coords = self.make_coords(size).to(self.weights.device)
        pe = self._pe_encoding(coords)
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
            self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)  # B x N x C


class WindowsAttention(CrossAttention3D):
    def __init__(self, window_size=0, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def forward(self, q, k=None, v=None, **attend_kwargs):
        if self.window_size > 0:
            assert k is None
            assert v is None
            H, W = q.shape[1], q.shape[2]
            q, pad_hw = self.window_partition(q)

        x = super().forward(q, k, v, **attend_kwargs)
        if self.window_size > 0:
            x = self.window_unpartition(x, pad_hw, (H, W))
        return x

    def window_partition(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Partition into non-overlapping windows with padding if needed.
        """
        B, H, W, C = x.shape

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        x = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows, (Hp, Wp)

    def window_unpartition(self, windows: torch.Tensor, pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> torch.Tensor:
        """
        Window unpartition into original sequences and removing padding.
        """
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp // self.window_size // self.window_size)
        x = windows.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

        if Hp > H or Wp > W:
            x = x[:, :H, :W, :].contiguous()
        return x


class RelScaleAttend(nn.Module):
    def __init__(self, image_size, n_heads, head_dim, **kwargs):
        super().__init__()
        self.image_size = image_size
        self.view_in = Rearrange('b 1 s (n d) -> (b n) s d', n=n_heads)  # s=h*w
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * image_size[0] - 1, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * image_size[1] - 1, head_dim))
        self.view_out = Rearrange('(b n) s d -> b 1 s (n d)', n=n_heads)  # s=h*w

    def forward(self, q, k, v, **kwargs):
        """
        in(q|k|v): (b n s d) or (b*n s d)
        out(attn): (b n s d) or (b*n s d)
        """
        q, k, v = [self.view_in(x).contiguous() for x in (q, k, v)]
        scale = q.shape[-1] ** -0.5
        # similarity -> (..., i, j), usually i=j=s
        # sim = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scale
        sim = torch.matmul(q, k.transpose(-2, -1)) * scale
        # sim = mask(sim, attention_mask, use_min=use_min)

        sim = self.add_decomposed_rel_pos(sim, q, self.image_size, self.image_size)

        attn = F.softmax(sim, dim=-1)

        # attn = einsum('... i j, ... j d -> ... i d', attn, v)
        attn = torch.matmul(attn, v)
        attn = self.view_out(attn)

        return attn

    def add_decomposed_rel_pos(
            self,
            attn: torch.Tensor,
            q: torch.Tensor,
            q_size: Tuple[int, int],
            k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
        """
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = self.get_rel_pos(q_h, k_h, self.rel_pos_h)
        Rw = self.get_rel_pos(q_w, k_w, self.rel_pos_w)

        B, _, dim = q.shape
        r_q = q.reshape(B, q_h, q_w, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

        attn = (
                attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        ).view(B, q_h * q_w, k_h * k_w)

        return attn

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.
        Args:
            q_size (int): size of query q.
            k_size (int): size of key k.
            rel_pos (Tensor): relative position embeddings (L, C).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos if needed.
        if rel_pos.shape[0] != max_rel_dist:
            # Interpolate rel pos.
            rel_pos_resized = F.interpolate(
                rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                size=max_rel_dist,
                mode="linear",
            )
            rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        else:
            rel_pos_resized = rel_pos

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]


class MaskDecoder(nn.Module):
    """
    Lightly adapted from
    https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
    """

    def __init__(
            self,
            transformer_dim: int,
            num_multimask_outputs: int = 3,
            act_fn: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            ff_hidden_size=2048,
            n_heads=8,
        )

        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            ConvT(transformer_dim, transformer_dim // 4, 2, 2, mode='cna', norm=LayerNorm2d(transformer_dim // 4), act=act_fn()),
            ConvT(transformer_dim // 4, transformer_dim // 8, 2, 2, mode='ca', act=act_fn()),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MaskDecoderHead(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MaskDecoderHead(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, int(tokens.shape[0]), dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, int(tokens.shape[0]), dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class TwoWayTransformer(nn.Module):
    def __init__(
            self,
            depth: int,
            embedding_dim: int,
            n_heads: int,
            ff_hidden_size: int,
            act_fn: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    n_heads=n_heads,
                    ff_hidden_size=ff_hidden_size,
                    act_fn=act_fn,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = CrossAttention2D(
            query_dim=embedding_dim,
            context_dim=embedding_dim,
            model_dim=embedding_dim // attention_downsample_rate,
            n_heads=n_heads
        )

        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
            self,
            image_embedding: Tensor,
            image_pe: Tensor,
            point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layer norm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe

        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            n_heads: int,
            ff_hidden_size: int = 2048,
            act_fn: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
            skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = CrossAttention2D(model_dim=embedding_dim, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = CrossAttention2D(
            query_dim=embedding_dim,
            context_dim=embedding_dim,
            model_dim=embedding_dim // attention_downsample_rate,
            n_heads=n_heads
        )

        self.norm2 = nn.LayerNorm(embedding_dim)

        self.ff = PositionWiseFeedForward(embedding_dim, ff_hidden_size, act=act_fn())
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)

        self.cross_attn_image_to_token = CrossAttention2D(
            query_dim=embedding_dim,
            context_dim=embedding_dim,
            model_dim=embedding_dim // attention_downsample_rate,
            n_heads=n_heads
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
            self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.ff(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class MaskDecoderHead(nn.Sequential):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        h = [hidden_dim] * (num_layers - 1)

        layers = []
        for i, (n, k) in enumerate(zip([input_dim] + h, h + [output_dim])):
            layers.append(Linear(n, k, mode='la', act=nn.ReLU(), is_act=i < num_layers - 1))

        if sigmoid_output:
            layers.append(nn.Sigmoid())

        super().__init__(*layers)
