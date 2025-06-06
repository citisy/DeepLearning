import torch
from torch import nn
from torch.nn import functional as F

from utils import torch_utils
from .. import bundles, activations, attentions
from ..image_classification.ViT import VisionEmbedding
from ..text_pretrain.transformers import DecoderEmbedding, TransformerSequential


class Config(bundles.Config):
    default_model = 'openai-vit-base-patch32'

    openai_model = dict(
        logit_scale_init_value=2.6592
    )

    openai_vision_base = dict(
        output_size=768,
        hidden_size=768,
        image_size=224,
        num_attention_heads=12,
        num_hidden_layers=12
    )

    openai_vision_base_patch32 = dict(
        **openai_vision_base,
        patch_size=32,
    )

    openai_vision_large = dict(
        output_size=768,
        hidden_size=1024,
        image_size=224,
        num_attention_heads=16,
        num_hidden_layers=24
    )

    openai_vision_large_patch14 = dict(
        **openai_vision_large,
        patch_size=14,
    )

    openai_text_base = dict(
        output_size=512,
        hidden_size=512,
        vocab_size=49408,
        max_seq_len=77,
        num_attention_heads=8,
        num_hidden_layers=12
    )

    openai_text_large = dict(
        output_size=768,
        hidden_size=768,
        vocab_size=49408,
        max_seq_len=77,
        num_attention_heads=12,
        num_hidden_layers=12
    )

    laion_model = dict(
        logit_scale_init_value=2.6592
    )

    laion_vision_H_14 = dict(
        output_size=1024,
        hidden_size=1280,
        image_size=224,
        num_attention_heads=16,
        num_hidden_layers=32,
        ff_ratio=4.,
        patch_size=14,
        act_type='GELU',
        separate=False
    )

    laion_vision_bigG_14 = dict(
        output_size=1280,
        hidden_size=1664,
        image_size=224,
        num_attention_heads=16,
        num_hidden_layers=48,
        ff_ratio=4.9231,
        patch_size=14,
        act_type='GELU',
        separate=False
    )

    laion_text_H_14 = dict(
        output_size=1024,
        hidden_size=1024,
        vocab_size=49408,
        max_seq_len=77,
        num_attention_heads=16,
        num_hidden_layers=24,
        act_type='GELU',
        separate=False
    )

    laion_text_bigG_14 = dict(
        output_size=1280,
        hidden_size=1280,
        vocab_size=49408,
        max_seq_len=77,
        num_attention_heads=20,
        num_hidden_layers=32,
        act_type='GELU',
        separate=False
    )

    @classmethod
    def make_full_config(cls) -> dict:
        config_dict = {
            'openai-vit-base-patch32': dict(
                vision_config=cls.openai_vision_base_patch32,
                text_config=cls.openai_text_large,
                **cls.openai_model
            ),

            'openai-vit-large-patch14': dict(
                vision_config=cls.openai_vision_large_patch14,
                text_config=cls.openai_text_large,
                **cls.openai_model
            ),

            'laion-ViT-H-14': dict(
                vision_config=cls.laion_vision_H_14,
                text_config=cls.laion_text_H_14,
                **cls.laion_model
            ),

            'laion-ViT-bigG-14': dict(
                vision_config=cls.laion_vision_bigG_14,
                text_config=cls.laion_text_bigG_14,
                **cls.laion_model
            )
        }
        return config_dict


class WeightLoader(bundles.WeightLoader):
    @classmethod
    def auto_download(cls, save_path, version='openai', **kwargs):
        if version == 'openai':
            return cls.download_from_openai(save_path, **kwargs)
        elif version == 'laion':
            return cls.download_from_laion(save_path, **kwargs)
        else:
            raise f'Dont support version: {version}'

    @staticmethod
    def download_from_openai(save_path, **kwargs):
        """auto download weight auto from transformers
        refer to: https://huggingface.co/openai
        """
        from transformers import CLIPModel

        model = CLIPModel.from_pretrained(save_path, num_labels=2)
        state_dict = model.state_dict()
        return state_dict

    @staticmethod
    def download_from_laion(save_path, arch='ViT-H-14', **kwargs):
        """auto download weight auto from open_clip
        refer to: https://huggingface.co/laion
        """
        import open_clip  # pip install open-clip-torch~=2.20.0
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=save_path)
        state_dict = model.state_dict()
        return state_dict


class WeightConverter:
    @classmethod
    def from_hf(cls, state_dict, version='openai'):
        """convert weights from huggingface model to my own model

        Usage:
            .. code-block:: python

                state_dict = WeightLoader.from_hf(...)
                state_dict = WeightConverter.from_hf(state_dict)
                Model(...).load_state_dict(state_dict)

        """
        if version == 'openai':
            return cls.from_openai(state_dict)
        elif version == 'laion':
            return cls.from_laion(state_dict)
        else:
            raise f'Dont support version: {version}'

    openai_convert_dict = {
        'vision_model.embeddings.patch_embedding': 'vision_model.embedding.patch.fn.0',
        'vision_model.embeddings.class_embedding': 'vision_model.embedding.cls',
        'vision_model.embeddings.position_embedding': 'vision_model.embedding.position',
        'vision_model.pre_layrnorm': 'vision_model.norm1',  # a stupid bug for the var name spelling
        'vision_model.post_layernorm': 'vision_model.norm2',

        '{1}.encoder.layers.{0}.self_attn.q_proj': '{1}.encoder.{0}.attn_res.fn.to_qkv.0',
        '{1}.encoder.layers.{0}.self_attn.k_proj': '{1}.encoder.{0}.attn_res.fn.to_qkv.1',
        '{1}.encoder.layers.{0}.self_attn.v_proj': '{1}.encoder.{0}.attn_res.fn.to_qkv.2',
        '{1}.encoder.layers.{0}.self_attn.out_proj': '{1}.encoder.{0}.attn_res.fn.to_out.linear',
        '{1}.encoder.layers.{0}.layer_norm1': '{1}.encoder.{0}.attn_res.norm',
        '{1}.encoder.layers.{0}.mlp.fc1': '{1}.encoder.{0}.ff_res.fn.0.linear',
        '{1}.encoder.layers.{0}.mlp.fc2': '{1}.encoder.{0}.ff_res.fn.1.linear',
        '{1}.encoder.layers.{0}.layer_norm2': '{1}.encoder.{0}.ff_res.norm',

        'text_model.embeddings.token_embedding': 'text_model.embedding.token',
        'text_model.embeddings.position_embedding': 'text_model.embedding.position',
        'text_model.final_layer_norm': 'text_model.norm',

        'visual_projection': 'vision_model.proj',
        'text_projection': 'text_model.proj'
    }

    @classmethod
    def from_openai(cls, state_dict):
        state_dict = torch_utils.Converter.convert_keys(state_dict, cls.openai_convert_dict)
        return state_dict

    laion_convert_dict = {
        'visual.conv1': 'vision_model.embedding.patch.fn.0',
        'visual.class_embedding': 'vision_model.embedding.cls',
        'visual.positional_embedding': 'vision_model.embedding.position.weight',
        'visual.ln_pre': 'vision_model.norm1',
        'visual.ln_post': 'vision_model.norm2',

        'visual.transformer.resblocks.{0}.ln_1': 'vision_model.encoder.{0}.attn_res.norm',
        'visual.transformer.resblocks.{0}.ln_2': 'vision_model.encoder.{0}.ff_res.norm',
        'visual.transformer.resblocks.{0}.attn.in_proj_weight': 'vision_model.encoder.{0}.attn_res.fn.to_qkv.weight',
        'visual.transformer.resblocks.{0}.attn.in_proj_bias': 'vision_model.encoder.{0}.attn_res.fn.to_qkv.bias',
        'visual.transformer.resblocks.{0}.attn.out_proj': 'vision_model.encoder.{0}.attn_res.fn.to_out.linear',
        'visual.transformer.resblocks.{0}.mlp.c_fc': 'vision_model.encoder.{0}.ff_res.fn.0.linear',
        'visual.transformer.resblocks.{0}.mlp.c_proj': 'vision_model.encoder.{0}.ff_res.fn.1.linear',

        'transformer.resblocks.{0}.ln_1': 'text_model.encoder.{0}.attn_res.norm',
        'transformer.resblocks.{0}.ln_2': 'text_model.encoder.{0}.ff_res.norm',
        'transformer.resblocks.{0}.attn.in_proj_weight': 'text_model.encoder.{0}.attn_res.fn.to_qkv.weight',
        'transformer.resblocks.{0}.attn.in_proj_bias': 'text_model.encoder.{0}.attn_res.fn.to_qkv.bias',
        'transformer.resblocks.{0}.attn.out_proj': 'text_model.encoder.{0}.attn_res.fn.to_out.linear',
        'transformer.resblocks.{0}.mlp.c_fc': 'text_model.encoder.{0}.ff_res.fn.0.linear',
        'transformer.resblocks.{0}.mlp.c_proj': 'text_model.encoder.{0}.ff_res.fn.1.linear',

        'token_embedding': 'text_model.embedding.token',
        'positional_embedding': 'text_model.embedding.position.weight',
        'ln_final': 'text_model.norm',

        'visual.proj': 'vision_model.proj.weight',
        'text_projection': 'text_model.proj.weight',

    }

    @classmethod
    def from_laion(cls, state_dict):
        state_dict = torch_utils.Converter.convert_keys(state_dict, cls.laion_convert_dict)

        for k in ['vision_model.proj.weight', 'text_model.proj.weight']:
            state_dict[k] = state_dict[k].t()

        return state_dict


class Model(nn.Module):
    def __init__(self, logit_scale_init_value=2.6592,
                 vision_config=Config.openai_vision_base_patch32,
                 text_config=Config.openai_text_base,
                 **kwargs
                 ):
        super().__init__()
        self.vision_model = VisionTransformer(**vision_config)
        self.text_model = TextTransformer(**text_config)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

    def forward(self, images, text_ids, attention_mask=None):
        vision_outputs = self.vision_model(images)['pooled_output']
        text_outputs = self.text_model(text_ids, attention_mask=attention_mask)['pooled_output']

        logits_per_text, logits_per_image = self.head(vision_outputs, text_outputs)

        loss = None
        if self.training:
            loss = self.loss(logits_per_text)

        return dict(
            logits_per_text=logits_per_text,
            logits_per_image=logits_per_image,
            loss=loss
        )

    def head(self, vision_outputs, text_outputs):
        image_embeds = vision_outputs / vision_outputs.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = logit_scale * text_embeds @ image_embeds.T
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        return logits_per_text, logits_per_image

    def loss(self, similarity):
        true = torch.arange(len(similarity), device=similarity.device)
        caption_loss = F.cross_entropy(similarity, true)
        image_loss = F.cross_entropy(similarity.t(), true)
        return (caption_loss + image_loss) / 2.0


class VisionModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.vision_model = VisionTransformer(**kwargs)

    def forward(self, image):
        return self.vision_model(image)


class TextModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.text_model = TextTransformer(**kwargs)

    def forward(self, text_ids, **kwargs):
        return self.text_model(text_ids, **kwargs)


class VisionTransformer(nn.Module):
    def __init__(
            self, image_size, hidden_size, patch_size, output_size=None,
            num_attention_heads=12, num_hidden_layers=12, ff_ratio=4.0,
            is_proj=True, separate=True,
            act_type='FasterGELU', attend_type='FlashAttend',
            use_checkpoint=True
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.embedding = VisionEmbedding(hidden_size, image_size, patch_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.encoder = TransformerSequential(
            hidden_size, num_attention_heads, int(hidden_size * ff_ratio), norm_first=True,
            attend_fn=attentions.make_attend_fn.get(attend_type),
            fn_kwargs=dict(separate=separate),
            ff_kwargs=dict(act=activations.make_act_fn.get(act_type)()),
            num_blocks=num_hidden_layers,
            use_checkpoint=use_checkpoint
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        if is_proj:
            self.proj = nn.Linear(hidden_size, output_size, bias=False)
        else:
            self.proj = nn.Identity()

    def backbone(self, image):
        x = self.embedding(image)
        x = self.norm1(x)
        x = self.encoder(x)
        return x

    def neck(self, x):
        h = self.norm2(x)
        return h

    def head(self, h):
        pooled_output = h[:, 0, :]
        pooled_output = self.proj(pooled_output)
        return pooled_output

    def forward(self, images):
        x = self.backbone(images)
        h = self.neck(x)
        pooled_output = self.head(h)
        return dict(
            hidden_state=x,
            pooled_output=pooled_output
        )


class TextTransformer(nn.Module):
    def __init__(
            self, vocab_size, hidden_size, output_size=None,
            max_seq_len=77, num_attention_heads=8, num_hidden_layers=12, ff_ratio=4.0,
            is_proj=True, separate=True,
            act_type='FasterGELU', attend_type='FlashAttend',
            use_checkpoint=True
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.embedding = DecoderEmbedding(vocab_size, hidden_size, max_seq_len=max_seq_len)
        self.encoder = TransformerSequential(
            hidden_size, num_attention_heads, int(hidden_size * ff_ratio), norm_first=True,
            attend_fn=attentions.make_attend_fn.get(attend_type),
            fn_kwargs=dict(separate=separate),
            ff_kwargs=dict(act=activations.make_act_fn.get(act_type)()),
            num_blocks=num_hidden_layers,
            use_checkpoint=use_checkpoint
        )
        self.norm = nn.LayerNorm(hidden_size)
        if is_proj:
            self.proj = nn.Linear(hidden_size, output_size, bias=False)
        else:
            self.proj = nn.Identity()

    def backbone(self,
                 sequence=None,
                 attention_mask=None,
                 **encoder_kwargs):
        x = self.embedding(sequence)

        causal_attention_mask = attentions.make_causal_attention_mask(x).to(dtype=torch.bool)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            attention_mask = attention_mask.view(x.shape[0], 1, 1, x.shape[1]).repeat(1, 1, x.shape[1], 1)
            attention_mask = torch.logical_and(attention_mask, causal_attention_mask)
        else:
            attention_mask = causal_attention_mask

        x = self.encoder(x, attention_mask=attention_mask, **encoder_kwargs)
        return x

    def neck(self, x):
        h = self.norm(x)
        return h

    def head(self, sequence, h):
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = h[
            torch.arange(h.shape[0], device=sequence.device), sequence.to(torch.int).argmax(dim=-1)
        ]
        pooled_output = self.proj(pooled_output)
        return pooled_output

    def forward(
            self,
            sequence=None,
            attention_mask=None,
            **encoder_kwargs
    ):
        x = self.backbone(
            sequence=sequence,
            attention_mask=attention_mask,
            **encoder_kwargs
        )

        h = self.neck(x)
        # note, so doubtful that use pooler or pooled?
        pooler_output = self.head(sequence, h)

        return dict(
            last_hidden_state=h,  # have been normalized
            pooler_output=pooler_output
        )
