import torch
from torch import nn
from torch.nn import functional as F
from utils import torch_utils
from .. import bundles
from ..text_pretrain.transformers import DecoderEmbedding, make_causal_attention_mask, TransformerSequential
from ..image_classification.ViT import VisionEmbedding


class Config(bundles.Config):
    default_model = 'base'

    model = dict(
        projection_dim=768,
        logit_scale_init_value=2.6592
    )

    vision_base = dict(
        hidden_size=768,
        image_size=224,
        patch_size=32,
        num_attention_heads=12,
        num_hidden_layers=12
    )

    vision_large = dict(
        hidden_size=1024,
        image_size=224,
        patch_size=14,
        num_attention_heads=16,
        num_hidden_layers=24
    )

    text_base = dict(
        hidden_size=512,
        vocab_size=49408,
        max_seq_len=77,
        num_attention_heads=8,
        num_hidden_layers=12
    )

    text_large = dict(
        hidden_size=768,
        vocab_size=49408,
        max_seq_len=77,
        num_attention_heads=12,
        num_hidden_layers=12
    )

    @classmethod
    def make_full_config(cls) -> dict:
        config_dict = dict(
            # base-patch32
            base=dict(
                vision_config=cls.vision_base,
                text_config=cls.text_large,
                **cls.model
            ),

            # large-patch14
            large=dict(
                vision_config=cls.vision_large,
                text_config=cls.text_large,
                **cls.model
            )
        )
        return config_dict


class WeightLoader(bundles.WeightLoader):
    @classmethod
    def auto_download(cls, save_path, save_name=''):
        """download weight auto from transformers
        refer to: https://huggingface.co/openai/clip-vit-base-patch32
        """
        from transformers import CLIPModel

        model = CLIPModel.from_pretrained(save_path, num_labels=2)
        state_dict = model.state_dict()
        return state_dict


class WeightConverter:
    @staticmethod
    def from_hf(state_dict):
        """convert weights from huggingface model to my own model

        Usage:
            .. code-block:: python

                state_dict = WeightLoader.from_hf(...)
                state_dict = WeightConverter.from_hf(state_dict)
                Model(...).load_state_dict(state_dict)

        """
        convert_dict = {
            'vision_model.embeddings.patch_embedding': 'vision_model.embedding.patch.fn.0',
            'vision_model.embeddings.class_embedding': 'vision_model.embedding.class',
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
            'text_model.final_layer_norm': 'text_model.norm'
        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        return state_dict


class Model(nn.Module):
    def __init__(self, projection_dim, logit_scale_init_value,
                 vision_config=Config.vision_base, text_config=Config.text_base
                 ):
        super().__init__()
        self.vision_model = VisionTransformer(**vision_config)
        self.text_model = TextTransformer(**text_config)
        self.visual_projection = nn.Linear(vision_config['hidden_size'], projection_dim, bias=False)
        self.text_projection = nn.Linear(text_config['hidden_size'], projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

    def forward(self, image, text_ids, attention_mask=None):
        vision_outputs = self.vision_model(image)
        text_outputs = self.text_model(text_ids, attention_mask=attention_mask)

        vision_outputs = self.visual_projection(vision_outputs)
        text_outputs = self.text_projection(text_outputs)

        image_embeds = vision_outputs / vision_outputs.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_outputs / text_outputs.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if self.training:
            loss = self.loss(logits_per_text)

        return dict(
            logits_per_text=logits_per_text,
            logits_per_image=logits_per_image,
            loss=loss
        )

    def loss(self, similarity):
        loss_fn = lambda x: F.cross_entropy(x, torch.arange(len(x), device=x.device))

        caption_loss = loss_fn(similarity)
        image_loss = loss_fn(similarity.t())
        return (caption_loss + image_loss) / 2.0


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate.
    See: https://github.com/hendrycks/GELUs
    """

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class VisionTransformer(nn.Module):
    def __init__(self, image_size, hidden_size, patch_size, num_attention_heads, num_hidden_layers):
        super().__init__()

        self.embedding = VisionEmbedding(hidden_size, image_size, patch_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.encoder = TransformerSequential(
            hidden_size, num_attention_heads, hidden_size * 4, norm_first=True,
            ff_kwargs=dict(act=QuickGELUActivation()),
            num_blocks=num_hidden_layers
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, image):
        x = self.embedding(image)
        x = self.norm1(x)
        x = self.encoder(x)

        pooled_output = x[:, 0, :]
        pooled_output = self.norm2(pooled_output)
        return pooled_output


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_seq_len, num_attention_heads, num_hidden_layers):
        super().__init__()
        self.embedding = DecoderEmbedding(vocab_size, hidden_size, max_seq_len=max_seq_len)
        self.encoder = TransformerSequential(
            hidden_size, num_attention_heads, hidden_size * 4, norm_first=True,
            ff_kwargs=dict(act=QuickGELUActivation()),
            num_blocks=num_hidden_layers
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
            self,
            sequence=None,
            attention_mask=None,
    ):
        x = self.embedding(sequence)

        causal_attention_mask = make_causal_attention_mask(x).to(dtype=torch.bool)
        attention_mask = attention_mask.to(dtype=torch.bool)
        attention_mask = attention_mask.view(x.shape[0], 1, 1, x.shape[1]).repeat(1, 1, x.shape[1], 1)
        attention_mask = torch.logical_and(attention_mask, causal_attention_mask)

        x = self.encoder(x, attention_mask=attention_mask)
        x = self.norm(x)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = x[
            torch.arange(x.shape[0], device=sequence.device), sequence.to(torch.int).argmax(dim=-1)
        ]

        return pooled_output
