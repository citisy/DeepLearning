from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils import torch_utils
from . import CosyVoice
from .. import attentions, bundles
from ..text_pretrain import qwen2, transformers
from torch.nn.utils.rnn import pad_sequence


class Config(bundles.Config):
    front = dict(
        n_fft=1920,
        hop_size=480,
        win_size=1920,
        sample_rate=24000,
    )

    llm = dict(
        llm_input_size=896,
        llm_output_size=896,
    )

    flow = dict(
        vocab_size=6561,
        input_frame_rate=25,
        decoder_config=dict(
            in_ch=320,
            out_ch=80,
            hidden_ch=(256,),
            attention_head_dim=64,
            nun_attention_blocks=4,
            num_mid_blocks=12,
            num_attention_heads=8,
        )
    )

    hift = dict(
        sample_rate=24000,
        upsample_rates=[8, 5, 3],
        upsample_kernel_sizes=[16, 11, 7],
        source_resblock_kernel_sizes=[7, 7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )

    default_model = '0.5b'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            '0.5b': dict(
                front_config=cls.front,
                llm_config=cls.llm,
                flow_config=cls.flow,
                hift_config=cls.hift,
            )
        }


class WeightConverter:
    @classmethod
    def from_official(cls, state_dict):
        state_dict = CosyVoice.WeightConverter.from_official(state_dict)

        convert_dict = {'llm.model.' + k: 'llm.model.' + v for k, v in qwen2.WeightConverter.convert_dict.items()}
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        return state_dict


class Model(CosyVoice.Model):
    """
    References:
        - paper:
            [CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models](https://arxiv.org/pdf/2412.10117)
        - code:
            https://github.com/FunAudioLLM/CosyVoice?tab=readme-ov-file
    """

    def init_components(self, front_config=Config.front, llm_config=Config.llm, flow_config=Config.flow, hift_config=Config.hift):
        self.front = SpkEmbedding(**front_config)
        self.llm = Qwen2LM(**llm_config)
        self.flow = CausalMaskedDiffWithXvec(**flow_config)
        self.hift = HiFTGenerator(**hift_config)


class SpkEmbedding(CosyVoice.SpkEmbedding):
    def make_instruct_inputs(self, **kwargs):
        model_inputs = self.make_zero_shot_inputs(**kwargs)
        # dont need lm_prompt_speech
        del model_inputs['llm_prompt_speech_ids']
        del model_inputs['llm_prompt_speech_ids_len']
        return model_inputs


class Qwen2LM(CosyVoice.TransformerLM):
    def __init__(self, llm_input_size=896, llm_output_size=896, speech_ids_size=6561):
        nn.Module.__init__(self)
        self.llm_input_size = llm_input_size
        self.speech_ids_size = speech_ids_size

        # 2. build speech token language model related modules
        self.llm_embedding = nn.Embedding(2, llm_input_size)
        self.llm = Qwen2Encoder()
        self.llm_decoder = nn.Linear(llm_output_size, speech_ids_size + 3)

        # 3. [Optional] build speech token related modules
        self.speech_embedding = nn.Embedding(speech_ids_size + 3, llm_input_size)

    def encode_text(self, text_ids, text_ids_len):
        return self.llm.model.embedding(text_ids), text_ids_len

    def encode_embedding(self, embedding, text_embedding):
        # dont need encoding embedding
        return torch.zeros(text_embedding.shape[0], 0, self.llm_input_size, dtype=text_embedding.dtype, device=text_embedding.device)

    def decode(self, lm_input, max_len, min_len):
        batch_size = lm_input.shape[0]
        out_ids = [[] for _ in range(batch_size)]
        out_lens = torch.zeros(batch_size, device=lm_input.device, dtype=torch.long) + max_len
        eos_flag = [False] * batch_size
        past_kvs = self.llm.model.make_caches()
        start_pos = 0
        for i in range(max_len):
            y_pred = self.llm(
                lm_input,
                start_pos=start_pos,
                past_kvs=past_kvs
            )
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = []
            for batch in range(batch_size):
                top_id = self.sampling_ids(logp[batch], out_ids[batch], ignore_eos=True if i < min_len else False)
                if top_id == self.speech_ids_size:
                    if not eos_flag[batch]:
                        out_lens[batch] = len(out_ids[batch])
                    eos_flag[batch] = True
                if top_id < self.speech_ids_size:
                    out_ids[batch].append(top_id)
                top_ids.append(out_ids[batch][-1])

            if all(eos_flag):
                break

            start_pos += lm_input.shape[1]
            top_ids = torch.tensor(top_ids, device=lm_input.device).reshape(batch_size, 1)
            lm_input = self.speech_embedding(top_ids)

        out_ids = [torch.tensor(ids, device=lm_input.device) for ids in out_ids]
        out_ids = pad_sequence(out_ids, batch_first=True, padding_value=0)
        return out_ids, out_lens


class Qwen2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = qwen2.Model(**qwen2.Config.get('0.5b'))

    def forward(self, xs, start_pos=0, past_kvs=None):
        y = self.model.decoder(xs, start_pos=start_pos, past_kvs=past_kvs)
        return y


class CausalMaskedDiffWithXvec(nn.Module):
    def __init__(
            self,
            input_size: int = 512,
            output_size: int = 80,
            spk_embed_dim: int = 192,
            output_type: str = "mel",
            vocab_size: int = 6561,
            input_frame_rate: int = 25,
            only_mask_loss: bool = True,
            token_mel_ratio: int = 2,
            pre_lookahead_len: int = 3,
            encoder_config: dict = {},
            decoder_config: dict = {},
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = UpsampleConformerEncoder(**encoder_config)
        self.encoder_proj = nn.Linear(self.encoder.output_size, output_size)
        self.decoder = CausalConditionalCFM(**decoder_config)
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    def forward(
            self,
            token,
            token_len,
            prompt_token,
            prompt_token_len,
            prompt_feat,
            prompt_feat_len,
            embedding,
            stream=False,
            finalize=True,
            flow_cache=None,
            **kwargs
    ):
        batch_size = token.shape[0]
        # assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (attentions.make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        if finalize is True:
            h, h_lengths = self.encoder(token, token_len, stream=stream)
        else:
            token, context = token[:, :-self.pre_lookahead_len], token[:, -self.pre_lookahead_len:]
            h, h_lengths = self.encoder(token, token_len, context=context, stream=stream)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]
        h = self.encoder_proj(h)

        # get conditions
        conds = torch.zeros([batch_size, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (attentions.make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            stream=stream,
            cache=flow_cache
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), flow_cache


class UpsampleConformerEncoder(nn.Module):
    def __init__(
            self,
            input_size: int = 512,
            output_size: int = 512,
            attention_heads: int = 8,
            linear_units: int = 2048,
            num_blocks: int = 6,
            drop_prob: float = 0.1,
            normalize_before: bool = True,
            static_chunk_size: int = 25,
            use_dynamic_chunk: bool = False,
            global_cmvn: nn.Module = None,
            use_dynamic_left_chunk: bool = False,
            macaron_style: bool = False,
            gradient_checkpointing: bool = False,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            drop_prob (float): dropout rate
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
            gradient_checkpointing: rerunning a forward-pass segment for each
                checkpointed segment during backward.
        """
        super().__init__()
        self.output_size = output_size

        self.global_cmvn = global_cmvn
        self.embed = CosyVoice.LinearNoSubsampling(
            input_size,
            output_size,
            drop_prob,
        )

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        self.pre_lookahead_layer = PreLookaheadLayer(ch=512, pre_lookahead_len=3)
        self.encoders = nn.ModuleList([
            CosyVoice.ConformerEncoderLayer(
                output_size,
                attentions.CrossAttention2D(
                    attention_heads, output_size,
                    attend=attentions.DynamicMemoryAttendWrapper(CosyVoice.RelPositionAttend(output_size, attention_heads))
                ),
                transformers.PositionWiseFeedForward(output_size, linear_units, act=nn.SiLU(), drop_prob=drop_prob),
                transformers.PositionWiseFeedForward(output_size, linear_units, act=nn.SiLU(), drop_prob=drop_prob) if macaron_style else None,
                drop_prob,
                normalize_before,
            ) for _ in range(num_blocks)
        ])
        self.up_layer = Upsample1D(in_ch=512, out_ch=512, stride=2)
        self.up_embed = CosyVoice.LinearNoSubsampling(
            input_size,
            output_size,
            drop_prob,
        )
        self.up_encoders = nn.ModuleList([
            CosyVoice.ConformerEncoderLayer(
                output_size,
                attentions.CrossAttention2D(
                    attention_heads, output_size,
                    attend=attentions.DynamicMemoryAttendWrapper(CosyVoice.RelPositionAttend(output_size, attention_heads))
                ),
                transformers.PositionWiseFeedForward(output_size, linear_units, act=nn.SiLU(), drop_prob=drop_prob),
                transformers.PositionWiseFeedForward(output_size, linear_units, act=nn.SiLU(), drop_prob=drop_prob) if macaron_style else None,
                drop_prob,
                normalize_before,
            ) for _ in range(4)
        ])

    def forward(
            self,
            xs: torch.Tensor,
            xs_lens: torch.Tensor,
            context: torch.Tensor = torch.zeros(0, 0, 0),
            stream: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        T = xs.size(1)
        masks = attentions.make_pad_mask(xs_lens, max_len=T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb = self.embed(xs)
        if context.size(1) != 0:
            assert self.training is False, 'you have passed context, make sure that you are running inference mode'
            context, _ = self.embed(context, offset=xs.size(1))
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = CosyVoice.add_optional_chunk_mask(xs, masks, False, 0, self.static_chunk_size if stream is True else 0)
        # lookahead + conformer encoder
        xs = self.pre_lookahead_layer(xs, context=context)
        for layer in self.encoders:
            xs = layer(xs, chunk_masks, pos_emb)

        # upsample + conformer encoder
        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        T = xs.size(1)
        masks = attentions.make_pad_mask(xs_lens, max_len=T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb = self.up_embed(xs)
        chunk_masks = CosyVoice.add_optional_chunk_mask(xs, masks, False, 0, self.static_chunk_size * self.up_layer.stride if stream is True else 0)
        for layer in self.up_encoders:
            xs = layer(xs, chunk_masks, pos_emb)

        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks


class PreLookaheadLayer(nn.Module):
    def __init__(self, ch: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = ch
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            ch, ch,
            kernel_size=pre_lookahead_len + 1,
            stride=1, padding=0,
        )
        self.conv2 = nn.Conv1d(
            ch, ch,
            kernel_size=3, stride=1, padding=0,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = torch.zeros(0, 0, 0)) -> torch.Tensor:
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs = x.transpose(1, 2).contiguous()
        context = context.transpose(1, 2).contiguous()
        # look ahead
        if context.size(2) == 0:
            outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0.0)
        else:
            assert self.training is False, 'you have passed context, make sure that you are running inference mode'
            assert context.size(2) == self.pre_lookahead_len
            outputs = F.pad(torch.concat([outputs, context], dim=2), (0, self.pre_lookahead_len - context.size(2)), mode='constant', value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        # outputs
        outputs = F.pad(outputs, (self.conv2.kernel_size[0] - 1, 0), mode='constant', value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        # residual connection
        outputs = outputs + x
        return outputs


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        # In this mode, first repeat interpolate, than conv with stride=1
        self.conv = nn.Conv1d(in_ch, out_ch, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride


class CausalConditionalCFM(CosyVoice.ConditionalCFM):
    def __init__(self, **decoder_kwargs):
        nn.Module.__init__(self)
        self.estimator = CausalConditionalDecoder(**decoder_kwargs)
        self.initialize_layers()

    def initialize_layers(self):
        self.register_buffer('rand_noise', torch.randn([1, 80, 50 * 300]), persistent=False)

    def _apply(self, fn, recurse=True):
        """apply for meta load"""
        if self.rand_noise.is_meta:
            self.initialize_layers()
        return super()._apply(fn, recurse)

    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, stream=False, cache=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = self.rand_noise[:, :, :mu.size(2)] * temperature
        # fix prompt and overlap part mu and z
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, stream=stream), cache


class CausalConditionalDecoder(CosyVoice.ConditionalDecoder):
    def __init__(self, **kwargs):
        """
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        """
        super().__init__(
            sample_conv_fn=CausalConv1d,
            block_fn=CausalBlock1D,
            **kwargs
        )


class CausalBlock1D(CosyVoice.Block1D):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        nn.Module.__init__(self)
        self.block = nn.Sequential(
            CausalConv1d(in_dim, out_dim, 3),
            Transpose(1, 2),
            nn.LayerNorm(out_dim),
            Transpose(1, 2),
            nn.Mish(),
        )


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs) -> None:
        kwargs.update(padding=0)  # padding step will do after
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        assert stride == 1
        self.causal_padding = kernel_size - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.causal_padding, 0), value=0.0)
        x = super().forward(x)
        return x


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class HiFTGenerator(CosyVoice.HiFTGenerator):
    def __init__(self, **kwargs):
        super().__init__(m_source_fn=SourceModuleHnNSF2, **kwargs)


class SourceModuleHnNSF2(nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sample_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sample_rate: sample_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(self, sample_rate, upsample_scale, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0):
        super().__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen2(sample_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class SineGen2(nn.Module):
    """ Definition of sine generator
    SineGen(sample_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    sample_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(
            self, sample_rate, upsample_scale, harmonic_num=0,
            sine_amp=0.1, noise_std=0.003,
            voiced_threshold=0,
            flag_for_pulse=False
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sample_rate = sample_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sample_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear").transpose(1, 2)

            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
            sines = torch.sin(phase)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        # fundamental component
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))

        # generate sine waveforms
        sine_waves = self._f02sine(fn) * self.sine_amp

        # generate uv signal
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn_like(sine_waves)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise
