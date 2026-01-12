import math
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from einops import pack, rearrange, repeat
from librosa.filters import mel as librosa_mel_fn
from scipy.signal import get_window
from torch import nn
from torch.distributions.uniform import Uniform
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.rnn import pad_sequence

from utils import torch_utils
from .. import attentions, embeddings
from ..layers import Conv, Linear, Upsample
from ..text_pretrain import transformers


class WeightConverter:
    @classmethod
    def from_official(cls, state_dict):
        convert_dict = {
            'decoder.estimator.up_blocks.0.2.conv': 'decoder.estimator.up_blocks.0.2.op',
            'decoder.estimator.{0}.1.{1}.norm1': 'decoder.estimator.{0}.1.{1}.attn_res.norm',
            'decoder.estimator.{0}.1.{1}.attn1.to_q': 'decoder.estimator.{0}.1.{1}.attn_res.fn.to_qkv.0',
            'decoder.estimator.{0}.1.{1}.attn1.to_k': 'decoder.estimator.{0}.1.{1}.attn_res.fn.to_qkv.1',
            'decoder.estimator.{0}.1.{1}.attn1.to_v': 'decoder.estimator.{0}.1.{1}.attn_res.fn.to_qkv.2',
            'decoder.estimator.{0}.1.{1}.attn1.to_out.0': 'decoder.estimator.{0}.1.{1}.attn_res.fn.to_out.linear',
            'decoder.estimator.{0}.1.{1}.norm3': 'decoder.estimator.{0}.1.{1}.ff_res.norm',
            'decoder.estimator.{0}.1.{1}.ff.net.0.proj': 'decoder.estimator.{0}.1.{1}.ff_res.fn.0.linear',
            'decoder.estimator.{0}.1.{1}.ff.net.2': 'decoder.estimator.{0}.1.{1}.ff_res.fn.1.linear',

            "{0}.feed_forward.w_1": "{0}.feed_forward.0.linear",
            "{0}.feed_forward.w_2": "{0}.feed_forward.1.linear",
            '{0}.self_attn.pos_bias_u': '{0}.self_attn.attend.base_layer.pos_bias_u',
            '{0}.self_attn.pos_bias_v': '{0}.self_attn.attend.base_layer.pos_bias_v',
            '{0}.self_attn.linear_pos': '{0}.self_attn.attend.base_layer.linear_pos',
            '{0}.self_attn.linear_q': '{0}.self_attn.to_qkv.0',
            '{0}.self_attn.linear_k': '{0}.self_attn.to_qkv.1',
            '{0}.self_attn.linear_v': '{0}.self_attn.to_qkv.2',
            '{0}.self_attn.linear_out': '{0}.self_attn.to_out.linear',

        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


class Model(nn.Module):
    """Codec-based synthesizer for Voice generation

    References:
        - paper:
            [CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens](https://funaudiollm.github.io/pdf/CosyVoice_v1.pdf)
        - code:
            https://github.com/FunAudioLLM/CosyVoice?tab=readme-ov-file
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.init_components(**kwargs)
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'

    def init_components(self, front_config=dict(), llm_config=dict(), flow_config=dict(), hift_config=dict()):
        self.front = SpkEmbedding(**front_config)
        self.llm = TransformerLM(**llm_config)
        self.flow = MaskedDiffWithXvec(**flow_config)
        self.hift = HiFTGenerator(**hift_config)

    def init_caches(self, batch_size=1):
        return dict(
            source_speech_ids=torch.zeros(batch_size, 0, dtype=torch.int64),
            batch=[dict(
                hift=None,
                mel_overlap=torch.zeros(1, 80, 0),
                flow=torch.zeros(1, 80, 0, 2)
            ) for _ in range(batch_size)]
        )

    def forward(self, **kwargs):
        if self.training:
            raise NotImplementedError
        else:
            return self.inference(**kwargs)

    def inference(
            self,
            text_ids=None, text_ids_len=None,
            prompt_text_ids=None, prompt_text_ids_len=None,
            prompt_speech=None, source_speech=None,
            spk_id='', is_instruct=False,
            stream=False, speed=1.0, caches=None,
            **kwargs
    ):
        batch_size = text_ids.shape[0]

        inputs = self.front(
            prompt_speech=prompt_speech, source_speech=source_speech,
            prompt_text_ids=prompt_text_ids, prompt_text_ids_len=prompt_text_ids_len,
            spk_id=spk_id, is_instruct=is_instruct,
        )
        if caches is None:
            caches = self.init_caches(batch_size)

        if 'source_speech_ids' not in inputs:
            source_speech_ids = caches['source_speech_ids'].to(text_ids.device)

            if text_ids_len is None:
                text_ids_len = torch.tensor([len(t) for t in text_ids], dtype=torch.int32, device=text_ids.device)

            llm_embedding = inputs.get('llm_embedding', torch.zeros(0, 192)).to(text_ids.device)  # embedding from front will have different devices
            prompt_text_ids = inputs.get('prompt_text_ids', torch.zeros(batch_size, 0, dtype=torch.int32, device=text_ids.device))
            prompt_text_ids_len = inputs.get('prompt_text_ids_len', torch.zeros(batch_size, dtype=torch.int32, device=text_ids.device))
            llm_prompt_speech_ids = inputs.get('llm_prompt_speech_ids', torch.zeros(batch_size, 0, dtype=torch.int32, device=text_ids.device))
            llm_prompt_speech_ids_len = inputs.get('llm_prompt_speech_ids_len', torch.zeros(batch_size, dtype=torch.int32, device=text_ids.device))

            out_ids, out_lens = self.llm(
                text_ids=text_ids,
                text_ids_len=text_ids_len,
                prompt_text_ids=prompt_text_ids,
                prompt_text_ids_len=prompt_text_ids_len,
                prompt_speech_ids=llm_prompt_speech_ids,
                prompt_speech_ids_len=llm_prompt_speech_ids_len,
                embedding=llm_embedding,
            )
            source_speech_ids_len = torch.tensor([source_speech_ids.shape[1]] * batch_size, dtype=torch.int32, device=source_speech_ids.device) + out_lens
            source_speech_ids = torch.cat([source_speech_ids, out_ids], dim=1)

        else:
            source_speech_ids = inputs['source_speech_ids']
            source_speech_ids_len = inputs['source_speech_ids_len']

        caches['source_speech_ids'] = source_speech_ids

        flow_embedding = inputs.get('flow_embedding', torch.zeros(0, 192)).to(source_speech_ids.device)
        flow_prompt_speech_ids = inputs.get('flow_prompt_speech_ids', torch.zeros(batch_size, 0, dtype=torch.int32, device=source_speech_ids.device))
        flow_prompt_speech_ids_len = inputs.get('flow_prompt_speech_ids_len', torch.zeros(batch_size, dtype=torch.int32, device=source_speech_ids.device))
        prompt_speech_feat = inputs.get('prompt_speech_feat', torch.zeros(batch_size, 0, 80, dtype=torch.int32, device=source_speech_ids.device))
        prompt_speech_feat_len = inputs.get('prompt_speech_feat_len', torch.zeros(batch_size, dtype=torch.int32, device=source_speech_ids.device))

        if stream is True:
            assert batch_size == 1, "stream mode only support batch_size=1"
            token_hop_len = self.token_min_hop_len
            this_tts_speech = []
            while True:
                if len(source_speech_ids) >= token_hop_len + self.token_overlap_len:
                    chunk_source_speech_ids = source_speech_ids[:, token_hop_len + self.token_overlap_len]
                    chunk_source_speech_ids_len = torch.tensor([chunk_source_speech_ids.shape[1]], dtype=torch.int32, device=chunk_source_speech_ids.device)
                    _this_tts_speech = self.token2wav(
                        token=chunk_source_speech_ids,
                        token_len=chunk_source_speech_ids_len,
                        prompt_token=flow_prompt_speech_ids,
                        prompt_token_len=flow_prompt_speech_ids_len,
                        prompt_feat=prompt_speech_feat,
                        prompt_feat_len=prompt_speech_feat_len,
                        embedding=flow_embedding,
                        caches=caches['batch'][0],
                        stream=stream,
                        finalize=False
                    )
                    this_tts_speech.append(_this_tts_speech)
                    source_speech_ids = source_speech_ids[token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if len(source_speech_ids) < token_hop_len + self.token_overlap_len:
                    break
            chunk_source_speech_ids = source_speech_ids
            chunk_source_speech_ids_len = torch.tensor([chunk_source_speech_ids.shape[1]], dtype=torch.int32, device=chunk_source_speech_ids.device)
            _this_tts_speech = self.token2wav(
                token=chunk_source_speech_ids,
                token_len=chunk_source_speech_ids_len,
                prompt_token=flow_prompt_speech_ids,
                prompt_token_len=flow_prompt_speech_ids_len,
                prompt_feat=prompt_speech_feat,
                prompt_feat_len=prompt_speech_feat_len,
                embedding=flow_embedding,
                caches=caches['batch'][0],
                stream=stream,
                finalize=True
            )
            this_tts_speech.append(_this_tts_speech)
            this_tts_speech = [torch.cat(this_tts_speech, dim=1)]
        else:
            this_tts_speech = []
            for i in range(batch_size):
                token_len = source_speech_ids_len[i]
                token = source_speech_ids[i, :token_len]
                _this_tts_speech = self.token2wav(
                    token=token[None],
                    token_len=token_len[None],
                    prompt_token=flow_prompt_speech_ids[i:i + 1],
                    prompt_token_len=flow_prompt_speech_ids_len[i:i + 1],
                    prompt_feat=prompt_speech_feat[i:i + 1],
                    prompt_feat_len=prompt_speech_feat_len[i:i + 1],
                    embedding=flow_embedding[i:i + 1],
                    caches=caches['batch'][i],
                    stream=stream,
                    finalize=True,
                    speed=speed
                )
                this_tts_speech.append(_this_tts_speech)

        results = {'tts_speech': this_tts_speech}
        return results

    def token2wav(self, token, token_len, prompt_token, prompt_token_len, prompt_feat, prompt_feat_len, embedding, caches, stream=False, finalize=False, speed=1.0):
        tts_mel, caches['flow'] = self.flow(
            token=token,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            stream=stream,
            finalize=finalize,
            flow_cache=caches['flow']
        )

        # mel overlap fade in out
        if caches['mel_overlap'].shape[2] != 0:
            tts_mel = self.fade_in_out(tts_mel, caches['mel_overlap'])
        # append hift cache
        if caches['hift'] is not None:
            hift_cache_mel, hift_cache_source = caches['hift']['mel'], caches['hift']['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            caches['mel_overlap'] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift(speech_feat=tts_mel, cache_source=hift_cache_source)
            if caches['hift'] is not None:
                tts_speech = self.fade_in_out(tts_speech, caches['hift']['speech'])
            caches['hift'] = {
                'mel': tts_mel[:, :, -self.mel_cache_len:],
                'source': tts_source[:, :, -self.source_cache_len:],
                'speech': tts_speech[:, -self.source_cache_len:]
            }
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert caches['hift'] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift(speech_feat=tts_mel, cache_source=hift_cache_source)
            if caches['hift'] is not None:
                tts_speech = self.fade_in_out(tts_speech, caches['hift']['speech'])
        return tts_speech

    def fade_in_out(self, fade_in_mel, fade_out_mel):
        device = fade_in_mel.device
        fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
        mel_overlap_len = int(self.speech_window.shape[0] / 2)
        if fade_in_mel.device == torch.device('cpu'):
            fade_in_mel = fade_in_mel.clone()
        fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * self.speech_window[:mel_overlap_len] + fade_out_mel[..., -mel_overlap_len:] * self.speech_window[mel_overlap_len:]
        return fade_in_mel.to(device)


class SpkEmbedding(nn.Module):
    mel_basis = {}
    hann_window = {}

    n_fft = 1024
    num_mels = 80
    sample_rate = 22050
    hop_size = 256
    win_size = 1024
    fmin = 0
    fmax = 8000
    center = False

    def __init__(self, campplus_model, speech_idsizer_model, **kwargs):
        """
        Args:
            campplus_model (str): like "xxx/campplus.onnx"
            speech_idsizer_model (str): like "speech_tokenizer_v1.onnx"
            **kwargs:
        """
        super().__init__()
        self.__dict__.update(kwargs)
        import onnxruntime

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"]
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=providers)
        self.speech_idsizer_session = onnxruntime.InferenceSession(speech_idsizer_model, sess_options=option, providers=providers)
        self.resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=self.sample_rate)
        self.spk2info = {}

    def update_spk2info(self, spk2info):
        self.spk2info.update(spk2info)

    def forward(self, spk_id='', is_instruct=False, **kwargs):
        if is_instruct:
            model_inputs = self.make_instruct_inputs(spk_id=spk_id, **kwargs)

        elif spk_id in self.spk2info:
            model_inputs = self.make_sft_inputs(spk_id=spk_id, **kwargs)

        else:
            model_inputs = self.make_zero_shot_inputs(**kwargs)

        return model_inputs

    def make_instruct_inputs(self, prompt_text_ids=None, prompt_text_ids_len=None, spk_id='', **kwargs):
        assert spk_id in self.spk2info, f'Instruct model must have pretrained embedding, pls check `{spk_id}` is in spk2info'
        embedding = self.spk2info[spk_id]['embedding']
        return {
            'prompt_text_ids': prompt_text_ids, 'prompt_text_ids_len': prompt_text_ids_len,
            'flow_embedding': embedding,  # no need to set llm_embedding
        }

    def make_sft_inputs(self, spk_id, **kwargs):
        embedding = self.spk2info[spk_id]['embedding']
        return {
            'llm_embedding': embedding, 'flow_embedding': embedding
        }

    def make_zero_shot_inputs(
            self,
            prompt_speech=None, source_speech=None,
            prompt_text_ids=None, prompt_text_ids_len=None,
            **kwargs
    ):
        model_inputs = {}
        prompt_speech_resample = self.resample(prompt_speech)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        speech_ids, speech_ids_len = self._extract_speech_ids(prompt_speech)
        if self.sample_rate == 24000:
            token_len = min(int(prompt_speech_feat.shape[1] / 2), speech_ids.shape[1])
            prompt_speech_feat, prompt_speech_feat_len[:] = prompt_speech_feat[:, :2 * token_len], 2 * token_len
            speech_ids, speech_ids_len[:] = speech_ids[:, :token_len], token_len
        embedding = self._extract_spk_embedding(prompt_speech)

        model_inputs.update({
            'flow_prompt_speech_ids': speech_ids, 'flow_prompt_speech_ids_len': speech_ids_len,
            'prompt_speech_feat': prompt_speech_feat, 'prompt_speech_feat_len': prompt_speech_feat_len,
        })

        if source_speech is not None:
            source_speech_ids, source_speech_ids_len = self._extract_speech_ids(source_speech)
            model_inputs.update({
                'source_speech_ids': source_speech_ids, 'source_speech_ids_len': source_speech_ids_len,
                'flow_embedding': embedding
            })
        else:
            model_inputs.update({
                'llm_embedding': embedding, 'flow_embedding': embedding
            })

            if prompt_text_ids is not None:
                model_inputs.update({
                    'prompt_text_ids': prompt_text_ids,
                    'llm_prompt_speech_ids': speech_ids,
                })

            if prompt_text_ids_len is not None:
                model_inputs.update({
                    'prompt_text_ids_len': prompt_text_ids_len,
                    'llm_prompt_speech_ids_len': speech_ids_len,
                })

        return model_inputs

    def _extract_speech_ids(self, speech):
        import whisper

        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feats = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_ids = []
        for feat in feats:
            feat = feat[None]
            speech_id = self.speech_idsizer_session.run(
                None,
                {
                    self.speech_idsizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                    self.speech_idsizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)
                })[0].flatten().tolist()
            speech_ids.append(torch.tensor(speech_id, dtype=torch.int32, device=speech.device))
        speech_ids_len = torch.tensor([len(speech_id) for speech_id in speech_ids], dtype=torch.int32, device=speech.device)
        speech_ids = pad_sequence(speech_ids, batch_first=True, padding_value=-1)
        return speech_ids, speech_ids_len

    def _extract_spk_embedding(self, speech):
        embedding = []
        for s in speech:
            s = s[None]
            feat = kaldi.fbank(s,
                               num_mel_bins=80,
                               dither=0,
                               sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            embed = self.campplus_session.run(
                None,
                {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()}
            )[0].flatten().tolist()
            embedding.append(embed)
        embedding = torch.tensor(embedding, device=speech.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).transpose(1, 2)
        speech_feat_len = torch.tensor([speech_feat.shape[2]], dtype=torch.int32)
        return speech_feat, speech_feat_len

    def feat_extractor(self, y):
        if f"{str(self.fmax)}_{str(y.device)}" not in self.mel_basis:
            mel = librosa_mel_fn(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax)
            self.mel_basis[str(self.fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
            self.hann_window[str(y.device)] = torch.hann_window(self.win_size).to(y.device)

        y = nn.functional.pad(y.unsqueeze(1), (int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)), mode="reflect")
        y = y.squeeze(1)

        spec = torch.view_as_real(
            torch.stft(
                y,
                self.n_fft,
                hop_length=self.hop_size,
                win_length=self.win_size,
                window=self.hann_window[str(y.device)],
                center=self.center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(self.mel_basis[str(self.fmax) + "_" + str(y.device)], spec)
        spec = torch.log(torch.clamp(spec, min=1e-5) * 1)
        return spec


class TransformerLM(nn.Module):
    ignore_id = -1
    sos_eos = 0
    task_id = 1

    def __init__(self, text_encoder_input_size=512, llm_input_size=1024, llm_output_size=1024, text_token_size=51866, speech_ids_size=4096, spk_embed_dim=192):
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_ids_size = speech_ids_size
        # 1. build text token inputs related modules
        self.text_embedding = nn.Embedding(text_token_size, text_encoder_input_size)
        self.text_encoder = ConformerEncoder()
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size,
            llm_input_size
        )

        # 2. build speech token language model related modules
        self.llm_embedding = nn.Embedding(2, llm_input_size)
        self.llm = TransformerEncoder()
        self.llm_decoder = nn.Linear(llm_output_size, speech_ids_size + 1)

        # 3. [Optional] build speech token related modules
        self.speech_embedding = nn.Embedding(speech_ids_size, llm_input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, llm_input_size)

    def forward(
            self,
            text_ids=None, text_ids_len=None,
            prompt_text_ids=None, prompt_text_ids_len=None,
            prompt_speech_ids=None, prompt_speech_ids_len=None,
            embedding: torch.Tensor = None,
            max_token_text_ratio: float = 20,
            min_token_text_ratio: float = 2,
            **kwargs
    ):
        if self.training:
            raise NotImplementedError

        text_ids = torch.concat([prompt_text_ids, text_ids], dim=1)
        text_ids_len = text_ids_len + prompt_text_ids_len

        # 1. encode text
        text_embedding, text_embedding_lens = self.encode_text(text_ids, text_ids_len)

        # 2. encode embedding
        embedding = self.encode_embedding(embedding, text_embedding)

        # 3. concat llm_input
        lm_input = self.make_lm_input(text_embedding, prompt_speech_ids, prompt_speech_ids_len, embedding)

        # 4. cal min/max_length
        min_len = ((text_embedding_lens - prompt_text_ids_len) * min_token_text_ratio).min()
        max_len = ((text_embedding_lens - prompt_text_ids_len) * max_token_text_ratio).max()

        # 5. step by step decode
        return self.decode(lm_input, max_len, min_len)

    def encode_text(self, text_ids, text_ids_len):
        text_embedding = self.text_embedding(text_ids)
        text_embedding, text_embedding_mask = self.text_encoder(text_embedding, text_ids_len, decoding_chunk_size=1)
        text_embedding_lens = text_embedding_mask.squeeze(1).sum(1)
        text_embedding = self.text_encoder_affine_layer(text_embedding)
        return text_embedding, text_embedding_lens

    def encode_embedding(self, embedding, text_embedding):
        if embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(text_embedding.shape[0], 0, self.llm_input_size, dtype=text_embedding.dtype, device=text_embedding.device)
        return embedding

    def make_lm_input(self, text_embedding, prompt_speech_ids, prompt_speech_ids_len, embedding):
        b = len(text_embedding)
        sos_eos_emb = self.llm_embedding(torch.tensor([self.sos_eos] * b, device=text_embedding.device))[:, None, :]
        task_id_emb = self.llm_embedding(torch.tensor([self.task_id] * b, device=text_embedding.device))[:, None, :]
        if (prompt_speech_ids_len != 0).all():
            prompt_speech_ids_emb = self.speech_embedding(prompt_speech_ids)
        else:
            prompt_speech_ids_emb = torch.zeros(b, 0, self.llm_input_size, dtype=text_embedding.dtype, device=text_embedding.device)
        lm_input = torch.concat([sos_eos_emb, embedding, text_embedding, task_id_emb, prompt_speech_ids_emb], dim=1)

        return lm_input

    def decode(self, lm_input, max_len, min_len):
        batch_size = lm_input.shape[0]
        out_ids = [[] for _ in range(batch_size)]
        out_lens = torch.zeros(batch_size, device=lm_input.device, dtype=torch.long) + max_len
        eos_flag = [False] * batch_size
        start_pos = 0
        past_kvs = [dict() for i in range(self.llm.num_blocks)]

        for i in range(max_len):
            y_pred = self.llm(
                lm_input, start_pos=start_pos, past_kvs=past_kvs,
                attention_mask=torch.tril(torch.ones((1, 1, lm_input.shape[1], lm_input.shape[1]), device=lm_input.device)).to(torch.bool)
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

        out_ids = torch.tensor(out_ids, device=lm_input.device)
        return out_ids, out_lens

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            decoded_tokens: List,
            ignore_eos: bool = True,
    ):
        num_trials, max_trials = 0, 100
        while True:
            top_ids = self.ras_sampling(weighted_scores, decoded_tokens)
            if (not ignore_eos) or (self.speech_ids_size not in top_ids):
                break
            num_trials += 1
            if num_trials > max_trials:
                raise RuntimeError(f'sampling reaches max_trials {max_trials} and still get eos when ignore_eos is True, check your input!')
        return top_ids

    def ras_sampling(self, weighted_scores, decoded_tokens, top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
        top_ids = self.nucleus_sampling(weighted_scores, top_p=top_p, top_k=top_k)
        rep_num = (torch.tensor(decoded_tokens[-win_size:]).to(weighted_scores.device) == top_ids).sum().item()
        if rep_num >= win_size * tau_r:
            top_ids = self.random_sampling(weighted_scores)
        return top_ids

    @staticmethod
    def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
        prob, indices = [], []
        cum_prob = 0.0
        sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
        for i in range(len(sorted_idx)):
            # sampling both top-p and numbers.
            if cum_prob < top_p and len(prob) < top_k:
                cum_prob += sorted_value[i]
                prob.append(sorted_value[i])
                indices.append(sorted_idx[i])
            else:
                break
        prob = torch.tensor(prob).to(weighted_scores)
        indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
        top_ids = indices[prob.multinomial(1, replacement=True)]
        return top_ids

    @staticmethod
    def random_sampling(weighted_scores):
        top_ids = weighted_scores.softmax(dim=0).multinomial(1, replacement=True)
        return top_ids


class LinearNoSubsampling(nn.Module):
    """Linear transform the input without subsampling"""

    def __init__(self, input_size: int, output_size: int, drop_prob: float):
        """Construct an linear object."""
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LayerNorm(output_size, eps=1e-5),
            nn.Dropout(drop_prob),
        )
        self.pos_enc = EspnetRelPositionalEncoding(5000, output_size, factor=math.sqrt(output_size), drop_prob=drop_prob)

    def forward(self, x, start_pos=0):
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, start_pos)
        return x, pos_emb


class LegacyLinearNoSubsampling(nn.Module):
    """Linear transform the input without subsampling"""

    def __init__(self, input_size: int, output_size: int, drop_prob=0.1):
        """Construct an linear object."""
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LayerNorm(output_size, eps=1e-5),
            nn.Dropout(drop_prob),
            nn.ReLU(),
        )
        self.pos_enc = EspnetRelPositionalEncoding(5000, output_size, factor=math.sqrt(output_size), drop_prob=drop_prob)

    def forward(self, x, start_pos=0):
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, start_pos)
        return x, pos_emb


class EspnetRelPositionalEncoding(embeddings.PositionalEmbedding):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    """

    def initialize_layers(self, num_embeddings=None):
        if num_embeddings is None:
            num_embeddings = self.num_embeddings

        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(num_embeddings, self.embedding_dim)
        pe_negative = torch.zeros(num_embeddings, self.embedding_dim)
        position = torch.arange(0, num_embeddings, dtype=torch.float32).unsqueeze(1)
        div_term = embeddings.make_pos_div_term(self.embedding_dim, self.theta)
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        weight = torch.cat([pe_positive, pe_negative], dim=1)
        self.register_buffer('weight', weight, persistent=False)

    def extend_weight(self, seq_len):
        """Reset the positional encodings."""
        if self.weight.shape[1] < seq_len * 2 - 1:
            self.initialize_layers(seq_len)

    def make_embedding(self, seq_len, start_pos=0) -> torch.Tensor:
        """ For getting encoding in a stream fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        stream way, but will call this function several times with
        increasing input size in a stream scenario, so the dropout will
        be applied several times.

        """
        pos_emb = self.weight[:, self.weight.shape[1] // 2 - seq_len - start_pos + 1: self.weight.shape[1] // 2 + seq_len + start_pos]
        return pos_emb

    def forward(self, x, start_pos=0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
            start_pos (int):

        """
        seq_len = x.shape[1]
        self.extend_weight(seq_len)
        x = x * self.factor
        emb = self.make_embedding(seq_len, start_pos=start_pos)
        return self.dropout(x), self.dropout(emb)


class ConformerEncoder(nn.Module):
    """Conformer encoder module."""

    def __init__(
            self,
            input_size: int = 512,
            output_size: int = 1024,
            attention_heads: int = 16,
            linear_units: int = 4096,
            num_blocks: int = 6,
            drop_prob: float = 0.1,
            normalize_before: bool = True,
            static_chunk_size: int = 1,
            use_dynamic_chunk: bool = False,
            use_dynamic_left_chunk: bool = False,
            macaron_style: bool = False,
    ):
        super().__init__()
        self.normalize_before = normalize_before
        self.output_size = output_size
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.num_blocks = num_blocks

        self.embed = LinearNoSubsampling(
            input_size,
            output_size,
            drop_prob,
        )

        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)

        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                attentions.CrossAttention2D(
                    attention_heads, output_size,
                    attend=attentions.DynamicMemoryAttendWrapper(RelPositionAttend(output_size, attention_heads))
                ),
                transformers.PositionWiseFeedForward(output_size, linear_units, act=nn.SiLU(), drop_prob=drop_prob),
                transformers.PositionWiseFeedForward(output_size, linear_units, act=nn.SiLU(), drop_prob=drop_prob) if macaron_style else None,
                drop_prob,
                normalize_before,
            ) for _ in range(num_blocks)
        ])

    def forward(
            self,
            xs: torch.Tensor,
            xs_lens: torch.Tensor,
            decoding_chunk_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample (B, 1, T' ~= T/subsample_rate)
        """
        masks = attentions.make_pad_mask(xs_lens, max_len=xs.shape[1]).unsqueeze(1)
        xs, pos_emb = self.embed(xs)
        attention_mask = add_optional_chunk_mask(
            xs, masks,
            self.use_dynamic_chunk,
            decoding_chunk_size,
            self.static_chunk_size,
        )

        for layer in self.encoders:
            xs = layer(xs, attention_mask, pos_emb)

        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module."""

    def __init__(
            self,
            size: int,
            self_attn: nn.Module,
            feed_forward: Optional[nn.Module] = None,
            feed_forward_macaron: Optional[nn.Module] = None,
            drop_prob: float = 0.1,
            normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        self.dropout = nn.Dropout(drop_prob)
        self.size = size
        self.normalize_before = normalize_before
        self.forward = partial(torch_utils.ModuleManager.checkpoint, self, self.forward)

    def forward(self, x, attention_mask, pos_emb, past_kv=None):
        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att = self.self_attn(
            x,
            attention_mask=attention_mask,
            pos_emb=pos_emb,
            cache_fn=partial(attentions.DynamicMemoryAttendWrapper.cache, past_kv=past_kv)
        )
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder module."""

    def __init__(
            self,
            input_size: int = 1024,
            output_size: int = 1024,
            attention_heads: int = 16,
            linear_units: int = 4096,
            num_blocks: int = 14,
            drop_prob: float = 0.1,
            normalize_before: bool = True,
            gradient_checkpoint: bool = False,
    ):
        super().__init__()
        self.output_size = output_size
        self.normalize_before = normalize_before
        self.num_blocks = num_blocks

        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)
        self.embed = LegacyLinearNoSubsampling(
            input_size,
            output_size,
            drop_prob,
        )

        self.encoders = nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                attentions.CrossAttention2D(
                    attention_heads, output_size,
                    attend=attentions.DynamicMemoryAttendWrapper(RelPositionAttend(output_size, attention_heads))
                ),
                transformers.PositionWiseFeedForward(output_size, linear_units, drop_prob=drop_prob, act=nn.ReLU()),
                drop_prob,
                normalize_before,
                gradient_checkpoint=gradient_checkpoint
            ) for _ in range(num_blocks)
        ])

    def forward(self, xs, start_pos=0, past_kvs=None, attention_mask=None):
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + subsample.right_context + 1`
            start_pos (int): current offset in encoder output time stamp
            past_kvs:
            attention_mask:

        Returns:
            torch.Tensor: output of current input xs, with shape (b=1, chunk_size, hidden-dim).
        """
        assert xs.size(0) == 1, 'Only for 1 batch'
        xs, pos_emb = self.embed(xs, start_pos)
        chunk_size = xs.size(1)
        attention_key_size = start_pos + chunk_size
        pos_emb = self.embed.pos_enc.make_embedding(attention_key_size)
        for i, layer in enumerate(self.encoders):
            xs = layer(
                xs,
                attention_mask,
                pos_emb,
                past_kv=past_kvs[i],
            )

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            size: int,
            self_attn: nn.Module,
            feed_forward: nn.Module,
            drop_prob: float = 0.1,
            normalize_before: bool = True,
            gradient_checkpoint=True
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(drop_prob)
        self.size = size
        self.normalize_before = normalize_before
        if gradient_checkpoint:
            self.forward = partial(torch_utils.ModuleManager.checkpoint, self, self.forward)

    def forward(self, x, attention_mask, pos_emb, past_kv=None):
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        x_att = self.self_attn(
            x,
            attention_mask=attention_mask,
            pos_emb=pos_emb,
            cache_fn=partial(attentions.DynamicMemoryAttendWrapper.cache, past_kv=past_kv)
        )
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x


class RelPositionAttend(nn.Module):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860"""

    def __init__(self, model_dim, n_heads, drop_prob=0., **kwargs):
        super().__init__()
        head_dim = model_dim // n_heads
        self.dropout = nn.Dropout(p=drop_prob)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(model_dim, model_dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(n_heads, head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(n_heads, head_dim))
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.n_heads = n_heads

    def forward(self, q, k, v, pos_emb=None, attention_mask=None, **attend_kwargs):
        # (b, h, t, d) -> (b, t, h, d)
        q = q.transpose(1, 2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb)
        p = p.view(n_batch_pos, -1, self.n_heads, self.head_dim).transpose(1, 2)  # (b, h, t, d)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scale = q.shape[-1] ** -0.5
        sim = (matrix_ac + matrix_bd) * scale

        sim = attentions.mask_values(sim, attention_mask, use_min=True)
        attn = torch.softmax(sim, dim=-1)

        attn = self.dropout(attn)
        x = torch.matmul(attn, v)
        return x

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        """
        zero_pad = torch.zeros((x.shape[0], x.shape[1], x.shape[2], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.shape[0], x.shape[1], x.shape[3] + 1, x.shape[2])
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.shape[-1] // 2 + 1]  # only keep the positions from 0 to time2
        return x


class MaskedDiffWithXvec(nn.Module):
    """generate Mel spectrogram"""

    def __init__(
            self,
            input_size: int = 512,
            output_size: int = 80,
            spk_embed_dim: int = 192,
            vocab_size: int = 4096,
            input_frame_rate: int = 50,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_frame_rate = input_frame_rate
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = nn.Linear(spk_embed_dim, output_size)
        self.encoder = ConformerEncoder(
            input_size=512,
            output_size=512,
            attention_heads=8,
            linear_units=2048,
        )
        self.encoder_proj = nn.Linear(self.encoder.output_size, output_size)
        self.decoder = ConditionalCFM()
        self.length_regulator = InterpolateRegulator()

    def forward(
            self,
            token, token_len,
            prompt_token, prompt_token_len,
            prompt_feat, prompt_feat_len,
            embedding, flow_cache=None,
            **kwargs
    ):
        if self.training:
            raise NotImplementedError

        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat speech token and prompt speech token
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = attentions.make_pad_mask(token_len).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = attentions.make_pad_mask(torch.tensor([mel_len1 + mel_len2])).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            cache=flow_cache
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), flow_cache


class ConditionalCFM(nn.Module):
    t_scheduler = 'cosine'
    inference_cfg_rate = 0.7

    def __init__(self):
        super().__init__()
        self.estimator = ConditionalDecoder()

    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=None):
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
            prompt_len:
            cache:

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        if cache is None:
            cache = torch.zeros(1, 80, 0, 2, device=mu.device)

        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, stream=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
                stream
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond, stream=False):
        return self.estimator(x, mask, mu, t, spks, cond, stream=stream)


class Block1D(nn.Module):
    def __init__(self, in_dim, out_dim, groups=8):
        super().__init__()
        self.block = Conv(
            in_dim, out_dim, 3, p=1,
            mode='cna',
            conv_fn=nn.Conv1d,
            norm=nn.GroupNorm(groups, out_dim),
            act=nn.Mish(),
            detail_name=False
        )

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_emb_dim, block_fn=Block1D, groups=8):
        super().__init__()

        self.block1 = block_fn(in_dim, out_dim, groups=groups)
        self.mlp = Linear(time_emb_dim, out_dim, mode='al', act=nn.Mish(), detail_name=False)
        self.block2 = block_fn(out_dim, out_dim, groups=groups)
        self.res_conv = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class ConditionalDecoder(nn.Module):
    def __init__(
            self,
            in_ch=320, out_ch=80, hidden_ch=(256, 256),
            num_mid_blocks=12, nun_attention_blocks=4, num_attention_heads=8, attention_head_dim=64,
            sample_conv_fn=nn.Conv1d, block_fn=Block1D
    ):
        """
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        """
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch

        self.time_embeddings = SinusoidalEmbedding(in_ch, factor=1000)
        time_embed_dim = hidden_ch[0] * 4
        self.time_mlp = TimestepEmbedding(
            input_size=in_ch,
            hidden_size=time_embed_dim,
        )
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        out_ch = in_ch
        for i in range(len(hidden_ch)):
            in_ch = out_ch
            out_ch = hidden_ch[i]
            is_last = i == len(hidden_ch) - 1
            resnet = ResnetBlock(in_dim=in_ch, out_dim=out_ch, time_emb_dim=time_embed_dim, block_fn=block_fn)
            transformer_blocks = transformers.TransformerSequential(
                num_blocks=nun_attention_blocks,
                hidden_size=out_ch,
                num_attention_heads=num_attention_heads,
                ff_hidden_size=out_ch * 4,
                norm_first=True,
                fn_kwargs=dict(
                    model_dim=num_attention_heads * attention_head_dim,
                    query_dim=out_ch,
                    context_dim=out_ch,
                    qkv_fn_kwargs=dict(
                        bias=False
                    ),
                    out_fn_kwargs=dict(
                        bias=True
                    )
                )
            )
            downsample = (
                Downsample1D(out_ch) if not is_last else sample_conv_fn(out_ch, out_ch, 3, padding=1)
            )
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for _ in range(num_mid_blocks):
            in_ch = hidden_ch[-1]
            out_ch = hidden_ch[-1]
            resnet = ResnetBlock(in_dim=in_ch, out_dim=out_ch, time_emb_dim=time_embed_dim, block_fn=block_fn)

            transformer_blocks = transformers.TransformerSequential(
                num_blocks=nun_attention_blocks,
                hidden_size=out_ch,
                num_attention_heads=num_attention_heads,
                ff_hidden_size=out_ch * 4,
                norm_first=True,
                fn_kwargs=dict(
                    model_dim=num_attention_heads * attention_head_dim,
                    query_dim=out_ch,
                    context_dim=out_ch,
                    qkv_fn_kwargs=dict(
                        bias=False
                    ),
                    out_fn_kwargs=dict(
                        bias=True
                    )
                )
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        hidden_ch = hidden_ch[::-1] + (hidden_ch[0],)
        for i in range(len(hidden_ch) - 1):
            in_ch = hidden_ch[i] * 2
            out_ch = hidden_ch[i + 1]
            is_last = i == len(hidden_ch) - 2
            resnet = ResnetBlock(in_dim=in_ch, out_dim=out_ch, time_emb_dim=time_embed_dim, block_fn=block_fn)
            transformer_blocks = transformers.TransformerSequential(
                num_blocks=nun_attention_blocks,
                hidden_size=out_ch,
                num_attention_heads=num_attention_heads,
                ff_hidden_size=out_ch * 4,
                norm_first=True,
                fn_kwargs=dict(
                    model_dim=num_attention_heads * attention_head_dim,
                    query_dim=out_ch,
                    context_dim=out_ch,
                    qkv_fn_kwargs=dict(
                        bias=False
                    ),
                    out_fn_kwargs=dict(
                        bias=True
                    )
                )
            )
            upsample = (
                Upsample(out_ch, op_fn=nn.ConvTranspose1d, use_conv=False) if not is_last else sample_conv_fn(out_ch, out_ch, 3, padding=1)
            )
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))
        self.final_block = block_fn(hidden_ch[-1], hidden_ch[-1])
        self.final_proj = nn.Conv1d(hidden_ch[-1], self.out_channels, 1)

    def forward(self, x, mask, mu, t, spks=None, cond=None, stream=False):
        """Forward pass of the UNet1DConditional model.

        Args:
            x (torch.Tensor): shape (batch_size, in_channels, time)
            mask (_type_): shape (batch_size, 1, time)
            t (_type_): shape (batch_size)
            spks (_type_, optional): shape: (batch_size, condition_channels). Defaults to None.
            cond (_type_, optional): placeholder for future use. Defaults to None.

        """

        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if stream is True:
                attention_mask = add_optional_chunk_mask(x, mask_down.bool(), False, 0, self.static_chunk_size)
            else:
                attention_mask = add_optional_chunk_mask(x, mask_down.bool(), False, 0, 0).repeat(1, x.size(1), 1)
            x = transformer_blocks(
                x,
                attention_mask=attention_mask,
                timestep=t,
            )
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)  # Save hidden states for skip connections
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if stream is True:
                attention_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, 0, self.static_chunk_size)
            else:
                attention_mask = add_optional_chunk_mask(x, mask_mid.bool(), False, 0, 0).repeat(1, x.size(1), 1)
            x = transformer_blocks(
                x,
                attention_mask=attention_mask,
                timestep=t,
            )
            x = rearrange(x, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()
            if stream is True:
                attention_mask = add_optional_chunk_mask(x, mask_up.bool(), False, 0, self.static_chunk_size)
            else:
                attention_mask = add_optional_chunk_mask(x, mask_up.bool(), False, 0, 0).repeat(1, x.size(1), 1)
            x = transformer_blocks(
                x,
                attention_mask=attention_mask,
                timestep=t,
            )
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask


class SinusoidalEmbedding(embeddings.SinusoidalEmbedding):
    def initialize_layers(self):
        # note, different here
        half_dim = self.embedding_dim // 2
        div_term = (torch.arange(half_dim).float() * -(math.log(self.theta) / (half_dim - 1))).exp()
        self.register_buffer('div_term', div_term, persistent=False)

    def forward(self, x):
        dtype = x.dtype
        div_term = self.div_term
        x = x * self.factor
        emb = x[:, None].float() * div_term[None, :]
        # note, different here
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = emb.to(dtype)
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, output_size or hidden_size)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """Residual block module in HiFiGAN/BigVGAN."""

    def __init__(
            self,
            ch: int = 512,
            k: int = 3,
            dilations: List[int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(ch, ch, k, 1, dilation=dilation, padding=self.get_padding(k, dilation))
                )
            )
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(ch, ch, k, 1, dilation=1, padding=self.get_padding(k, 1))
                )
            )
        self.activations1 = nn.ModuleList([Snake(ch, alpha_logscale=False) for _ in range(len(self.convs1))])
        self.activations2 = nn.ModuleList([Snake(ch, alpha_logscale=False) for _ in range(len(self.convs2))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for idx in range(len(self.convs1)):
            remove_weight_norm(self.convs1[idx])
            remove_weight_norm(self.convs2[idx])

    @staticmethod
    def get_padding(kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)


class Snake(nn.Module):
    """Implementation of a sine-based periodic activation function
    References:
        - paper: https://arxiv.org/abs/2006.0819
        - code: https://github.com/EdwardDixon/snake
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        """
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        """
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.eps = 1e-9

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.eps)) * torch.pow(torch.sin(x * alpha), 2)

        return x


class InterpolateRegulator(nn.Module):
    def __init__(
            self,
            channels: int = 80,
            sampling_ratios: Tuple = (1, 1, 1, 1),
            out_channels: int = None,
            groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        model.append(
            nn.Conv1d(channels, out_channels, 1, 1)
        )
        self.model = nn.Sequential(*model)

    def forward(self, x1, x2, mel_len1, mel_len2, input_frame_rate=50):
        if self.training:
            raise NotImplementedError

        # in inference mode, interploate prompt token and token(head/mid/tail) seprately, so we can get a clear separation point of mel
        # NOTE 20 corresponds to token_overlap_len in cosyvoice/cli/model.py
        # x in (B, T, D)
        if x2.shape[1] > 40:
            x2_head = F.interpolate(x2[:, :20].transpose(1, 2).contiguous(), size=int(20 / input_frame_rate * 22050 / 256), mode='linear')
            x2_mid = F.interpolate(x2[:, 20:-20].transpose(1, 2).contiguous(), size=mel_len2 - int(20 / input_frame_rate * 22050 / 256) * 2, mode='linear')
            x2_tail = F.interpolate(x2[:, -20:].transpose(1, 2).contiguous(), size=int(20 / input_frame_rate * 22050 / 256), mode='linear')
            x2 = torch.concat([x2_head, x2_mid, x2_tail], dim=2)
        else:
            x2 = F.interpolate(x2.transpose(1, 2).contiguous(), size=mel_len2, mode='linear')
        if x1.shape[1] != 0:
            x1 = F.interpolate(x1.transpose(1, 2).contiguous(), size=mel_len1, mode='linear')
            x = torch.concat([x1, x2], dim=2)
        else:
            x = x2
        out = self.model(x).transpose(1, 2).contiguous()
        return out, mel_len1 + mel_len2


class SourceModuleHnNSF(nn.Module):
    """ SourceModule for hn-nsf
    SourceModule(sample_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0)
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

    def __init__(self, sample_rate, upsample_scale, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshod=0.):
        super().__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sample_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x):
        # source for harmonic branch
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x.transpose(1, 2))
            sine_wavs = sine_wavs.transpose(1, 2)
            uv = uv.transpose(1, 2)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


class SineGen(nn.Module):
    """Definition of sine generator"""

    def __init__(self, sample_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003, voiced_threshold=0.):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sample_rate = sample_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        # generate uv signal
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        return uv

    def forward(self, f0):
        """
        :param f0: [B, 1, sample_len], Hz
        :return: [B, 1, sample_len]
        """

        F_mat = torch.zeros((f0.size(0), self.harmonic_num + 1, f0.size(-1))).to(f0.device)
        for i in range(self.harmonic_num + 1):
            F_mat[:, i: i + 1, :] = f0 * (i + 1) / self.sample_rate

        theta_mat = 2 * torch.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        u_dist = Uniform(low=-torch.pi, high=torch.pi)
        phase_vec = u_dist.sample(sample_shape=(f0.size(0), self.harmonic_num + 1, 1)).to(F_mat.device)
        phase_vec[:, 0, :] = 0

        # generate sine waveforms
        sine_waves = self.sine_amp * torch.sin(theta_mat + phase_vec)

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


class HiFTGenerator(nn.Module):
    """HiFTNet Generator: Neural Source Filter + ISTFTNet
    https://arxiv.org/abs/2309.09493
    """

    def __init__(
            self,
            in_ch: int = 80,
            hidden_ch: int = 512,
            nb_harmonics: int = 8,
            sample_rate: int = 22050,
            nsf_alpha: float = 0.1,
            nsf_sigma: float = 0.003,
            nsf_voiced_threshold: float = 10,
            upsample_rates: List[int] = (8, 8),
            upsample_kernel_sizes: List[int] = (16, 16),
            istft_params: Dict[str, int] = {"n_fft": 16, "hop_len": 4},
            resblock_kernel_sizes: List[int] = (3, 7, 11),
            resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            source_resblock_kernel_sizes: List[int] = (7, 11),
            source_resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5]],
            lrelu_slope: float = 0.1,
            audio_limit: float = 0.99,
            m_source_fn=SourceModuleHnNSF,
    ):
        super().__init__()

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = m_source_fn(
            sample_rate=sample_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold
        )
        self.f0_upsamp = nn.Upsample(scale_factor=np.prod(upsample_rates) * istft_params["hop_len"])

        self.conv_pre = weight_norm(
            nn.Conv1d(in_ch, hidden_ch, 7, 1, padding=3)
        )

        # Up
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(hidden_ch // (2 ** i), hidden_ch // (2 ** (i + 1)), k, u, padding=(k - u) // 2)
                )
            )

        # Down
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)):
            if u == 1:
                self.source_downs.append(
                    nn.Conv1d(istft_params["n_fft"] + 2, hidden_ch // (2 ** (i + 1)), 1, 1)
                )
            else:
                self.source_downs.append(
                    nn.Conv1d(istft_params["n_fft"] + 2, hidden_ch // (2 ** (i + 1)), u * 2, u, padding=(u // 2))
                )

            self.source_resblocks.append(
                ResBlock(hidden_ch // (2 ** (i + 1)), k, d)
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = hidden_ch // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, istft_params["n_fft"] + 2, 7, 1, padding=3))
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft_window = torch.from_numpy(get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32))
        self.f0_predictor = ConvRNNF0Predictor()

    def _stft(self, x):
        spec = torch.stft(
            x,
            self.istft_params["n_fft"], self.istft_params["hop_len"], self.istft_params["n_fft"], window=self.stft_window.to(x.device),
            return_complex=True
        )
        spec = torch.view_as_real(spec)  # [B, F, TT, 2]
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(torch.complex(real, img), self.istft_params["n_fft"], self.istft_params["hop_len"],
                                        self.istft_params["n_fft"], window=self.stft_window.to(magnitude.device))
        return inverse_transform

    def decode(self, x: torch.Tensor, s: torch.Tensor = torch.zeros(1, 1, 0)) -> torch.Tensor:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, :self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1:, :])  # actually, sin is redundancy

        x = self._istft(magnitude, phase)
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    def forward(self, speech_feat: torch.Tensor, cache_source: torch.Tensor = torch.zeros(1, 1, 0)):
        if self.training:
            raise NotImplementedError

        # mel->f0
        f0 = self.f0_predictor(speech_feat)
        # f0->source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        # use cache_source to avoid glitch
        if cache_source.shape[2] != 0:
            s[:, :, :cache_source.shape[2]] = cache_source
        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, s


class ConvRNNF0Predictor(nn.Module):
    def __init__(self, num_class=1, in_ch=80, out_ch=512):
        super().__init__()

        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
            ),
            nn.ELU(),
            weight_norm(
                nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
            ),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=out_ch, out_features=self.num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.condnet(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


def add_optional_chunk_mask(
        xs: torch.Tensor,
        masks: torch.Tensor,
        use_dynamic_chunk: bool,
        decoding_chunk_size: int,
        static_chunk_size: int,
        enable_full_context: bool = True
):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        masks (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        enable_full_context (bool):
            True: chunk size is either [1, 25] or full context(max_len)
            False: chunk size ~ U[1, 25]

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.shape[1]
        if decoding_chunk_size < 0:
            chunk_size = max_len
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = torch.randint(1, max_len, (1,)).item()
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
        chunk_masks = attentions.make_chunk_mask(xs.shape[1], chunk_size).to(xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)

    elif static_chunk_size > 0:
        chunk_masks = attentions.make_chunk_mask(xs.shape[1], static_chunk_size).to(xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)

    else:
        chunk_masks = masks

    assert chunk_masks.dtype == torch.bool

    if (chunk_masks.sum(dim=-1) == 0).sum().item() != 0:
        chunk_masks[chunk_masks.sum(dim=-1) == 0] = True
    return chunk_masks
