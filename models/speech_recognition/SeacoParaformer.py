import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from . import BiCifParaformer, Paraformer
from .. import bundles


class Config(bundles.Config):
    encoder = dict(
        input_size=560
    )

    seaco_decoder = dict(
        linear_units=1024,
        num_blocks=4,
        kernel_size=21,
        att_layer_num=6,
        use_embed=False,
        use_output_layer=False
    )

    decoder = dict(
        seaco_decoder_config=seaco_decoder
    )

    default_model = ''

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            '': dict(
                encoder_config=cls.encoder,
                decoder_config=cls.decoder,
            )
        }


class WeightConverter(Paraformer.WeightConverter):
    """https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"""
    decoder_convert_dict = {
        **Paraformer.WeightConverter.decoder_convert_dict,
        'seaco_decoder.{0}.feed_forward.w_1': 'seaco_decoder.{0}.feed_forward.0.linear',
        'seaco_decoder.{0}.feed_forward.w_2': 'seaco_decoder.{0}.feed_forward.1.linear',
        'seaco_decoder.{0}.feed_forward.norm': 'seaco_decoder.{0}.feed_forward.0.norm',
    }


class WeightLoader(Paraformer.WeightLoader):
    pass


class Model(BiCifParaformer.Model):
    """refer to:
    paper:
        - [SeACo-Paraformer: A Non-Autoregressive ASR System with Flexible and Effective Hotword Customization Ability](https://arxiv.org/abs/2308.03266)
    """
    NO_BIAS = 8377

    def make_decoder(self, input_size=512, bias_encoder_drop_prob=0, bias_encoder_bid=False, seaco_decoder_config=dict(), **decoder_config):
        self.bias_encoder = nn.LSTM(
            input_size,
            input_size,
            2,
            batch_first=True,
            dropout=bias_encoder_drop_prob,
            bidirectional=bias_encoder_bid,
        )
        self.decoder = Paraformer.SANMDecoder(
            vocab_size=self.vocab_size,
            encoder_output_size=self.encoder.output_size,
            **decoder_config,
        )
        self.seaco_decoder = Paraformer.SANMDecoder(
            vocab_size=self.vocab_size,
            encoder_output_size=input_size,
            **seaco_decoder_config,
        )
        self.hotword_output_layer = nn.Linear(input_size, self.vocab_size)

    def decode(
            self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens,
            hotword_ids=None, hotword_lens=None, nfilter=50, seaco_weight=1.0,
            **kwargs
    ):
        decoder_outputs = self.decoder(encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens)
        decoder_outputs['x'] = F.log_softmax(decoder_outputs['x'], dim=-1)

        if hotword_ids is not None:
            x = decoder_outputs['x']
            decoder_hidden = decoder_outputs['hidden']
            selected = self._hotword_representation(hotword_ids, torch.Tensor(hotword_lens).int().to(encoder_out.device))

            contextual_info = selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
            num_hot_word = contextual_info.shape[1]
            _contextual_length = torch.Tensor([num_hot_word]).int().repeat(encoder_out.shape[0]).to(encoder_out.device)

            # ASF Core
            if 0 < nfilter < num_hot_word:
                hotword_scores = self.seaco_decoder.forward_asf6(contextual_info, _contextual_length, decoder_hidden, ys_pad_lens)
                hotword_scores = hotword_scores[0].sum(0).sum(0)
                add_filter = torch.topk(hotword_scores, min(nfilter, num_hot_word - 1))[1].tolist()
                # note, must contain sos token
                add_filter.append(len(hotword_ids) - 1)
                selected = selected[add_filter]
                contextual_info = selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
                num_hot_word = contextual_info.shape[1]
                _contextual_length = torch.Tensor([num_hot_word]).int().repeat(encoder_out.shape[0]).to(encoder_out.device)

            # SeACo Core
            cif_attended = self.seaco_decoder(
                contextual_info, _contextual_length, sematic_embeds, ys_pad_lens
            )['x']
            dec_attended = self.seaco_decoder(
                contextual_info, _contextual_length, decoder_hidden, ys_pad_lens
            )['x']
            merged = cif_attended + dec_attended

            dha_output = self.hotword_output_layer(merged)  # remove the last token in loss calculation
            dha_pred = torch.log_softmax(dha_output, dim=-1)

            merged_pred = self._merge_res(x, dha_pred, seaco_weight)
            decoder_outputs['x'] = merged_pred

        return decoder_outputs

    def _hotword_representation(self, hotword_pad, hotword_lengths):
        hw_embed = self.decoder.embed(hotword_pad)
        hw_embed = pack_padded_sequence(
            hw_embed,
            hotword_lengths.cpu().type(torch.int64),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_rnn_output, _ = self.bias_encoder(hw_embed)
        rnn_output = pad_packed_sequence(packed_rnn_output, batch_first=True)[0]
        hw_hidden = rnn_output
        _ind = np.arange(0, hw_hidden.shape[0]).tolist()
        selected = hw_hidden[_ind, [i - 1 for i in hotword_lengths.detach().cpu().tolist()]]
        return selected

    def _merge_res(self, dec_output, dha_output, seaco_weight):
        lmbd = torch.Tensor([seaco_weight] * dha_output.shape[0])
        dha_ids = dha_output.max(-1)[-1]  # [0]
        dha_mask = (dha_ids == self.NO_BIAS).int().unsqueeze(-1)
        a = (1 - lmbd) / lmbd
        b = 1 / lmbd
        a, b = a.to(dec_output.device), b.to(dec_output.device)
        dha_mask = (dha_mask + a.reshape(-1, 1, 1)) / b.reshape(-1, 1, 1)
        # logits = dec_output * dha_mask + dha_output[:,:,:-1] * (1-dha_mask)
        logits = dec_output * dha_mask + dha_output[:, :, :] * (1 - dha_mask)
        return logits
