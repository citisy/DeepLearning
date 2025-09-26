import torch
from torch import nn

from data_parse.nl_data_parse.pre_process import chunker
from utils import torch_utils
from .. import bundles
from ..speech_recognition import Paraformer


class Config(bundles.Config):
    encoder = dict(
        input_size=256,
        output_size=256,
        attention_heads=8,
        linear_units=1024,
        num_blocks=4,
    )

    default_model = ''

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            '': dict(
                encoder_configs=cls.encoder
            )
        }


class WeightConverter:
    @classmethod
    def from_official(cls, state_dict):
        convert_dict = {
            **Paraformer.WeightConverter.encoder_convert_dict,
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


class Model(nn.Module):
    """
    refer to:
    paper:
        (CT-Transformer: Controllable time-delay transformer for real-time punctuation prediction and disfluency detection)[https://arxiv.org/pdf/2003.01309.pdf]
    code:
        `funasr.models.ct_transformer.model.CTTransformer`
    """

    vocab_size: int = 272727
    punc_list: list = ['<unk>', '_', '，', '。', '？', '、']
    punc_weight: list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    embed_unit: int = 256
    att_unit: int = 256
    sentence_end_id: int = 3

    def __init__(self, encoder_configs=Config.encoder, model_conf={}, **kwargs):
        super().__init__()
        self.__dict__.update(model_conf)

        punc_size = len(self.punc_list)
        if self.punc_weight is None:
            self.punc_weight = [1] * punc_size

        self.embed = nn.Embedding(self.vocab_size, self.embed_unit)
        self.encoder = Paraformer.SANMEncoder(**encoder_configs)
        self.decoder = nn.Linear(self.att_unit, punc_size)

    @property
    def device(self):
        return torch_utils.ModuleInfo.possible_device(self)

    def forward(self, *args, **kwargs):
        if self.training:
            raise NotImplementedError
        else:
            return self.inference(*args, **kwargs)

    def inference(
            self,
            text_ids,  # (b, s)
            caches=[],
            chunk_size=20,
            max_sentence_len=200,   # {3,4} is the end flag of a sequence.
            is_streaming_input=False,
            **kwargs,
    ):
        if not caches:
            caches = [torch.empty(0, dtype=torch.long, device=self.device) for _ in text_ids]
        preds = []
        cache_ids = []
        for ids, _cache_ids in zip(text_ids, caches):
            chunk_ids = chunker.WindowToChunkedParagraphs(max_len=chunk_size).from_paragraph(ids)
            n_chunk_seq = len(chunk_ids)

            _preds = []
            for si in range(n_chunk_seq):
                per_chunk_ids = chunk_ids[si]
                per_chunk_ids = torch.cat((_cache_ids, per_chunk_ids), dim=0)

                y = self.punc_forward(
                    ids=per_chunk_ids[None],
                    seq_lens=torch.tensor([len(per_chunk_ids)]).to(self.device)
                )
                _, indices = y.view(-1, y.shape[-1]).topk(1, dim=1)
                punc_ids = indices[:, 0]

                # Search for the last Period/QuestionMark as cache
                if is_streaming_input or si < n_chunk_seq - 1:
                    eos_idx = -1
                    last_comma_index = -1
                    for i in range(2, len(punc_ids) - 1)[::-1]:
                        if punc_ids[i] in (3, 4):
                            eos_idx = i
                            break

                        if last_comma_index < 0 and punc_ids[i] == 2:
                            last_comma_index = i

                    if eos_idx < 0 <= last_comma_index and len(per_chunk_ids) > max_sentence_len:
                        # The sentence it too long, cut off at a comma.
                        eos_idx = last_comma_index
                        punc_ids[eos_idx] = self.sentence_end_id

                    _cache_ids = per_chunk_ids[eos_idx + 1:]
                    punc_ids = punc_ids[0: eos_idx + 1]

                if not is_streaming_input and si == n_chunk_seq - 1:
                    # Add Period for the end of the sentence
                    if punc_ids[-1] == 1:
                        punc_ids[-1] = 2

                    _cache_ids = torch.empty(0, dtype=torch.long, device=self.device)

                _preds.append(punc_ids)

            _preds = torch.cat(_preds)
            preds.append(_preds)
            cache_ids.append(_cache_ids)
        return dict(
            preds=preds,
            cache_ids=cache_ids,
        )

    def punc_forward(self, ids: torch.Tensor, seq_lens: torch.Tensor, **kwargs):
        x = self.embed(ids)
        h = self.encoder(x, seq_lens)
        y = self.decoder(h)
        return y
