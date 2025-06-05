import numpy as np
import torch
from torch import nn

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
    ignore_id: int = 0
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
            return self.post_process(*args, **kwargs)

    def post_process(
            self,
            batch_tokens,
            batch_ids,
            split_size=20,
            cache_pop_trigger_limit=200,
            **kwargs,
    ):
        results = []
        for tokens, ids in zip(batch_tokens, batch_ids):
            mini_sentences = self.split_to_mini_sentence(tokens, split_size)
            mini_sentences_id = self.split_to_mini_sentence(ids, split_size)
            assert len(mini_sentences) == len(mini_sentences_id)
            cache_sent = []
            cache_sent_id = np.array([], dtype="int32")
            new_mini_sentence = ""
            new_mini_sentence_punc = []
            new_mini_sentence_out = ''

            punc_array = None
            for mini_sentence_i in range(len(mini_sentences)):
                mini_sentence = mini_sentences[mini_sentence_i]
                mini_sentence_id = mini_sentences_id[mini_sentence_i]
                mini_sentence = cache_sent + mini_sentence
                mini_sentence_id = np.concatenate((cache_sent_id, mini_sentence_id), axis=0)

                y, _ = self.punc_forward(
                    ids=torch.unsqueeze(torch.from_numpy(mini_sentence_id), 0).to(self.device),
                    seq_lens=torch.from_numpy(np.array([len(mini_sentence_id)], dtype="int32")).to(self.device)
                )
                _, indices = y.view(-1, y.shape[-1]).topk(1, dim=1)
                punctuations = torch.squeeze(indices, dim=1)
                assert punctuations.size()[0] == len(mini_sentence)

                # Search for the last Period/QuestionMark as cache
                if mini_sentence_i < len(mini_sentences) - 1:
                    sentence_end = -1
                    last_comma_index = -1
                    for i in range(len(punctuations) - 2, 1, -1):
                        if (
                                self.punc_list[punctuations[i]] == "。"
                                or self.punc_list[punctuations[i]] == "？"
                        ):
                            sentence_end = i
                            break
                        if last_comma_index < 0 and self.punc_list[punctuations[i]] == "，":
                            last_comma_index = i

                    if (
                            sentence_end < 0 <= last_comma_index
                            and len(mini_sentence) > cache_pop_trigger_limit
                    ):
                        # The sentence it too long, cut off at a comma.
                        sentence_end = last_comma_index
                        punctuations[sentence_end] = self.sentence_end_id
                    cache_sent = mini_sentence[sentence_end + 1:]
                    cache_sent_id = mini_sentence_id[sentence_end + 1:]
                    mini_sentence = mini_sentence[0: sentence_end + 1]
                    punctuations = punctuations[0: sentence_end + 1]

                punctuations_np = punctuations.cpu().numpy()
                new_mini_sentence_punc += [int(x) for x in punctuations_np]
                words_with_punc = []
                for i in range(len(mini_sentence)):
                    if (
                            i == 0
                            or self.punc_list[punctuations[i - 1]] == "。"
                            or self.punc_list[punctuations[i - 1]] == "？"
                    ) and len(mini_sentence[i][0].encode()) == 1:
                        mini_sentence[i] = mini_sentence[i].capitalize()
                    if i == 0:
                        if len(mini_sentence[i][0].encode()) == 1:
                            mini_sentence[i] = " " + mini_sentence[i]
                    if i > 0:
                        if (
                                len(mini_sentence[i][0].encode()) == 1
                                and len(mini_sentence[i - 1][0].encode()) == 1
                        ):
                            mini_sentence[i] = " " + mini_sentence[i]
                    words_with_punc.append(mini_sentence[i])
                    if self.punc_list[punctuations[i]] != "_":
                        punc_res = self.punc_list[punctuations[i]]
                        if len(mini_sentence[i][0].encode()) == 1:
                            if punc_res == "，":
                                punc_res = ","
                            elif punc_res == "。":
                                punc_res = "."
                            elif punc_res == "？":
                                punc_res = "?"
                        words_with_punc.append(punc_res)
                new_mini_sentence += "".join(words_with_punc)
                # Add Period for the end of the sentence
                new_mini_sentence_out = new_mini_sentence
                if mini_sentence_i == len(mini_sentences) - 1:
                    if new_mini_sentence[-1] == "，" or new_mini_sentence[-1] == "、":
                        new_mini_sentence_out = new_mini_sentence[:-1] + "。"
                    elif new_mini_sentence[-1] == ",":
                        new_mini_sentence_out = new_mini_sentence[:-1] + "."
                    elif (
                            new_mini_sentence[-1] != "。"
                            and new_mini_sentence[-1] != "？"
                            and len(new_mini_sentence[-1].encode()) != 1
                    ):
                        new_mini_sentence_out = new_mini_sentence + "。"
                        if len(punctuations):
                            punctuations[-1] = 2
                    elif (
                            new_mini_sentence[-1] != "."
                            and new_mini_sentence[-1] != "?"
                            and len(new_mini_sentence[-1].encode()) == 1
                    ):
                        new_mini_sentence_out = new_mini_sentence + "."
                        if len(punctuations):
                            punctuations[-1] = 2
                # keep a punctuations array for punc segment
                if punc_array is None:
                    punc_array = punctuations
                else:
                    punc_array = torch.cat([punc_array, punctuations], dim=0)

            results.append({"text": new_mini_sentence_out, "punc_array": punc_array})
        return results

    @staticmethod
    def split_to_mini_sentence(words: list, word_limit: int = 20):
        assert word_limit > 1
        if len(words) <= word_limit:
            return [words]
        sentences = []
        length = len(words)
        sentence_len = length // word_limit
        for i in range(sentence_len):
            sentences.append(words[i * word_limit: (i + 1) * word_limit])
        if length % word_limit > 0:
            sentences.append(words[sentence_len * word_limit:])
        return sentences

    def punc_forward(self, ids: torch.Tensor, seq_lens: torch.Tensor, **kwargs):
        x = self.embed(ids)
        h, _ = self.encoder(x, seq_lens)
        y = self.decoder(h)
        return y, None
