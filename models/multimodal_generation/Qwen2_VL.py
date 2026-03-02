from ..multimodal_pretrain.Qwen2_VL import *
from data_parse.nl_data_parse.pre_process.decoder import beam_search


class Model(Model):
    def inference(
            self,
            text_ids, generate_content=True, seq_lens=None, vlm_past_kvs=None, caches=None,
            **decode_kwargs
    ):
        if vlm_past_kvs is None:
            vlm_past_kvs = self.vlm.make_caches()

        if caches is None:
            caches = dict()

        if generate_content:
            model_outputs = beam_search(text_ids, seq_lens, self.decode, eos_ids=self.eos_ids, past_kvs=vlm_past_kvs, caches=caches, **decode_kwargs)

            return dict(
                **model_outputs,
                vlm_past_kvs=vlm_past_kvs,
                caches=caches
            )
        else:
            return dict(
                logits=self.decode(text_ids, **decode_kwargs),
                vlm_past_kvs=vlm_past_kvs,
                caches=caches
            )