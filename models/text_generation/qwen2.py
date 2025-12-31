from ..text_pretrain.qwen2 import *
from data_parse.nl_data_parse.pre_process.decoder import beam_search


class Model(Model):
    def inference(
            self,
            text_ids, content_generator=True, seq_lens=None, past_kvs=None,
            **decode_kwargs
    ):
        if past_kvs is None:
            past_kvs = self.make_caches()

        model_outputs = beam_search(text_ids, seq_lens, self.decode, eos_ids=self.eos_ids, past_kvs=past_kvs, **decode_kwargs)

        torch_utils.ModuleManager.torch_gc()

        return dict(
            past_kvs=past_kvs,
            **model_outputs
        )
