from ..text_pretrain.qwen2 import *


class Model(Model):
    def inference(
            self,
            x, content_generator=True, seq_lens=None, past_kvs=None,
            **decode_kwargs
    ):
        if past_kvs is None:
            past_kvs = self.make_caches()

        preds = beam_search(x, seq_lens, self.decode, eos_ids=self.eos_ids, past_kvs=past_kvs, **decode_kwargs)

        torch_utils.ModuleManager.torch_gc()

        return dict(
            preds=preds,
            past_kvs=past_kvs
        )
