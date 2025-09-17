from ..text_pretrain.llama import *


class Model(Model):
    def inference(
            self,
            x, seq_lens=None,
            **decode_kwargs
    ):
        preds = beam_search(x, seq_lens, self.decode, eos_ids=self.eos_ids, **decode_kwargs)
        return {'preds': preds}
