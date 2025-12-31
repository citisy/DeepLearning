from ..text_pretrain.gpt2 import *


class Model(Model):
    def inference(
            self,
            x, seq_lens=None,
            **decode_kwargs
    ):
        return beam_search(x, seq_lens, self.decode, **decode_kwargs)
