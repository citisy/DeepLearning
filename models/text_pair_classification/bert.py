from ..text_pretrain.bert import Model as Model_, Config


class Model(Model_):
    def __init__(self, vocab_size, seq_len, sp_tag_dict, bert_config=Config.base):
        super().__init__(vocab_size, seq_len, sp_tag_dict, bert_config=bert_config, n_segment=2, nsp_out_features=2, is_mlm=False)
