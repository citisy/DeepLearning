from ..text_pretrain.bge_m3 import *
from metrics.regression import Similarity


class Model(Model):
    def inference(
            self,
            text_ids1, attention_mask1,
            text_ids2, attention_mask2,
            output1=None, output2=None,
            **kwargs
    ):
        if output1 is None:
            output1 = super().inference(text_ids1, attention_mask1, **kwargs)

        if output2 is None:
            output2 = super().inference(text_ids2, attention_mask2, **kwargs)

        output1.update(
            text_ids=text_ids1,
            attention_mask=attention_mask1,
        )
        output2.update(
            text_ids=text_ids2,
            attention_mask=attention_mask2,
        )
        return self.post_process(output1, output2)

    def post_process(self, output1, output2):
        output = {}
        if 'dense_vecs' in output1:
            dense_vecs1 = output1['dense_vecs']
            dense_vecs2 = output2['dense_vecs']
            scores = dense_vecs1 @ dense_vecs2.T
            output.update(
                dense_vecs1=dense_vecs1,
                dense_vecs2=dense_vecs2,
                dense_scores=scores,
            )

        if 'sparse_vecs' in output1:
            sparse_vecs1 = output1['sparse_vecs']
            text_ids1 = output1['text_ids']
            sparse_vecs2 = output2['sparse_vecs']
            text_ids2 = output2['text_ids']
            scores = Similarity.lexical_matching_score(sparse_vecs1, sparse_vecs2, text_ids1, text_ids2, filter_tokens=(0, 1, 2))
            scores = torch.tensor(scores)
            output.update(
                sparse_vecs1=sparse_vecs1,
                sparse_vecs2=sparse_vecs2,
                sparse_scores=scores,
            )

        if 'colbert_vecs' in output1:
            colbert_vecs1 = output1['colbert_vecs']
            attention_mask1 = output1['attention_mask']
            colbert_vecs2 = output2['colbert_vecs']
            attention_mask2 = output2['attention_mask']

            tokens_num1 = torch.sum(attention_mask1, dim=1)
            tokens_num2 = torch.sum(attention_mask2, dim=1)

            scores = []
            for vecs1, num1 in zip(colbert_vecs1, tokens_num1):
                score = []
                for vecs2, num2 in zip(colbert_vecs2, tokens_num2):
                    q_reps = vecs1[:num1 - 1]
                    p_reps = vecs2[:num2 - 1]
                    token_scores = torch.einsum('in,jn->ij', q_reps, p_reps)
                    s, _ = token_scores.max(-1)
                    s = torch.sum(s) / q_reps.size(0)
                    score.append(s)
                scores.append(score)
            scores = torch.tensor(scores, device=colbert_vecs1.device)

            output.update(
                colbert_vecs1=colbert_vecs1,
                colbert_vecs2=colbert_vecs2,
                colbert_scores=scores,
            )

        return output
