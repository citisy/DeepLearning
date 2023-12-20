from . import classification
from utils import nlp_utils


class LineConfusionMatrix:
    def tp(self, true, pred, **kwargs):
        tp = [t == p for t, p in zip(true, pred)]
        return dict(
            tp=tp,
            acc_tp=sum(tp)
        )

    def fp(self, true, pred, **kwargs):
        """useless"""
        fp = [t != p for t, p in zip(true, pred)]
        return dict(
            fp=fp,
            acc_fp=sum(fp)
        )

    def cp(self, true, **kwargs):
        return dict(
            cp=[True] * len(true),
            acc_cp=len(true)
        )

    def op(self, pred, **kwargs):
        return dict(
            op=[True] * len(pred),
            acc_op=len(pred)
        )


class WordConfusionMatrix:
    """ROUGE-N
    see also `rouge.Rouge`"""

    def __init__(self, n_gram=2):
        self.n_gram = n_gram

    def make_n_grams(self, lines):
        return nlp_utils.Sequencer.n_grams(lines, n_gram=self.n_gram)

    def tp(self, true=None, pred=None, true_with_n_grams=None, pred_with_n_grams=None, **kwargs):
        """

        Args:
            true (List[list]): text list after cut
            pred (List[list]): text list after cut
            true_with_n_grams (List[set]):
            pred_with_n_grams (List[set]):

        Returns:

        """
        true_with_n_grams = true_with_n_grams or self.make_n_grams(true)
        pred_with_n_grams = pred_with_n_grams or self.make_n_grams(pred)
        tp = [len(t & p) for t, p in zip(true_with_n_grams, pred_with_n_grams)]

        return dict(
            tp=tp,
            acc_tp=sum(tp),
            true_with_n_grams=true_with_n_grams,
            pred_with_n_grams=pred_with_n_grams
        )

    def fp(self, true=None, pred=None, true_with_n_grams=None, pred_with_n_grams=None, **kwargs):
        """useless"""
        true_with_n_grams = true_with_n_grams or self.make_n_grams(true)
        pred_with_n_grams = pred_with_n_grams or self.make_n_grams(pred)
        fp = [len(t - p) for t, p in zip(true_with_n_grams, pred_with_n_grams)]

        return dict(
            fp=fp,
            acc_fp=sum(fp),
            true_with_n_grams=true_with_n_grams,
            pred_with_n_grams=pred_with_n_grams
        )

    def cp(self, true=None, true_with_n_grams=None, **kwargs):
        true_with_n_grams = true_with_n_grams or self.make_n_grams(true)
        cp = [len(t) for t in true_with_n_grams]

        return dict(
            cp=cp,
            acc_cp=sum(cp),
            true_with_n_grams=true_with_n_grams,
        )

    def op(self, pred=None, pred_with_n_grams=None, **kwargs):
        pred_with_n_grams = pred_with_n_grams or self.make_n_grams(pred)
        op = [len(p) for p in pred_with_n_grams]

        return dict(
            op=op,
            acc_op=sum(op),
            pred_with_n_grams=pred_with_n_grams,
        )


class WordLCSConfusionMatrix:
    """ROUGE-L and ROUGE-W
    see also `rouge.Rouge`"""

    def __init__(self, is_cut=False, filter_blank=True, lcs_method=None):
        self.is_cut = is_cut
        self.filter_blank = filter_blank
        self.lcs = lcs_method or nlp_utils.Sequencer.longest_common_subsequence

    def tp(self, true=None, pred=None, tp=None, **kwargs):
        """

        Args:
            true (List[list]): text list after cut
            pred (List[list]): text list after cut
            tp:
            **kwargs:

        Returns:

        """
        tp = tp if tp is not None else [self.lcs(t, p)['score'] for t, p in zip(true, pred)]

        return dict(
            tp=tp,
            acc_tp=sum(tp),
        )

    def cp(self, true=None, **kwargs):
        cp = [len(t) for t in true]

        return dict(
            cp=cp,
            acc_cp=sum(cp),
        )

    def op(self, pred=None, **kwargs):
        op = [len(p) for p in pred]

        return dict(
            op=sum(op),
            acc_op=sum(op),
        )


class CharConfusionMatrix:
    def tp(self, true, pred, **kwargs):
        tp = [len(set(t) & set(p)) for t, p in zip(true, pred)]
        return dict(
            tp=tp,
            acc_tp=sum(tp)
        )

    def fp(self, true, pred, **kwargs):
        fp = [len(set(t) - set(p)) for t, p in zip(true, pred)]
        return dict(
            fp=fp,
            acc_fp=sum(fp)
        )

    def cp(self, true, **kwargs):
        cp = [len(set(t)) for t in true]
        return dict(
            cp=cp,
            acc_cp=sum(cp)
        )

    def op(self, pred, **kwargs):
        op = [len(set(p)) for p in pred]
        return dict(
            op=op,
            acc_op=sum(op)
        )


class PR(classification.PR):
    def __init__(self, return_more_info=False, confusion_method=None, **confusion_method_kwarg):
        super().__init__(return_more_info=return_more_info, confusion_method=confusion_method or LineConfusionMatrix, **confusion_method_kwarg)


class TopMetric(classification.TopMetric):
    """
    only support `f_measure` or `f1`

    Usage:
        .. code-block:: python

            from utils import nlp_utils

            det_text, gt_text = ['your det text'], ['your gt text']

            # char fine-grained
            ret = TopMetric(confusion_method=CharConfusionMatrix).f_measure(det_text, gt_text)

            # word fine-grained by n gram algorithm
            ret = TopMetric(confusion_method=WordConfusionMatrix, n_gram=2).f_measure(det_text, gt_text)

            # word fine-grained by lcs algorithm
            ret = TopMetric(confusion_method=WordLCSConfusionMatrix).f_measure(det_text, gt_text)
            ret = TopMetric(confusion_method=WordLCSConfusionMatrix, lcs_method=nlp_utils.Sequence.weighted_longest_common_subsequence).f_measure(det_text, gt_text)

            # line fine-grained
            ret = TopMetric(confusion_method=LineConfusionMatrix).f_measure(det_text, gt_text)

            # if your text is after cut, set `is_cut=True`
            det_text, gt_text = ['your', 'det', 'text'], ['your', 'gt', 'text']
            ret = TopMetric(is_cut=True).f_measure(det_text, gt_text)

    """

    def __init__(self, pr_method=None, **pr_method_kwarg):
        super().__init__(pr_method=pr_method or PR, **pr_method_kwarg)


pr = PR()
top_metric = TopMetric()
