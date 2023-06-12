import numpy as np
from . import classifier
from utils import nlp_utils


class LineConfusionMatrix:
    def tp(self, true, pred, **kwargs):
        return dict(
            tp=sum(1 for t, p in zip(true, pred) if t == p)
        )

    def fp(self, true, pred, **kwargs):
        return dict(
            fp=sum(1 for t, p in zip(true, pred) if t != p)
        )

    def cp(self, true, **kwargs):
        return dict(
            cp=len(true)
        )

    def op(self, pred, **kwargs):
        return dict(
            op=len(pred)
        )


class WordConfusionMatrix:
    """ROUGE-N
    see also `rouge.Rouge`"""

    def __init__(self, n_gram=2, is_cut=False, filter_blank=True):
        self.is_cut = is_cut
        self.filter_blank = filter_blank
        self.n_gram = n_gram

    def make_n_grams(self, lines):
        if not self.is_cut:
            lines = nlp_utils.cut_word_by_jieba(lines, self.filter_blank)

        return nlp_utils.Sequence.n_grams(lines, n_gram=self.n_gram)

    def tp(self, true=None, pred=None, true_with_n_grams=None, pred_with_n_grams=None, **kwargs):
        """

        Args:
            true (List[list]):
            pred (List[list]):
            true_with_n_grams (List[set]):
            pred_with_n_grams (List[set]):

        Returns:

        """
        true_with_n_grams = true_with_n_grams or self.make_n_grams(true)
        pred_with_n_grams = pred_with_n_grams or self.make_n_grams(pred)
        s = sum(len(t & p) for t, p in zip(true_with_n_grams, pred_with_n_grams))

        return dict(
            tp=s,
            true_with_n_grams=true_with_n_grams,
            pred_with_n_grams=pred_with_n_grams
        )

    def fp(self, true=None, pred=None, true_with_n_grams=None, pred_with_n_grams=None, **kwargs):
        true_with_n_grams = true_with_n_grams or self.make_n_grams(true)
        pred_with_n_grams = pred_with_n_grams or self.make_n_grams(pred)
        s = sum(len(t - p) for t, p in zip(true_with_n_grams, pred_with_n_grams))

        return dict(
            fp=s,
            true_with_n_grams=true_with_n_grams,
            pred_with_n_grams=pred_with_n_grams
        )

    def cp(self, true=None, true_with_n_grams=None, **kwargs):
        true_with_n_grams = true_with_n_grams or self.make_n_grams(true)
        s = sum(len(t) for t in true_with_n_grams)

        return dict(
            cp=s,
            true_with_n_grams=true_with_n_grams,
        )

    def op(self, pred=None, pred_with_n_grams=None, **kwargs):
        pred_with_n_grams = pred_with_n_grams or self.make_n_grams(pred)
        s = sum(len(p) for p in pred_with_n_grams)

        return dict(
            op=s,
            pred_with_n_grams=pred_with_n_grams,
        )


class WordLCSConfusionMatrix:
    """ROUGE-L and ROUGE-W
    see also `rouge.Rouge`"""

    def __init__(self, is_cut=False, filter_blank=True, lcs_method=None):
        self.is_cut = is_cut
        self.filter_blank = filter_blank
        self.lcs = lcs_method or nlp_utils.Sequence.longest_common_subsequence

    def tp(self, true=None, pred=None, true_cut=None, pred_cut=None, lcs=None, **kwargs):
        true_cut = true_cut or nlp_utils.cut_word_by_jieba(true, self.filter_blank) if not self.is_cut else true
        pred_cut = pred_cut or nlp_utils.cut_word_by_jieba(pred, self.filter_blank) if not self.is_cut else pred
        lcs = lcs if lcs is not None else [self.lcs(t, p)['score'] for t, p in zip(true_cut, pred_cut)]
        s = sum(lcs)

        return dict(
            tp=s,
            lcs=lcs,
            true_cut=true_cut,
            pred_cut=pred_cut
        )

    def fp(self, true=None, pred=None, true_cut=None, pred_cut=None, lcs=None, **kwargs):
        true_cut = true_cut or nlp_utils.cut_word_by_jieba(true, self.filter_blank) if not self.is_cut else true
        pred_cut = pred_cut or nlp_utils.cut_word_by_jieba(pred, self.filter_blank) if not self.is_cut else pred
        lcs = lcs if lcs is not None else [self.lcs(t, p)['score'] for t, p in zip(true_cut, pred_cut)]
        s = sum(lcs)

        return dict(
            fp=s,
            lcs=lcs,
            true_cut=true_cut,
            pred_cut=pred_cut
        )

    def cp(self, true=None, true_cut=None, **kwargs):
        true_cut = true_cut or nlp_utils.cut_word_by_jieba(true, self.filter_blank) if not self.is_cut else true
        s = sum(len(t) for t in true_cut)

        return dict(
            cp=s,
            true_cut=true_cut,
        )

    def op(self, pred=None, pred_cut=None, **kwargs):
        pred_cut = pred_cut or nlp_utils.cut_word_by_jieba(pred, self.filter_blank) if not self.is_cut else pred
        s = sum(len(p) for p in pred_cut)

        return dict(
            op=s,
            pred_cut=pred_cut
        )


class CharConfusionMatrix:
    def tp(self, true, pred, **kwargs):
        return dict(
            tp=sum(len(set(t) & set(p)) for t, p in zip(true, pred))
        )

    def fp(self, true, pred, **kwargs):
        return dict(
            fp=sum(len(set(t) - set(p)) for t, p in zip(true, pred))
        )

    def cp(self, true, **kwargs):
        return dict(
            cp=sum(len(set(t)) for t in true)
        )

    def op(self, pred, **kwargs):
        return dict(
            op=sum(len(set(p)) for p in pred)
        )


class PR(classifier.PR):
    def __init__(self, confusion_method=None, **confusion_method_kwarg):
        super().__init__(confusion_method=confusion_method or LineConfusionMatrix, **confusion_method_kwarg)


class TopMetric(classifier.TopMetric):
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
