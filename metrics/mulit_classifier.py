import numpy as np
from . import classifier


class FullConfusionMatrix:
    def __init__(self, n_class):
        self.n_class = n_class

    def tp(self, true, pred, **kwargs):
        """true positive"""
        s = self.n_class * true + pred
        tp = np.bincount(s, minlength=self.n_class ** 2).reshape(self.n_class, self.n_class)
        return dict(
            tp=tp,
            acc_tp=np.sum(np.diag(tp))
        )

    def cp(self, true, **kwargs):
        """condition positive"""
        cp = np.bincount(true, minlength=self.n_class)
        return dict(
            cp=cp,
            acc_cp=len(true)
        )

    def op(self, pred, **kwargs):
        """outcome positive"""
        op = np.bincount(pred, minlength=self.n_class)
        return dict(
            op=op,
            acc_op=len(pred)
        )


class SingleConfusionMatrix:
    def __init__(self, pos: int or tuple):
        if not isinstance(pos, (list, tuple)):
            pos = [pos]
        self.pos = pos

    def tp(self, true, pred, **kwargs):
        """true positive"""
        tp = np.zeros_like(true, dtype=bool)
        for pos in self.pos:
            tp &= (true == pos) & (pred == pos)
        return dict(
            tp=tp,
            acc_tp=np.sum(tp)
        )

    def cp(self, true, **kwargs):
        """condition positive"""
        cp = np.zeros_like(true, dtype=bool)
        for pos in self.pos:
            cp &= true == pos
        return dict(
            cp=cp,
            acc_cp=np.sum(cp)
        )

    def op(self, pred, **kwargs):
        """outcome positive"""
        op = np.zeros_like(pred, dtype=bool)
        for pos in self.pos:
            op &= pred == pos
        return dict(
            op=op,
            acc_op=np.sum(op)
        )


class PR(classifier.PR):
    def __init__(self, return_more_info=False, confusion_method=None, **confusion_method_kwarg):
        super().__init__(return_more_info=return_more_info, confusion_method=confusion_method or FullConfusionMatrix, **confusion_method_kwarg)


class TopMetric(classifier.TopMetric):
    def __init__(self, pr_method=None, **pr_method_kwarg):
        super().__init__(pr_method=pr_method or PR, **pr_method_kwarg)
