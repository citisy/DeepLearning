import numpy as np
from . import classifier


class FullConfusionMatrix:
    def __init__(self, n_class, ignore_class=None):
        self.n_class = n_class
        self.ignore_class = np.array(ignore_class).reshape((-1,))

    def tp(self, true, pred, **kwargs):
        """true positive"""
        if self.ignore_class.size:
            true, pred = true.copy(), pred.copy()
            n_class = self.n_class + 1
            true[np.any(true == self.ignore_class[:, None], axis=0)] = self.n_class
            pred[np.any(pred == self.ignore_class[:, None], axis=0)] = self.n_class
            s = n_class * true + pred
            tp = np.bincount(s, minlength=n_class ** 2).reshape(n_class, n_class)
            tp = tp[:self.n_class, :self.n_class]
        else:
            n_class = self.n_class
            s = n_class * true + pred
            tp = np.bincount(s, minlength=n_class ** 2).reshape(n_class, n_class)
        return dict(
            tp=tp,
            acc_tp=np.sum(np.diag(tp))
        )

    def cp(self, true, **kwargs):
        """condition positive"""
        if self.ignore_class.size:
            true = true.copy()
            true[np.any(true == self.ignore_class[:, None], axis=0)] = self.n_class
            cp = np.bincount(true, minlength=self.n_class + 1)
            cp = cp[:self.n_class]
        else:
            cp = np.bincount(true, minlength=self.n_class)
        return dict(
            cp=cp,
            acc_cp=sum(cp)
        )

    def op(self, pred, **kwargs):
        """outcome positive"""
        if self.ignore_class.size:
            pred = pred.copy()
            pred[np.any(pred == self.ignore_class[:, None], axis=0)] = self.n_class
            op = np.bincount(pred, minlength=self.n_class + 1)
            op = op[:self.n_class]
        else:
            op = np.bincount(pred, minlength=self.n_class)
        return dict(
            op=op,
            acc_op=sum(op)
        )


class SingleConfusionMatrix:
    def __init__(self, pos: int or tuple):
        self.pos = np.array(pos).reshape((-1,))

    def tp(self, true, pred, **kwargs):
        """true positive"""
        a = true == self.pos[:, None]
        b = pred == self.pos[:, None]
        tp = np.any(a & b, axis=0)

        return dict(
            tp=tp,
            acc_tp=np.sum(tp)
        )

    def cp(self, true, **kwargs):
        """condition positive"""
        cp = np.any(true == self.pos[:, None], axis=0)
        return dict(
            cp=cp,
            acc_cp=np.sum(cp)
        )

    def op(self, pred, **kwargs):
        """outcome positive"""
        op = np.any(pred == self.pos[:, None], axis=0)
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
