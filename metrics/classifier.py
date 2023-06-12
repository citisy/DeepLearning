import numpy as np


class ConfusionMatrix:
    def __init__(self, pos=1):
        self.pos = pos

    def tp(self, true, pred, **kwargs):
        """true positive"""
        return dict(
            tp=np.sum((true == self.pos) & (pred == self.pos))
        )

    def fp(self, true, pred, **kwargs):
        """false positive"""
        return dict(
            fp=np.sum((true != self.pos) & (pred == self.pos))
        )

    def fn(self, true, pred, **kwargs):
        """false negative"""
        return dict(
            fn=np.sum((true == self.pos) & (pred != self.pos))
        )

    def tn(self, true, pred, **kwargs):
        """true negative"""
        return dict(
            tn=np.sum((true != self.pos) & (pred != self.pos))
        )

    def cp(self, true, **kwargs):
        """condition positive"""
        return dict(
            cp=np.sum(true == self.pos)
        )

    def cn(self, true, **kwargs):
        """condition negative"""
        return dict(
            cn=np.sum(true != self.pos)
        )

    def op(self, pred, **kwargs):
        """outcome positive"""
        return dict(
            op=np.sum(pred == self.pos)
        )

    def on(self, pred, **kwargs):
        """outcome negative"""
        return dict(
            on=np.sum(pred != self.pos)
        )


class PR:
    def __init__(self, confusion_method=None, **confusion_method_kwarg):
        self.confusion_matrix = confusion_method(**confusion_method_kwarg) if confusion_method is not None else ConfusionMatrix(**confusion_method_kwarg)
        # alias
        self.recall = self.tpr
        self.precision = self.ppv
        self.fallout = self.fpr

    def tpr(self, true=None, pred=None, tp=None, cp=None, **kwargs):
        """recall"""
        r = {}
        if tp is None:
            r = self.confusion_matrix.tp(true, pred)
            tp = r.pop('tp')

        if cp is None:
            r = self.confusion_matrix.cp(true, **r)
            cp = r.pop('cp')

        return dict(
            tpr=tp / cp,
            tp=tp,
            cp=cp
        )

    def ppv(self, true=None, pred=None, tp=None, op=None, **kwargs):
        """precision"""
        r = {}
        if tp is None:
            r = self.confusion_matrix.tp(true, pred)
            tp = r.pop('tp')

        if op is None:
            r = self.confusion_matrix.op(pred, **r)
            op = r.pop('op')

        return dict(
            ppv=tp / op,
            tp=tp,
            op=op
        )

    def fpr(self, true=None, pred=None, fp=None, cn=None, **kwargs):
        """fallout"""
        r = {}
        if fp is None:
            r = self.confusion_matrix.fp(true, pred)
            fp = r.pop('fp')

        if cn is None:
            r = self.confusion_matrix.cn(true, **r)
            cn = r.pop('cn')

        return dict(
            fpr=fp / cn,
            fp=fp,
            cn=cn
        )

    def acc(self, true=None, pred=None, tp=None, tn=None, **kwargs):
        r = {}
        if tp is None:
            r = self.confusion_matrix.tp(true, pred)
            tp = r.pop('tp')

        if tn is None:
            r = self.confusion_matrix.tn(true, **r)
            tn = r.pop('tn')

        return dict(
            acc=(tp + tn) / len(true),
            tp=tp,
            tn=tn
        )


class TopMetric:
    def __init__(self, return_more_info=False, pr_method=None, **pr_method_kwarg):
        self.return_more_info = return_more_info
        self.pr = pr_method(**pr_method_kwarg) if pr_method is not None else PR(**pr_method_kwarg)

    def f_measure(self, true=None, pred=None, a=1, eps=1e-6, p=None, r=None, **kwargs):
        ret = {}
        if p is None:
            ret = self.pr.precision(true, pred)
            p = ret.pop('ppv')

        if r is None:
            ret.update(self.pr.recall(true, pred, **ret))
            r = ret.pop('tpr')

        result = dict(
            p=p,
            r=r,
            f=(a ** 2 + 1) * p * r / (a ** 2 * p + r + eps)
        )

        if self.return_more_info:
            result.update(ret)

        return result

    def f1(self, true, pred, **kwargs):
        return self.f_measure(true, pred, a=1, **kwargs)

    def pr_curve(self, true, pred, thres_list=np.arange(0, 1, 0.05)):
        pr = {}
        for thres in thres_list:
            true_ = np.where(true < thres, 0, 1)
            pred_ = np.where(pred < thres, 0, 1)

            ret = self.f_measure(true_, pred_)
            pr[thres] = ret

        p = np.array([_['p'] for _ in pr.values()])
        r = np.array([_['r'] for _ in pr.values()])
        best_point = np.argmin(np.abs(p / r - 1))

        return dict(
            pr=pr,
            best_point=best_point
        )

    def roc(self, true, pred, thres_list=np.arange(0, 1, 0.05)):
        pr = {}
        for thres in thres_list:
            true_ = np.where(true < thres, 0, 1)
            pred_ = np.where(pred < thres, 0, 1)

            ret = self.f_measure(true_, pred_)
            pr[thres] = ret

        pr_ = np.array([[_['p'], _['r']] for _ in pr.values()])
        base = np.ones_like(pr_)

        cos_sim = pr_ * base / np.linalg.norm(pr_, axis=1) * np.sqrt(2)
        best_point = np.argmin(cos_sim)

        return dict(
            pr=pr,
            best_point=best_point
        )

    def auc(self, true, pred):
        pass

    def ks(self, true, pred, thres_list=np.arange(0, 1, 0.05)):
        pr = {}
        for thres in thres_list:
            true_ = np.where(true < thres, 0, 1)
            pred_ = np.where(pred < thres, 0, 1)

            ret = self.f_measure(true_, pred_)
            pr[thres] = ret

        pr_ = np.array([[_['p'], _['r']] for _ in pr.values()])
        pr_[1] = 1 - pr_[1]
        base = np.ones_like(pr_)

        cos_sim = pr_ * base / np.linalg.norm(pr_, axis=1) * np.sqrt(2)
        best_point = np.argmin(cos_sim)

        return dict(
            pr=pr,
            best_point=best_point
        )


confusion_matrix = ConfusionMatrix()
pr = PR()
top_metric = TopMetric()
