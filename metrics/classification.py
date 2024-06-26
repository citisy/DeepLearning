import numpy as np


class ConfusionMatrix:
    def __init__(self, pos=1):
        self.pos = pos

    def tp(self, true, pred, **kwargs):
        """true positive"""
        tp = (true == self.pos) & (pred == self.pos)
        return dict(
            tp=tp,
            acc_tp=np.sum(tp)
        )

    def fp(self, true, pred, **kwargs):
        """false positive"""
        fp = (true != self.pos) & (pred == self.pos)
        return dict(
            fp=fp,
            acc_fp=np.sum(fp)
        )

    def fn(self, true, pred, **kwargs):
        """false negative"""
        fn = (true == self.pos) & (pred != self.pos)
        return dict(
            fn=fn,
            acc_fn=np.sum(fn)
        )

    def tn(self, true, pred, **kwargs):
        """true negative"""
        tn = (true != self.pos) & (pred != self.pos)
        return dict(
            tn=tn,
            acc_tn=np.sum(tn)
        )

    def cp(self, true, **kwargs):
        """condition positive"""
        cp = true == self.pos
        return dict(
            cp=cp,
            acc_cp=np.sum(cp)
        )

    def cn(self, true, **kwargs):
        """condition negative"""
        cn = true != self.pos
        return dict(
            cn=cn,
            acc_cn=np.sum(cn)
        )

    def op(self, pred, **kwargs):
        """outcome positive"""
        op = pred == self.pos
        return dict(
            op=op,
            acc_op=np.sum(op)
        )

    def on(self, pred, **kwargs):
        """outcome negative"""
        on = pred != self.pos
        return dict(
            on=on,
            acc_on=np.sum(on)
        )


class PR:
    def __init__(self, return_more_info=False, confusion_method=None, **confusion_method_kwarg):
        self.return_more_info = return_more_info
        self.confusion_matrix = confusion_method(**confusion_method_kwarg) if confusion_method is not None else ConfusionMatrix(**confusion_method_kwarg)

        # alias functions
        self.recall = self.tpr
        self.precision = self.ppv
        self.fallout = self.fpr

    def get_pr(self, true=None, pred=None):
        r = {}
        r.update(self.confusion_matrix.tp(true, pred))
        r.update(self.confusion_matrix.cp(true, **r))
        r.update(self.confusion_matrix.op(pred, **r))

        r.update(self.tpr(**r))
        r.update(self.ppv(**r))
        return r

    def tpr(self, true=None, pred=None, acc_tp=None, acc_cp=None, **kwargs):
        """recall"""
        r = {}
        if acc_tp is None:
            r.update(self.confusion_matrix.tp(true, pred, **r))
            acc_tp = r.pop('acc_tp')

        if acc_cp is None:
            r.update(self.confusion_matrix.cp(true, **r))
            acc_cp = r.pop('acc_cp')

        ret = dict(
            tpr=acc_tp / (acc_cp + 1e-6),
            acc_tp=acc_tp,
            acc_cp=acc_cp
        )

        if self.return_more_info:
            ret.update(r)

        return ret

    def ppv(self, true=None, pred=None, acc_tp=None, acc_op=None, **kwargs):
        """precision"""
        r = {}
        if acc_tp is None:
            r.update(self.confusion_matrix.tp(true, pred, **r))
            acc_tp = r.pop('acc_tp')

        if acc_op is None:
            r.update(self.confusion_matrix.op(pred, **r))
            acc_op = r.pop('acc_op')

        ret = dict(
            ppv=acc_tp / (acc_op + 1e-6),
            acc_tp=acc_tp,
            acc_op=acc_op
        )

        if self.return_more_info:
            ret.update(r)

        return ret

    def fpr(self, true=None, pred=None, acc_fp=None, acc_cn=None, **kwargs):
        """fallout"""
        r = {}
        if acc_fp is None:
            r.update(self.confusion_matrix.fp(true, pred, **r))
            acc_fp = r.pop('acc_fp')

        if acc_cn is None:
            r.update(self.confusion_matrix.cn(true, **r))
            acc_cn = r.pop('acc_cn')

        ret = dict(
            fpr=acc_fp / acc_cn,
            acc_fp=acc_fp,
            acc_cn=acc_cn
        )

        if self.return_more_info:
            ret.update(r)

        return ret

    def acc(self, true=None, pred=None, acc_tp=None, acc_tn=None, **kwargs):
        """accuracy, only apply for 2-class problem"""
        r = {}
        if acc_tp is None:
            r.update(self.confusion_matrix.tp(true, pred, **r))
            acc_tp = r.pop('acc_tp')

        if acc_tn is None:
            r.update(self.confusion_matrix.tn(true, pred, **r))
            acc_tn = r.pop('acc_tn')

        ret = dict(
            acc=(acc_tp + acc_tn) / len(true),
            acc_tp=acc_tp,
            acc_tn=acc_tn
        )

        if self.return_more_info:
            ret.update(r)

        return ret

    def mcc(self, true, pred, **kwargs):
        """Matthews correlation coefficient"""
        ret = {}
        ret.update(self.confusion_matrix.tp(true, pred))
        ret.update(self.confusion_matrix.tn(true, pred))
        ret.update(self.confusion_matrix.fp(true, pred))
        ret.update(self.confusion_matrix.fn(true, pred))

        # note, convert to int64 to avoid overflow
        acc_tp = ret.pop('acc_tp').astype(np.int64)
        acc_tn = ret.pop('acc_tn').astype(np.int64)
        acc_fp = ret.pop('acc_fp').astype(np.int64)
        acc_fn = ret.pop('acc_fn').astype(np.int64)

        result = dict(
            mcc=(acc_tp * acc_tn - acc_fp * acc_fn) / np.sqrt((acc_tp + acc_fp) * (acc_tp + acc_fn) * (acc_tn + acc_fp) * (acc_tn + acc_fn) + 1e-6),
            acc_tp=acc_tp,
            acc_tn=acc_tn,
            acc_fp=acc_fp,
            acc_fn=acc_fn
        )

        if self.return_more_info:
            result.update(ret)

        return result


class TopMetric:
    def __init__(self, return_more_info=False, pr_method=None, **pr_method_kwarg):
        self.return_more_info = return_more_info
        self.pr = pr_method(**pr_method_kwarg) if pr_method is not None else PR(**pr_method_kwarg)
        self.pr.return_more_info = return_more_info

    def f_measure(self, true=None, pred=None, a=1, eps=1e-6, **kwargs):
        ret = self.pr.get_pr(true, pred)
        p = ret.pop('ppv')
        r = ret.pop('tpr')

        result = dict(
            p=p,
            r=r,
            f=(a ** 2 + 1) * p * r / (a ** 2 * p + r + eps),
            acc_tp=ret.pop('acc_tp'),
            acc_cp=ret.pop('acc_cp'),
            acc_op=ret.pop('acc_op')
        )

        if self.return_more_info:
            result.update(ret)

        return result

    def f1(self, true, pred, **kwargs):
        return self.f_measure(true, pred, a=1, **kwargs)

    def pr_curve(self, true, pred, thres_list=np.arange(0, 1, 0.05)):
        pr = {}
        for thres in thres_list:
            true_ = np.where(true <= thres, 0, 1)
            pred_ = np.where(pred <= thres, 0, 1)

            ret = self.f_measure(true_, pred_)
            pr[thres] = ret

        p = np.array([_['p'] for _ in pr.values()])
        r = np.array([_['r'] for _ in pr.values()])
        points = p / r
        best_arg = np.argmin(np.abs(points - 1))
        best_thres = thres_list[best_arg]

        return dict(
            pr=pr,
            points=points,
            best_arg=best_arg,
            best_thres=best_thres
        )

    def roc(self, true, pred, thres_list=np.arange(0, 1, 0.05)):
        pr = {}
        for thres in thres_list:
            true_ = np.where(true <= thres, 0, 1)
            pred_ = np.where(pred <= thres, 0, 1)

            ret = self.pr.tpr(true_, pred_)
            ret.update(self.pr.fpr(true_, pred_))
            pr[thres] = ret

        tpr = np.array([_['tpr'] for _ in pr.values()])
        fpr = np.array([_['fpr'] for _ in pr.values()])
        points = tpr / fpr
        best_arg = np.argmin(points)
        best_thres = thres_list[best_arg]

        return dict(
            pr=pr,
            points=points,
            best_arg=best_arg,
            best_thres=best_thres
        )

    def auc(self, true, pred):
        pass

    def ks(self, true, pred, thres_list=np.arange(0, 1, 0.05)):
        pr = {}
        for thres in thres_list:
            true_ = np.where(true <= thres, 0, 1)
            pred_ = np.where(pred <= thres, 0, 1)

            ret = self.pr.tpr(true_, pred_)
            ret.update(self.pr.fpr(true_, pred_))
            pr[thres] = ret

        tpr = np.array([_['tpr'] for _ in pr.values()])
        fpr = np.array([_['fpr'] for _ in pr.values()])
        points = np.abs(tpr - fpr)
        best_arg = np.argmin(points)
        best_thres = thres_list[best_arg]

        return dict(
            pr=pr,
            points=points,
            best_arg=best_arg,
            best_thres=best_thres
        )


confusion_matrix = ConfusionMatrix()
pr = PR()
top_metric = TopMetric()
