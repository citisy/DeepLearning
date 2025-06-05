from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi
from einops.layers.torch import Rearrange

from torch import nn

from utils import torch_utils
from ..layers import Conv


class WeightConverter:
    @classmethod
    def from_official(cls, state_dict):
        convert_dict = {
            'head.conv1': 'head.to_in.conv',
            'head.bn1': 'head.to_in.norm',
            'head.conv2': 'head.to_out.conv',
            'head.bn2': 'head.to_out.norm',

            'head.{0}.conv1': 'head.{0}.layer1.conv',
            'head.{0}.bn1': 'head.{0}.layer1.norm',
            'head.{0}.conv2': 'head.{0}.layer2.conv',
            'head.{0}.bn2': 'head.{0}.layer2.norm',

            'xvector.tdnn.linear': 'xvector.tdnn.conv',
            'xvector.tdnn.nonlinear.batchnorm': 'xvector.tdnn.norm',

            'xvector.{0}.nonlinear1.batchnorm': 'xvector.{0}.0.norm',
            'xvector.{0}.linear1': 'xvector.{0}.0.conv',
            'xvector.{0}.nonlinear2.batchnorm': 'xvector.{0}.1.norm',
            'xvector.{0}.cam_layer.linear_local': 'xvector.{0}.1.conv.linear_local',
            'xvector.{0}.cam_layer.linear1': 'xvector.{0}.1.conv.layer1.conv',
            'xvector.{0}.cam_layer.linear2': 'xvector.{0}.1.conv.layer2.conv',

            'xvector.transit{0}.nonlinear.batchnorm': 'xvector.transit{0}.norm',
            'xvector.transit{0}.linear': 'xvector.transit{0}.conv',

            'xvector.out_nonlinear.batchnorm': 'xvector.out_nonlinear.0',

            'xvector.dense.linear': 'xvector.dense.1.conv',
            'xvector.dense.nonlinear.batchnorm': 'xvector.dense.1.norm'
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


class Model(nn.Module):
    def __init__(
            self,
            feat_dim=80,
            embedding_size=192,
            growth_rate=32,
            bn_size=4,
            in_ch=128,
            output_level="segment",
            **kwargs,
    ):
        super().__init__()
        self.head = FCM(feat_dim=feat_dim)

        layers = OrderedDict()
        k = 5
        dilation = 1
        layers['tdnn'] = Conv(
            self.head.out_channels, in_ch, k, p=(k - 1) // 2 * dilation, dilation=dilation,
            bias=False, mode='cna',
            conv_fn=nn.Conv1d, norm_fn=nn.BatchNorm1d, act=nn.ReLU(),
        )

        ch = in_ch
        for i, (num_layers, kernel_size, dilation) in enumerate(zip(
                (12, 24, 16),
                (3, 3, 3),
                (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_ch=ch,
                out_ch=growth_rate,
                hidden_ch=bn_size * growth_rate,
                k=kernel_size,
                dilation=dilation,
            )
            layers["block%d" % (i + 1)] = block
            ch = ch + num_layers * growth_rate
            layers["transit%d" % (i + 1)] = Conv(
                ch, ch // 2, 1, bias=False, mode='nac',
                conv_fn=nn.Conv1d, norm_fn=nn.BatchNorm1d, act=nn.ReLU(),
            )
            ch //= 2

        layers["out_nonlinear"] = nn.Sequential(
            nn.BatchNorm1d(ch),
            nn.ReLU(),
        )

        if output_level == "segment":
            layers['stats'] = StatsPool()
            layers['dense'] = nn.Sequential(
                Rearrange('b c -> b c 1'),
                Conv(
                    ch * 2, embedding_size, 1, bias=False, mode='cn',
                    conv_fn=nn.Conv1d, norm=nn.BatchNorm1d(embedding_size, affine=False)
                ),
                Rearrange('b c 1 -> b c'),
            )
        elif output_level == "frame":
            layers['view_out'] = Rearrange('b w h -> b h w')
        self.xvector = nn.Sequential(layers)
        self.cb_model = ClusterBackend()

    def forward(self, x, is_final=True, segments=None, **kwargs):
        x, _, _ = self.extract_feature(x)

        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)

        outputs = dict(
            hidden=x,
        )
        if is_final:
            preds = self.cb_model(x)
            outputs['preds'] = preds
            if segments is not None:
                outputs['segment_preds'] = self.extract_segments(segments, preds)
        return outputs

    @staticmethod
    def extract_feature(audio):
        features = []
        feature_times = []
        feature_lengths = []
        for au in audio:
            feature = Kaldi.fbank(au.unsqueeze(0), num_mel_bins=80)
            feature = feature - feature.mean(dim=0, keepdim=True)
            features.append(feature)
            feature_times.append(au.shape[0])
            feature_lengths.append(feature.shape[0])

        features = torch.stack(features)
        return features, feature_lengths, feature_times

    def extract_segments(self, segments: list, labels: np.ndarray) -> list:
        labels = self.correct_labels(labels)
        distribute_res = []
        for i in range(len(segments)):
            distribute_res.append([segments[i][0], segments[i][1], labels[i]])
        # merge the same speakers chronologically
        distribute_res = self.merge_seque(distribute_res)

        def is_overlapped(t1, t2):
            if t1 > t2 + 1e-4:
                return True
            return False

        # distribute the overlap region
        for i in range(1, len(distribute_res)):
            if is_overlapped(distribute_res[i - 1][1], distribute_res[i][0]):
                p = (distribute_res[i][0] + distribute_res[i - 1][1]) / 2
                distribute_res[i][0] = p
                distribute_res[i - 1][1] = p

        # smooth the result
        distribute_res = self.smooth(distribute_res)

        return distribute_res

    @staticmethod
    def correct_labels(labels):
        labels_id = 0
        id2id = {}
        new_labels = []
        for i in labels:
            if i not in id2id:
                id2id[i] = labels_id
                labels_id += 1
            new_labels.append(id2id[i])
        return np.array(new_labels)

    @staticmethod
    def merge_seque(distribute_res):
        res = [distribute_res[0]]
        for i in range(1, len(distribute_res)):
            if distribute_res[i][2] != res[-1][2] or distribute_res[i][0] > res[-1][1]:
                res.append(distribute_res[i])
            else:
                res[-1][1] = distribute_res[i][1]
        return res

    def smooth(self, res, mindur=0.7):
        # if only one segment, return directly
        if len(res) < 2:
            return res
        # short segments are assigned to nearest speakers.
        for i in range(len(res)):
            res[i][0] = round(res[i][0], 2)
            res[i][1] = round(res[i][1], 2)
            if res[i][1] - res[i][0] < mindur:
                if i == 0:
                    res[i][2] = res[i + 1][2]
                elif i == len(res) - 1:
                    res[i][2] = res[i - 1][2]
                elif res[i][0] - res[i - 1][1] <= res[i + 1][0] - res[i][1]:
                    res[i][2] = res[i - 1][2]
                else:
                    res[i][2] = res[i + 1][2]
        # merge the speakers
        res = self.merge_seque(res)

        return res


class FCM(nn.Module):
    def __init__(self, num_blocks=(2, 2), hidden_ch=32, feat_dim=80):
        super().__init__()
        self.hidden_ch = hidden_ch

        self.to_in = Conv(1, hidden_ch, 3, bias=False, mode='cna', norm_fn=nn.BatchNorm2d, act=nn.ReLU())

        self.layer1 = self._make_layer(BasicResBlock, hidden_ch, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(BasicResBlock, hidden_ch, num_blocks[1], stride=2)

        self.to_out = Conv(hidden_ch, hidden_ch, 3, s=(2, 1), p=1, bias=False, mode='cna', norm_fn=nn.BatchNorm2d, act=nn.ReLU())
        self.out_channels = hidden_ch * (feat_dim // 8)

    def _make_layer(self, block, hidden_ch, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.hidden_ch, hidden_ch, stride))
            self.hidden_ch = hidden_ch * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.to_in(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.to_out(out)

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class BasicResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.layer1 = Conv(in_ch, out_ch, 3, s=(stride, 1), p=1, bias=False, mode='cna', norm_fn=nn.BatchNorm2d, act=nn.ReLU())
        self.layer2 = Conv(out_ch, out_ch, 3, bias=False, mode='cn', norm_fn=nn.BatchNorm2d)

        if stride != 1 or in_ch != self.expansion * out_ch:
            self.shortcut = Conv(in_ch, self.expansion * out_ch, 1, s=(stride, 1), p=0, bias=False, mode='cn', norm_fn=nn.BatchNorm2d, detail_name=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CAMDenseTDNNBlock(nn.ModuleList):
    def __init__(self, num_layers, in_ch, out_ch, hidden_ch, k, s=1, dilation=1, bias=False):
        super().__init__()
        for i in range(num_layers):
            p = (k - 1) // 2 * dilation
            layer = nn.Sequential(
                Conv(in_ch + i * out_ch, hidden_ch, 1, bias=False, mode='nac', conv_fn=nn.Conv1d, norm_fn=nn.BatchNorm1d, act=nn.ReLU()),
                Conv(hidden_ch, out_ch, k, s=s, p=p, dilation=dilation, bias=bias, mode='nac', conv_fn=CAMLayer, norm_fn=nn.BatchNorm1d)
            )
            self.add_module("tdnnd%d" % (i + 1), layer)

    def forward(self, x):
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class CAMLayer(nn.Module):
    def __init__(
            self, in_ch, out_ch, k, s, p, dilation, bias, reduction=2
    ):
        super().__init__()
        self.linear_local = nn.Conv1d(in_ch, out_ch, k, stride=s, padding=p, dilation=dilation, bias=bias)
        self.layer1 = Conv(in_ch, in_ch // reduction, 1, mode='ca', conv_fn=nn.Conv1d, act=nn.ReLU())
        self.layer2 = Conv(in_ch // reduction, out_ch, 1, mode='ca', conv_fn=nn.Conv1d, act=nn.Sigmoid())

    def forward(self, x):
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.layer1(context)
        m = self.layer2(context)
        return y * m

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., : x.shape[-1]]
        return seg


class StatsPool(nn.Module):
    dim = -1
    keepdim = False
    unbiased = True

    def forward(self, x):
        mean = x.mean(dim=self.dim)
        std = x.std(dim=self.dim, unbiased=self.unbiased)
        stats = torch.cat([mean, std], dim=-1)
        if self.keepdim:
            stats = stats.unsqueeze(dim=self.dim)
        return stats


class ClusterBackend:
    """Perfom clustering for input embeddings and output the labels."""

    def __init__(self, merge_thr=0.78):
        self.merge_thr = merge_thr

        self.spectral_cluster = SpectralCluster()
        self.umap_hdbscan_cluster = UmapHdbscan()

    def __call__(self, X, **params):
        # clustering and return the labels
        k = params["oracle_num"] if "oracle_num" in params else None
        assert len(X.shape) == 2, "modelscope error: the shape of input should be [N, C]"
        if X.shape[0] < 20:
            return np.zeros(X.shape[0], dtype="int")
        if X.shape[0] < 2048 or k is not None:
            # unexpected corner case
            labels = self.spectral_cluster(X, k)
        else:
            labels = self.umap_hdbscan_cluster(X)

        labels = self.merge_by_cos(labels, X, self.merge_thr)
        return labels

    def merge_by_cos(self, labels, embs, cos_thr):
        # merge the similar speakers by cosine similarity
        assert cos_thr > 0 and cos_thr <= 1
        while True:
            spk_num = labels.max() + 1
            if spk_num == 1:
                break
            spk_center = []
            for i in range(spk_num):
                spk_emb = embs[labels == i].mean(0)
                spk_center.append(spk_emb)
            assert len(spk_center) > 0
            spk_center = np.stack(spk_center, axis=0)
            norm_spk_center = spk_center / np.linalg.norm(spk_center, axis=1, keepdims=True)
            affinity = np.matmul(norm_spk_center, norm_spk_center.T)
            affinity = np.triu(affinity, 1)
            spks = np.unravel_index(np.argmax(affinity), affinity.shape)
            if affinity[spks] < cos_thr:
                break
            for i in range(len(labels)):
                if labels[i] == spks[1]:
                    labels[i] = spks[0]
                elif labels[i] > spks[1]:
                    labels[i] -= 1
        return labels


class SpectralCluster:
    r"""A spectral clustering mehtod using unnormalized Laplacian of affinity matrix.
    This implementation is adapted from https://github.com/speechbrain/speechbrain.
    """

    def __init__(self, min_num_spks=1, max_num_spks=15, pval=0.022):
        self.min_num_spks = min_num_spks
        self.max_num_spks = max_num_spks
        self.pval = pval

    def __call__(self, X, oracle_num=None):
        # Similarity matrix computation
        sim_mat = self.get_sim_mat(X)

        # Refining similarity matrix with pval
        prunned_sim_mat = self.p_pruning(sim_mat)

        # Symmetrization
        sym_prund_sim_mat = 0.5 * (prunned_sim_mat + prunned_sim_mat.T)

        # Laplacian calculation
        laplacian = self.get_laplacian(sym_prund_sim_mat)

        # Get Spectral Embeddings
        emb, num_of_spk = self.get_spec_embs(laplacian, oracle_num)

        # Perform clustering
        labels = self.cluster_embs(emb, num_of_spk)

        return labels

    def get_sim_mat(self, X):
        import sklearn
        # Cosine similarities
        M = sklearn.metrics.pairwise.cosine_similarity(X, X)
        return M

    def p_pruning(self, A):
        if A.shape[0] * self.pval < 6:
            pval = 6.0 / A.shape[0]
        else:
            pval = self.pval

        n_elems = int((1 - pval) * A.shape[0])

        # For each row in a affinity matrix
        for i in range(A.shape[0]):
            low_indexes = np.argsort(A[i, :])
            low_indexes = low_indexes[0:n_elems]

            # Replace smaller similarity values by 0s
            A[i, low_indexes] = 0
        return A

    def get_laplacian(self, M):
        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    def get_spec_embs(self, L, k_oracle=None):
        lambdas, eig_vecs = scipy.linalg.eigh(L)

        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(
                lambdas[self.min_num_spks - 1: self.max_num_spks + 1]
            )
            num_of_spk = np.argmax(lambda_gap_list) + self.min_num_spks

        emb = eig_vecs[:, :num_of_spk]
        return emb, num_of_spk

    def cluster_embs(self, emb, k):
        from sklearn.cluster._kmeans import k_means
        _, labels, _ = k_means(emb, k)
        return labels

    def getEigenGaps(self, eig_vals):
        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            eig_vals_gap_list.append(gap)
        return eig_vals_gap_list


class UmapHdbscan:
    r"""
    Reference:
    - Siqi Zheng, Hongbin Suo. Reformulating Speaker Diarization as Community Detection With
      Emphasis On Topological Structure. ICASSP2022
    """

    def __init__(
            self, n_neighbors=20, n_components=60, min_samples=10, min_cluster_size=10, metric="cosine"
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.metric = metric

    def __call__(self, X):
        from sklearn.cluster import HDBSCAN
        import umap.umap_ as umap

        umap_X = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=0.0,
            n_components=min(self.n_components, X.shape[0] - 2),
            metric=self.metric,
        ).fit_transform(X)
        labels = HDBSCAN(
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size,
            allow_single_cluster=True,
        ).fit_predict(umap_X)
        return labels
