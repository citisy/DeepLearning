import numpy as np
from scipy import linalg
import torch
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm


def get_activations(images, model, device, batch_size=16, dims=2048):
    model.eval()
    pred_arr = np.empty((len(images), dims))
    start_idx = 0
    for i in tqdm(range(0, len(images), batch_size), desc='Count fid activations'):
        batch = images[i: i + batch_size]
        batch = [transforms.ToTensor()(img) for img in batch]
        batch = torch.stack(batch).to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
    return pred_arr


def get_default_cls_model(device='cuda', dims=2048):
    from pytorch_fid import fid_score  # pip install pytorch-fid==0.3.0
    block_idx = fid_score.InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    cls_model = fid_score.InceptionV3([block_idx]).to(device)
    return cls_model


def fid(images1, images2, cls_model=None, device='cuda', dims=2048):
    """Fréchet Inception Distance, refer to
    paper:
        - https://browse.arxiv.org/pdf/1706.08500.pdf
    code:
        - https://github.com/bioinf-jku/TTUR/blob/master/FIDvsINC/fid.py
        - https://github.com/mseitzer/pytorch-fid

    Args:
        images1 (np.ndarray):
            all images must have the same shape
        images2 (np.ndarray):
            all images must have the same shape
        cls_model:
            an image classifier model, if not set, use a default InceptionV3 model from `pytorch_fid.fid_score`
        device:
        dims:

    """
    if cls_model is None:
        cls_model = get_default_cls_model(device=device, dims=dims)

    act1 = get_activations(images1, cls_model, device, dims=dims)
    act2 = get_activations(images2, cls_model, device, dims=dims)

    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # calculate sum squared difference between means
    diff = np.sum((mu1 - mu2) * 2.0)

    # calculate sqrt of product between cov
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0])
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # calculate score
    return diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
