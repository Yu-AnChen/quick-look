import functools

import cv2
import numpy as np
import skimage.measure
import skimage.util
import tifffile
from joblib import Parallel, delayed


def shannon_entropy(img, block_size):
    assert img.ndim == 2
    wimg = skimage.util.view_as_windows(img, block_size, block_size)
    h, w, *_ = wimg.shape
    out = np.zeros((h, w))
    out.flat = Parallel(n_jobs=4)(
        delayed(skimage.measure.shannon_entropy)(wimg[i, j])
        for i, j in np.mgrid[:h, :w].reshape(2, -1).T
    )
    return out


def var_of_laplacian(img, block_size, sigma=0):
    assert img.ndim == 2
    wimg = skimage.util.view_as_windows(img, block_size, block_size)
    h, w, *_ = wimg.shape
    out = np.zeros((h, w))

    func = lambda x: x
    if sigma != 0:
        func = functools.partial(cv2.GaussianBlur, ksize=(0, 0), sigmaX=sigma)
    for i, j in np.mgrid[:h, :w].reshape(2, -1).T:
        out[i, j] = np.var(
            cv2.Laplacian(func(wimg[i, j]), cv2.CV_32F, ksize=1)
        )
    return out


def var_block(img, block_size):
    assert img.ndim == 2
    wimg = skimage.util.view_as_windows(img, block_size, block_size)
    return np.var(wimg, axis=(2, 3))


def mean_block(img, block_size):
    assert img.ndim == 2
    wimg = skimage.util.view_as_windows(img, block_size, block_size)
    return np.mean(wimg, axis=(2, 3))


import sklearn.mixture


def gmm_cutoffs(var_img, plot=False):
    gmm = sklearn.mixture.GaussianMixture(n_components=3)
    limg = np.sort(np.log1p(var_img.flat))
    labels = gmm.fit_predict(limg.reshape(-1, 1))
    diff_idxs = np.where(np.diff(labels))
    diffs = np.mean(
        (limg[diff_idxs[0]], limg[diff_idxs[0]+1]),
        axis=0
    )
    filtered_diffs = diffs[
        (diffs > gmm.means_.min()) & (diffs < gmm.means_.max())
    ]
    if plot:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots()
        h, *_ = ax.hist(limg, bins=200)
        ax.vlines(filtered_diffs, 0, h.max(), colors='salmon')
    return np.expm1(filtered_diffs)


import logging
import pathlib

import skimage.filters

logging.basicConfig( 
    format="%(asctime)s | %(levelname)-8s | %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S", 
    level=logging.INFO 
)


def process_file(img_path, plot=False, out_dir=None):
    img_path = pathlib.Path(img_path)
    
    logging.info(f"Reading {img_path}")
    img = tifffile.imread(img_path, key=0)

    logging.info("Computing entropy")
    entropy_img = shannon_entropy(img, 128)
    tissue_mask = entropy_img > skimage.filters.threshold_triangle(entropy_img)

    logging.info("Computing variance")
    lvar_img = var_of_laplacian(img, 128, 1)
    # qc_img = lvar_img / (mean_block(img, 128)+1)
    # qc_img = np.nan_to_num(qc_img)
    # qc_mask = qc_img > skimage.filters.threshold_triangle(qc_img)
    # qc_mask = lvar_img > skimage.filters.threshold_triangle(lvar_img)
    qc_mask = lvar_img >= gmm_cutoffs(lvar_img)[1]
    iou = (tissue_mask & qc_mask).sum() / (tissue_mask | qc_mask).sum()

    text = f"""

    Tissue area fraction {100*tissue_mask.sum() / tissue_mask.size:.1f}%
    Good quality area fraction {100*qc_mask.sum() / qc_mask.size:.1f}%
    IoU {iou*100:.1f}%

    """
    logging.info(text)
    logging.info('Done')

    if plot:
        if out_dir is None:
            out_dir = img_path.stem
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        fig = plot_tissue_quality(
            mean_block(img, 32),
            entropy_img,
            tissue_mask,
            lvar_img,
            qc_mask,
            thumbnail_extent_factor=32/128
        )
        fig.suptitle(img_path.name, y=.90, va='top')
        fig.set_size_inches(fig.get_size_inches()*2)
        fig.savefig(out_dir / f"{img_path.stem}-qc.png", dpi=144, bbox_inches='tight')
