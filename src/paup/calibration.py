"""Matched-residual calibration; no fitted scales or local datasets are distributed."""

import argparse, csv, json

from pathlib import Path

import numpy as np

import torch

from torch.nn import functional as F

from scipy import ndimage

from paup.causal import CALIBRATION_PROTOCOL

def match_centers(probabilities, gt_xy, threshold=.1, max_distance_px=20.):
    """Greedy one-to-one nearest matching of NMS heatmap peaks, per class.

    Unmatched objects are counted separately: this calibrates localization of
    matched detections, not recall or a probabilistic occupancy guarantee.
    """
    heat = torch.as_tensor(probabilities).float()[None, None]
    if not torch.isfinite(heat).all() or torch.any((heat < 0) | (heat > 1)):
        raise ValueError('Peak extraction requires finite probabilities in [0,1]')
    maximum = F.max_pool2d(heat, 3, stride=1, padding=1)
    minimum = -F.max_pool2d(-heat, 3, stride=1, padding=1)
    # A flat image contains no localized detections, even above the threshold.
    peaks = ((heat == maximum) & (heat > minimum + 1e-7) & (heat >= threshold))[0,0].cpu().numpy()
    labels, _ = ndimage.label(peaks, structure=np.ones((3,3)))
    predicted = []
    # Collapse a tied local maximum into one deterministic representative.
    for label, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None:
            continue
        yx = np.argwhere(labels[region] == label)
        yx += np.array([region[0].start, region[1].start])
        representative = yx[np.argmin(((yx-yx.mean(axis=0))**2).sum(axis=1))]
        predicted.append(representative[::-1])
    predicted = np.asarray(predicted).reshape(-1,2)
    gt_xy = np.asarray(gt_xy, dtype=float).reshape(-1, 2)
    candidates = []
    for p, center in enumerate(predicted):
        for g, target in enumerate(gt_xy):
            distance = float(np.linalg.norm(center-target))
            if distance <= max_distance_px:
                candidates.append((distance, p, g))
    used_pred, used_gt, residuals = set(), set(), []
    for _, p, g in sorted(candidates):
        if p not in used_pred and g not in used_gt:
            used_pred.add(p)
            used_gt.add(g)
            residuals.append((predicted[p]-gt_xy[g]).tolist())
    return residuals, len(gt_xy)-len(used_gt), len(predicted)-len(used_pred)

def fit(rows, edges, min_count, heatmap_hw):
    array = np.asarray([[float(r[k]) for k in ('aoi_ms', 'dx_px', 'dy_px')] for r in rows])
    if not len(array) or not np.isfinite(array).all() or np.any(array[:, 0] < 0):
        raise ValueError('Residuals must be finite, nonempty and have nonnegative ages')
    if min_count < 2 or len(edges) < 3 or np.any(np.diff(edges) <= 0):
        raise ValueError('Need >=2 bins, increasing edges and min_count >=2')
    ages, sigmas, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = array[(array[:, 0] >= lo) & (array[:, 0] < hi)]
        if len(selected) < min_count:
            raise ValueError('AoI bin [%s, %s) has %d residuals; need %d' % (lo, hi, len(selected), min_count))
        ages.append(float(selected[:, 0].mean()))
        # RMS about zero includes bias; not an empirical covariance claim.
        sigmas.append(float(np.sqrt(np.mean(selected[:, 1:] ** 2))))
        counts.append(len(selected))
    return {'source': 'simulation_dpp_gt_residuals', 'units': 'heatmap_pixels',
            'heatmap_hw': list(heatmap_hw), 'aoi_ms': ages, 'sigma_px': sigmas,
            'bin_edges_ms': list(edges), 'counts': counts,
            'estimator': 'sqrt(mean((dx^2+dy^2)/2)); bias included; matched detections only',
            'outside_age_domain': 'clamp; never extrapolate calibrated precision'}


def main(argv=None):
    import hashlib
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--residuals',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--edges',type=float,nargs='+',default=[0,100,300,500,700,901])
    p.add_argument('--min-count',type=int,default=30)
    p.add_argument('--heatmap-hw',type=int,nargs=2,default=[96,288])
    p.add_argument('--sensor-frame-ms',type=float,required=True)
    p.add_argument('--split',required=True)
    a=p.parse_args(argv)
    if a.sensor_frame_ms<=0:raise ValueError('Native sensor interval must be positive')
    rows=[]
    for path in a.residuals:
        with path.open(newline='') as f:rows.extend(csv.DictReader(f))
    result=fit(rows,a.edges,a.min_count,a.heatmap_hw)
    result.update(protocol=CALIBRATION_PROTOCOL,calibration_split=a.split,sensor_frame_ms=a.sensor_frame_ms,
                  matching_scope='Successfully matched targets only',
                  residual_files=[dict(name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in a.residuals])
    if a.output.exists():raise FileExistsError('Choose a new calibration output file')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
