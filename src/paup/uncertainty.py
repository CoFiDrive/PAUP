"""Parameter-free, peak-preserving PAUP in probability space."""
import json
import math
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from paup.causal import CALIBRATION_PROTOCOL


class PlanningAwareAgeUncertainty(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mode = config.get('mode', 'planning_aware')
        if self.mode not in ('none', 'fixed', 'age', 'planning_aware'):
            raise ValueError('Unknown PAUP ablation')
        self.weight = float(config.get('weight', 1.))
        self.fixed_sigma = float(config.get('fixed_sigma_px', 2.))
        self.max_sigma = float(config.get('max_sigma_px', 12.))
        self.calibration = None
        self._kernels = OrderedDict()
        if not 0 <= self.weight <= 1 or self.fixed_sigma < 0 or self.max_sigma <= 0:
            raise ValueError('Invalid PAUP parameters')
        if self.mode in ('age', 'planning_aware'):
            path = config.get('calibration_path')
            if not path:
                raise ValueError('PAUP requires DPP/GT residual calibration; see paup/README.md')
            with open(path, encoding='utf-8') as stream:
                self.calibration = json.load(stream)
            c = self.calibration
            if c.get('protocol') != CALIBRATION_PROTOCOL:
                raise ValueError('Calibration predates the probability-domain DPP/strict-peak fix; '
                                 'recollect with: python -m paup.run_local calibrate --limit 1000')
            if c.get('units') != 'heatmap_pixels' or c.get('source') != 'simulation_dpp_gt_residuals':
                raise ValueError('Calibration must use simulated DPP/GT residuals in heatmap pixels')
            self.ages = np.asarray(c['aoi_ms'], dtype=float)
            self.sigmas = np.asarray(c['sigma_px'], dtype=float)
            if (len(self.ages) < 2 or self.ages.shape != self.sigmas.shape
                    or not np.isfinite(self.ages).all() or not np.isfinite(self.sigmas).all()
                    or np.any(np.diff(self.ages) <= 0) or np.any(self.sigmas < 0)):
                raise ValueError('Invalid calibration knots')

    def forward(self, probabilities, aoi_ms, request=None):
        if self.mode == 'none':
            return probabilities
        if probabilities.ndim != 4 or len(aoi_ms) != probabilities.shape[0]:
            raise ValueError('PAUP requires NCHW probabilities and one AoI per agent')
        if self.calibration and list(probabilities.shape[-2:]) != self.calibration['heatmap_hw']:
            raise ValueError('Calibration heatmap resolution does not match model')
        if self.mode == 'planning_aware' and request is None:
            # First closed-loop frame has no previous planned waypoints.
            return probabilities
        results = []
        for i, age in enumerate(aoi_ms):
            age = float(age)
            if not math.isfinite(age) or age < 0:
                raise ValueError('AoI must be finite and nonnegative for received agents')
            h = probabilities[i:i+1]
            sigma = self.fixed_sigma if self.mode == 'fixed' else float(np.interp(age, self.ages, self.sigmas))
            sigma = min(sigma, self.max_sigma)
            if age == 0 or sigma < 1e-6:
                results.append(h)
                continue
            radius = int(math.ceil(3 * sigma))
            channels = h.shape[1]
            key = (sigma, channels, h.device, h.dtype)
            if key not in self._kernels:
                coords = torch.arange(-radius, radius + 1, device=h.device, dtype=h.dtype)
                kernel = torch.exp(-coords.square() / (2 * sigma * sigma))
                kernel = kernel / kernel.sum()
                self._kernels[key] = (kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1),
                                      kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1))
                if len(self._kernels) > 128:
                    self._kernels.popitem(last=False)
            self._kernels.move_to_end(key)
            horizontal, vertical = self._kernels[key]
            smooth = F.conv2d(h, horizontal,
                              padding=(0, radius), groups=channels)
            smooth = F.conv2d(smooth, vertical,
                              padding=(radius, 0), groups=channels)
            envelope = torch.maximum(h, smooth)
            risk = request[i:i+1].to(h).clamp(0, 1) if self.mode == 'planning_aware' else 1.
            results.append((h + self.weight * risk * (envelope - h)).clamp(0, 1))
        return torch.cat(results, dim=0)
