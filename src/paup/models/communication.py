"""Parameter-free PAUP scores and bounded admission for encoded supporter inputs."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from paup.uncertainty import PlanningAwareAgeUncertainty
from paup.selection import DEFAULTS, bounded_masks


class PAUPCommunication(nn.Module):
    def __init__(self, uncertainty, threshold=.05, parameters=None, smooth_size=5, smooth_sigma=1.):
        super().__init__()
        self.uncertainty = PlanningAwareAgeUncertainty(uncertainty)
        self.threshold = float(threshold)
        self.parameters_config = dict(DEFAULTS, **(parameters or {}))
        if not 0 < threshold < 1 or smooth_size % 2 != 1 or smooth_sigma <= 0:
            raise ValueError('Invalid selection threshold or smoothing kernel')
        coords = torch.arange(-(smooth_size//2), smooth_size//2+1, dtype=torch.float64)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        # Match the fixed score-smoothing convention used by the perception backend.
        kernel = torch.exp(-(x*x+y*y)/(2*smooth_sigma**2))/(2*math.pi*smooth_sigma)
        self.register_buffer('score_kernel', kernel.float()[None,None])

    def forward(self, predicted, anchors, aoi_ms, request, history_available, horizon_valid):
        if predicted.shape != anchors.shape or predicted.ndim != 4:
            raise ValueError('Expected matching N,C,H,W probability tensors')
        if not torch.isfinite(predicted).all() or not torch.isfinite(anchors).all():
            raise ValueError('Non-finite probability inputs')
        if ((predicted < 0) | (predicted > 1)).any() or ((anchors < 0) | (anchors > 1)).any():
            raise ValueError('Predictor inputs must be probabilities, not logits')
        n, _, h, w = predicted.shape
        if len(aoi_ms) != n or len(history_available) != n or len(horizon_valid) != n:
            raise ValueError('One age/history/horizon flag is required per physical agent')
        if float(aoi_ms[0]) != 0:
            raise ValueError('Agent zero must be the current local ego observation')
        if request is not None and request.shape != (n,1,h,w):
            raise ValueError('Request shape must be N,1,H,W')
        confidence = predicted.max(1, keepdim=True).values
        previous = anchors.max(1, keepdim=True).values
        expanded = self.uncertainty(confidence, aoi_ms, request)
        padding = self.score_kernel.shape[-1]//2
        kernel = self.score_kernel.to(confidence)
        base = F.conv2d(confidence, kernel, padding=padding)
        candidate = F.conv2d(expanded, kernel, padding=padding)
        if request is not None:
            base, candidate = base*request, candidate*request
        q=[]; cfg=self.parameters_config
        for j in range(n):
            fg = torch.maximum(previous[j],confidence[j]) >= cfg['foreground_threshold']
            denom = torch.maximum(previous[j],confidence[j])[fg].sum()
            drift = float((previous[j]-confidence[j]).abs()[fg].sum()/(denom+1e-6))
            valid = bool(history_available[j]) and bool(horizon_valid[j]) and bool(fg.any()) and request is not None
            gate = math.exp(-float(aoi_ms[j])/cfg['age_tau_ms'])/(1+cfg['drift_weight']*drift) if valid else 0.
            if j==0: gate=1.
            q.append(dict(physical_index=j,aoi_ms=float(aoi_ms[j]),gate=gate,prediction_change_proxy=drift))
        selected, baseline, raw, rows = bounded_masks(base,candidate,q,self.threshold,cfg)
        return dict(mask=selected,baseline=baseline,candidate=raw,confidence=confidence,
                    expanded_confidence=expanded,base_score=base,candidate_score=candidate,quality=rows)


def scaled_agent_attention(features):
    """N,C,H,W -> C,H,W; per-pixel agent attention, ego query at index zero."""
    if features.ndim != 4 or len(features)==0:
        raise ValueError('Expected one or more agent feature maps')
    n,c,h,w=features.shape
    x=features.reshape(n,c,h*w).permute(2,0,1)
    attention=torch.softmax(torch.bmm(x,x.transpose(1,2))/math.sqrt(c),dim=-1)
    return torch.bmm(attention,x)[:,0].T.reshape(c,h,w)
