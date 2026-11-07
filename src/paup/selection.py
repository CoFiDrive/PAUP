"""Opt-in request geometry and bounded PAUP increments; no model parameters."""
import math

import numpy as np
import torch


DEFAULTS = dict(request_max_age_ms=600., waypoint_interval_ms=200.,
                age_tau_ms=800., drift_weight=2., foreground_threshold=.1,
                extra_fraction=.25, extra_min_cells=4, extra_max_cells=32)


def aligned_request(points, old, current, age_ms, config):
    """Planner convention: world q=(gps_y,-gps_x), local p=R(theta).T(q-qego).

    Only actual past/current poses are used. Plan points retain their original
    future timestamps; elapsed points are removed, never synthesized again.
    """
    info=dict(age_ms=float(age_ms),aligned=False,reason=None)
    if not math.isfinite(age_ms) or age_ms < -1e-4:
        raise ValueError('Invalid planning request age')
    if age_ms>config['request_max_age_ms']:
        return None,dict(info,reason='expired')
    if points.ndim!=3 or points.shape[0]!=1 or points.shape[-1]!=2:
        raise ValueError('Request alignment requires one ego and B,T,2 waypoints')
    poses=np.asarray([[m['gps_y'],-m['gps_x'],m['theta']] for m in (old,current)],dtype=float)
    if not np.isfinite(poses).all() or not torch.isfinite(points).all():
        raise ValueError('Non-finite request geometry')
    def rotation(angle):
        c,s=math.cos(angle),math.sin(angle)
        return points.new_tensor([[c,-s],[s,c]])
    world=points @ rotation(poses[0,2]).T + points.new_tensor(poses[0,:2]-poses[1,:2])
    projected=world @ rotation(poses[1,2])
    times=torch.arange(1,points.shape[1]+1,device=points.device)*config['waypoint_interval_ms']
    keep=times>age_ms+1e-3
    if not bool(keep.any()):
        return None,dict(info,reason='no_future_points')
    result=projected[:,keep]
    return result,dict(info,aligned=True,reason=None,
        removed_points=int((~keep).sum()),remaining_points=int(keep.sum()),
        future_offsets_ms=(times[keep]-age_ms).cpu().tolist(),
        mean_reprojection_m=float((projected-points).norm(dim=-1).mean()))


def prediction_quality(logits, metadata, counts, predicted_logits, cap, config):
    """A heuristic change proxy, NOT a calibrated reliability probability."""
    rows=[];offset=0;selected=0
    for batch,count in enumerate(counts.tolist()):
        meta=metadata[offset:offset+count]
        anchors=torch.nonzero(meta[:,0]==1,as_tuple=False).flatten().tolist()
        for physical,anchor in enumerate(anchors):
            end=anchors[physical+1] if physical+1<len(anchors) else count
            age=float(meta[anchor,1]);has_history=end-anchor==2
            gap=float(meta[anchor+1,1])-age if has_history else None
            before=logits[offset+anchor].detach().sigmoid().max(dim=0)[0]
            after=predicted_logits[selected].detach().sigmoid().max(dim=0)[0]
            foreground=torch.maximum(before,after)>=config['foreground_threshold']
            denominator=torch.maximum(before,after)[foreground].sum()
            drift=float((before-after).abs()[foreground].sum()/(denominator+1e-6))
            reason=None
            if physical==0:reason='ego'
            elif age>1e-5 and not has_history:reason='no_history'
            elif has_history and (gap<=0 or age/gap>cap):reason='forecast_limit'
            elif not bool(foreground.any()):reason='no_foreground_support'
            gate=(math.exp(-age/config['age_tau_ms'])/(1+config['drift_weight']*drift)
                  if reason is None else (1. if physical==0 else 0.))
            rows.append(dict(batch_index=batch,physical_index=physical,aoi_ms=age,
                has_history=has_history,history_gap_ms=gap,prediction_change_proxy=drift,
                gate=gate,gate_reason=reason,calibrated_probability=False))
            selected+=1
        offset+=count
    if selected!=len(predicted_logits):raise ValueError('Quality/feature indexing mismatch')
    return rows


def bounded_masks(base, candidate, qualities, threshold, config, override=None):
    """Preserve base cells; admit only bounded, deterministically ranked extras."""
    if base.shape!=candidate.shape or len(base)!=len(qualities):
        raise ValueError('Mask and reliability rows differ')
    if not torch.isfinite(base).all() or not torch.isfinite(candidate).all():
        raise ValueError('Non-finite communication scores')
    baseline=base>=threshold;baseline[0]=True
    raw=candidate>=threshold;raw[0]=True
    result=baseline.clone();rows=[]
    for i,quality in enumerate(qualities):
        q=dict(quality);gate=q['gate'] if override is None else float(override)
        if not 0<=gate<=1:raise ValueError('Gate must be in [0,1]')
        delta=(candidate[i]-base[i]).clamp_min(0)
        eligible=((base[i]+gate*delta)>=threshold)&~baseline[i]&raw[i]
        base_count=int(baseline[i].sum())
        budget=min(config['extra_max_cells'],max(config['extra_min_cells'],
                   int(math.floor(config['extra_fraction']*base_count))))
        if i==0:budget=0;eligible[:]=False
        ids=torch.nonzero(eligible.flatten(),as_tuple=False).flatten()
        if len(ids):
            scores=(gate*delta).flatten()[ids]
            # The deployed torch build predates argsort(stable=True). Keep
            # tied scores deterministic using the flattened cell index.
            order=torch.as_tensor(np.lexsort((ids.cpu().numpy(),-scores.detach().cpu().numpy())),
                                  device=ids.device)
            admitted=ids[order[:budget]]
            result[i].view(-1)[admitted]=True
        rows.append(dict(q,gate=gate,base_selected_cells=base_count,
            candidate_added_cells=int((raw[i]&~baseline[i]).sum()),
            eligible_added_cells=len(ids),extra_budget_cells=budget,
            admitted_added_cells=int((result[i]&~baseline[i]).sum())))
    return result,baseline,raw,rows
