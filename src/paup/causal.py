"""Select received feature anchors and extrapolate only from their packet history."""
import math
import torch

CALIBRATION_PROTOCOL = 'probability_dpp_strict_peaks_v2'


def causal_forecast(x, logits, record_len, pairwise, metadata, predict, max_forecast_steps=4,
                    probability_prediction=False):
    if max_forecast_steps < 1:
        raise ValueError('max_forecast_steps must be positive')
    if metadata.shape != (len(logits), 2):
        raise ValueError('Expected [is_anchor, age_ms] for every flattened sensor record')
    if not torch.isfinite(metadata).all() or (metadata[:, 1] < 0).any():
        raise ValueError('Invalid causal timestamps')
    if not ((metadata[:, 0] == 0) | (metadata[:, 0] == 1)).all():
        raise ValueError('Anchor flags must be binary')
    chosen, predictions, ages, counts, matrices = [], [], [], [], []
    offset = 0
    for count in record_len.tolist():
        meta = metadata[offset:offset+count]
        anchors = torch.nonzero(meta[:, 0] == 1, as_tuple=False).flatten().tolist()
        if not anchors or anchors[0] != 0 or meta[0, 1] != 0:
            raise ValueError('First anchor must be the current ego')
        matrix = torch.zeros_like(pairwise[len(counts)])
        ids = torch.tensor(anchors, device=pairwise.device)
        matrix[:len(ids), :len(ids)] = pairwise[len(counts)].index_select(0, ids).index_select(1, ids)
        matrices.append(matrix)
        counts.append(len(anchors))
        for k, anchor in enumerate(anchors):
            end = anchors[k+1] if k+1 < len(anchors) else count
            if end - anchor not in (1, 2):
                raise ValueError('A packet must contain one anchor and at most one preceding sensor frame')
            idx = offset + anchor
            age = float(meta[anchor, 1])
            pred = logits[idx:idx+1]
            if k != 0 and age > 1e-5 and end-anchor == 2:
                gap = float(meta[anchor+1, 1]) - age
                if gap <= 0:
                    raise ValueError('DPP history must predate the delivered anchor')
                # Forecast at native sensor interval, then linearly interpolate the
                # final fractional step. Uses predictions, never future observations.
                effective_steps = min(age / gap, max_forecast_steps)
                steps = int(math.ceil(effective_steps))
                older = logits[idx+1:idx+2].unsqueeze(1)
                latest = pred.unsqueeze(1)
                # DMVFN clamps generated images to [0,1]. Feeding detector
                # logits and sigmoid'ing that output makes empty space 0.5.
                # PAUP predicts/interpolates probabilities, then returns logits
                # to preserve the existing fusion/communication interface.
                if probability_prediction:
                    older, latest = older.sigmoid(), latest.sigmoid()
                sequence = predict(older, latest, steps)
                if probability_prediction:
                    sequence = sequence.clamp(0, 1)
                initial = latest[:, 0]
                lower = initial if steps == 1 else sequence[:, steps-2]
                fraction = effective_steps - (steps - 1)
                pred = lower * (1-fraction) + sequence[:, steps-1] * fraction
                if probability_prediction:
                    pred = torch.logit(pred.clamp(1e-6, 1-1e-6))
            chosen.append(idx)
            predictions.append(pred)
            ages.append(age)
        offset += count
    if offset != len(logits):
        raise ValueError('record_len does not match flattened data')
    return (x[chosen], torch.cat(predictions), record_len.new_tensor(counts),
            torch.stack(matrices), metadata.new_tensor(ages))
