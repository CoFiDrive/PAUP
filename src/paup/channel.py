"""Aggregate-table channel, not a reconstruction of measured packet traces."""
import json
import math
import warnings
from pathlib import Path

import numpy as np


class MeasuredV2XChannel:
    def __init__(self, table_path=None, seed=20, bounds='error', mode='measured',
                 delay_ms=0., loss_rate=0., jitter_ms=0.):
        if bounds not in ('error', 'clip') or mode not in ('measured', 'fixed'):
            raise ValueError('Invalid channel bounds/mode')
        if not 0 <= loss_rate <= 1 or delay_ms < 0 or jitter_ms < 0:
            raise ValueError('Invalid delay, loss probability, or synthetic jitter')
        self.mode, self.bounds = mode, bounds
        self.delay_ms, self.loss_rate, self.jitter_ms = delay_ms, loss_rate, jitter_ms
        self.rng = np.random.RandomState(seed)
        self.clipped_queries = 0
        if mode == 'fixed':
            return
        if table_path is None:
            raise ValueError('Measured mode requires an explicit external table_path')
        with open(table_path, encoding='utf-8') as stream:
            self.table = json.load(stream)
        primary = self.table['primary']
        self.distances = np.asarray(primary['distance_m'], dtype=float)
        self.payloads = np.asarray(primary['payload_bytes'], dtype=float)
        self.latency = np.asarray(primary['latency_ms'], dtype=float)
        self.loss = np.asarray(primary['loss_probability'], dtype=float)
        if (np.any(np.diff(self.distances) <= 0) or np.any(np.diff(self.payloads) <= 0)
                or self.latency.shape != (len(self.payloads), len(self.distances))
                or self.loss.shape != self.latency.shape or not np.isfinite(self.latency).all()
                or not np.isfinite(self.loss).all() or np.any(self.latency < 0)
                or np.any((self.loss < 0) | (self.loss > 1))):
            raise ValueError('Malformed channel table')

    def parameters(self, distance_m, payload_bytes, send_rate_hz):
        if not all(math.isfinite(v) for v in (distance_m, payload_bytes, send_rate_hz)):
            raise ValueError('Channel inputs must be finite')
        if distance_m < 0 or payload_bytes <= 0 or send_rate_hz <= 0:
            raise ValueError('Invalid distance, payload or send rate')
        if self.mode == 'fixed':
            return float(self.delay_ms), float(self.loss_rate)
        outside = (not self.distances[0] <= distance_m <= self.distances[-1]
                   or not self.payloads[0] <= payload_bytes <= self.payloads[-1])
        if outside:
            if self.bounds == 'error':
                raise ValueError('Query outside measured domain; explicitly select bounds=clip to clamp')
            if self.clipped_queries == 0:
                warnings.warn('Channel query clamped to measured domain; inspect clipped_queries')
            self.clipped_queries += 1
        # np.interp clamps at boundaries; two 1-D interpolations are bilinear.
        def interpolate(matrix):
            return float(np.interp(payload_bytes, self.payloads,
                                  [np.interp(distance_m, self.distances, row) for row in matrix]))
        # Primary workbook does NOT identify rate dependence. Rate controls the
        # sender schedule only. Supplementary directional records stay separate.
        return interpolate(self.latency), interpolate(self.loss)

    def sample(self, distance_m, payload_bytes, send_rate_hz):
        latency, loss = self.parameters(distance_m, payload_bytes, send_rate_hz)
        dropped = bool(self.rng.random_sample() < loss)
        if self.jitter_ms:
            latency = max(0., latency + self.rng.uniform(-self.jitter_ms, self.jitter_ms))
        return {'latency_ms': latency, 'dropped': dropped, 'loss_probability': loss}
