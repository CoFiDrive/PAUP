"""Message-level simulation. The configured payload is a PROXY, not encoded BEV bytes.

Each packet bundles its generating frame and at most one preceding sensor frame.
No raw supporter frame is exposed to inference before that packet arrives.
"""
import copy
import hashlib
import json
import math
from collections import deque
from pathlib import Path

from paup.aoi import AoIManager
from paup.channel import MeasuredV2XChannel


class MessageTransport:
    def __init__(self, config):
        self.config = config
        self.rate = float(config.get('send_rate_hz', 10))
        self.payload_bytes = int(config.get('payload_bytes', 3000))
        if self.rate <= 0 or self.payload_bytes <= 0:
            raise ValueError('send_rate_hz and payload_bytes must be positive')
        self.aoi = AoIManager()
        self.history_frame_ms = config.get('history_frame_ms')
        self.max_aoi_ms = config.get('max_aoi_ms')
        if self.history_frame_ms is not None and (not math.isfinite(self.history_frame_ms) or self.history_frame_ms <= 0):
            raise ValueError('history_frame_ms must be finite and positive')
        if self.max_aoi_ms is not None and (not math.isfinite(self.max_aoi_ms) or self.max_aoi_ms < 0):
            raise ValueError('max_aoi_ms must be finite and nonnegative')
        self.sensor_buffers = {}
        self.channels, self.history, self.next_send = {}, {}, {}
        self.stats = {'sent': 0, 'dropped': 0, 'proxy_payload_bytes': 0, 'expired_queries': 0}

    def channel(self, sender):
        if sender not in self.channels:
            seed = (int(self.config.get('seed', 20)) + int(hashlib.sha256(str(sender).encode()).hexdigest()[:8], 16)) % (2**32)
            args = {k: self.config[k] for k in ('table_path', 'bounds', 'mode', 'delay_ms', 'loss_rate', 'jitter_ms') if k in self.config}
            self.channels[sender] = MeasuredV2XChannel(seed=seed, **args)
        return self.channels[sender]

    def observe(self, sender, frame, now_ms, distance_m):
        if now_ms < self.aoi.now_ms:
            raise ValueError('Cannot generate packets in the past')
        old = self.history.get(sender)
        if old and now_ms <= old[0]:
            raise ValueError('Sender sensor timestamps must strictly increase')
        frame = copy.deepcopy(frame)
        if self.history_frame_ms is not None:
            buffer = self.sensor_buffers.setdefault(sender, deque())
            threshold = now_ms - self.history_frame_ms
            while len(buffer) > 1 and buffer[1][0] <= threshold + 1e-7:
                buffer.popleft()
            old = buffer[0] if buffer and buffer[0][0] <= threshold + 1e-7 else None
            buffer.append((now_ms, frame))
        if now_ms + 1e-7 >= self.next_send.get(sender, now_ms):
            result = self.channel(sender).sample(distance_m, self.payload_bytes, self.rate)
            # Metadata and the two historical sensor views constitute one virtual packet.
            bundle = [(now_ms, frame)] + ([old] if old else [])
            self.aoi.enqueue(sender, now_ms, now_ms + result['latency_ms'], bundle, result['dropped'])
            self.next_send[sender] = now_ms + 1000. / self.rate
            self.stats['sent'] += 1
            self.stats['dropped'] += int(result['dropped'])
            self.stats['proxy_payload_bytes'] += self.payload_bytes
        self.history[sender] = (now_ms, frame)

    def advance(self, now_ms):
        self.aoi.advance(now_ms)

    def records(self, sender):
        packet = self.aoi.last_valid.get(sender)
        if packet is None:
            return []
        if self.max_aoi_ms is not None and self.aoi.now_ms - packet.generation_ms > self.max_aoi_ms:
            self.stats['expired_queries'] += 1
            return []
        return [(copy.deepcopy(frame), [float(i == 0), self.aoi.now_ms - generation_ms])
                for i, (generation_ms, frame) in enumerate(packet.message)]

    def forget(self, sender):
        self.aoi.forget(sender)
        self.history.pop(sender, None)
        self.next_send.pop(sender, None)
        self.sensor_buffers.pop(sender, None)

    def reset_sensor_history(self, sender):
        self.history.pop(sender, None)
        self.next_send.pop(sender, None)
        self.sensor_buffers.pop(sender, None)


class OnlineTransport:
    def __init__(self, config, ego_vehicles_num=1):
        if ego_vehicles_num != 1:
            raise ValueError('PAUP online currently requires one controlled ego, as original APC does')
        self.transport = MessageTransport(config)
        self.identities = {}
        self.trace_path = config.get('trace_path')
        if self.trace_path:
            Path(self.trace_path).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def position(frame):
        m = frame['measurements']
        return float(m['lidar_pose_x']), float(m['lidar_pose_y'])

    def update(self, cars, rsus, timestamp_seconds):
        now_ms = float(timestamp_seconds) * 1000.
        if not cars or cars[0] is None:
            raise ValueError('Current ego sensors are required')
        ego = copy.deepcopy(cars[0])
        ego['paup_meta'] = [1., 0.]
        ego['time_delay'] = 0
        ex, ey = self.position(ego)
        groups = {'car': [ego], 'rsu': []}
        # Cars keep their slot identity; an RSU slot may be respawned at a new pose.
        for kind, frames in (('car', cars[1:]), ('rsu', rsus)):
            for index, frame in enumerate(frames):
                if frame is None:
                    continue  # Loss of an observation does not erase the last received message.
                x, y = self.position(frame)
                slot = (kind, index)
                identity = (kind, index, round(x, 3), round(y, 3)) if kind == 'rsu' else slot
                previous = self.identities.get(slot)
                if previous is not None and previous != identity:
                    # A respawned RSU must not use the former sensor's frame as
                    # its DPP history. Already received/in-flight packets remain
                    # valid observations in their OWN recorded coordinate frames.
                    self.transport.reset_sensor_history(slot)
                self.identities[slot] = identity
                self.transport.observe(slot, frame, now_ms, math.hypot(x-ex, y-ey))
        self.transport.advance(now_ms)
        for slot in sorted(self.identities):
            kind, _ = slot
            for frame, meta in self.transport.records(slot):
                frame['paup_meta'], frame['time_delay'] = meta, 0
                groups[kind].append(frame)
        if self.trace_path:
            snapshot = {'now_ms': now_ms, 'counts': self.transport.stats,
                        'clipped_queries': sum(c.clipped_queries for c in self.transport.channels.values()),
                        'links': [{'sender': str(sender), 'generation_ms': packet.generation_ms,
                                   'receive_ms': packet.receive_ms, 'aoi_ms': self.transport.aoi.age_ms(sender)}
                                  for sender, packet in self.transport.aoi.last_valid.items()]}
            with open(self.trace_path, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps(snapshot) + '\n')
        return groups['car'], groups['rsu']
