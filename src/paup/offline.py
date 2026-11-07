"""Index-independent replay for shuffled, multi-worker multi-agent datasets."""
import copy
import json
import math
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

from paup.transport import MessageTransport


class OfflineTimeline:
    def __init__(self, root, ego_route, sender_route, config, frame_ms):
        if frame_ms <= 0:
            raise ValueError('frame_ms must match the dataset sampling interval')
        self.frame_ms = frame_ms
        self.packets = []
        transport = MessageTransport(config)
        self.config_max_aoi_ms = transport.max_aoi_ms
        sender = str(sender_route).replace('\\', '/')
        sender_path = Path(root) / sender_route / 'measurements'
        ego_path = Path(root) / ego_route / 'measurements'
        # Some datasets replace rsu_1000 with rsu_2000 etc. The last three
        # digits identify a logical link. Keep received/in-flight observations
        # across replacement, but never form DPP history across two sensors.
        routes = [Path(sender_route)]
        if Path(sender_route).name.startswith('rsu_'):
            parent = Path(sender_route).parent
            suffix = Path(sender_route).name[-3:]
            routes = [parent / p.name for p in (Path(root) / parent).glob('rsu_*') if p.name[-3:] == suffix]
            sender = str(parent / ('rsu_link_' + suffix)).replace('\\', '/')
        files = []
        for route in routes:
            files.extend((int(p.stem), str(route), p) for p in (Path(root) / route / 'measurements').glob('*.json') if p.stem.isdigit())
        previous = None
        next_send = -math.inf
        channel = transport.channel(sender)
        # Read ONLY pose/timestamp metadata. Point clouds are loaded at query time
        # only for messages that have actually arrived.
        previous_route = None
        history_frames = []
        for frame, route, path in sorted(files):
            if route != previous_route:
                previous = None
                next_send = -math.inf
                history_frames = []
            previous_route = route
            now = frame * frame_ms
            ego_file = ego_path / path.name
            if not ego_file.exists():
                continue
            with path.open(encoding='utf-8') as stream:
                pose = json.load(stream)
            with ego_file.open(encoding='utf-8') as stream:
                ego = json.load(stream)
            if now + 1e-7 >= next_send:
                distance = math.hypot(pose['lidar_pose_x']-ego['lidar_pose_x'], pose['lidar_pose_y']-ego['lidar_pose_y'])
                outcome = channel.sample(distance, transport.payload_bytes, transport.rate)
                history_ms = config.get('history_frame_ms')
                if history_ms is not None:
                    while len(history_frames) > 1 and (frame-history_frames[1][1])*frame_ms >= history_ms:
                        history_frames.pop(0)
                    previous = (history_frames[0] if history_frames
                        and (frame-history_frames[0][1])*frame_ms >= history_ms else None)
                if not outcome['dropped']:
                    self.packets.append((now + outcome['latency_ms'], route, frame, previous))
                next_send = now + 1000. / transport.rate
            previous = (route, frame)
            history_frames.append(previous)
        # Arrival-sorted prefix maxima preserve out-of-order semantics without
        # scanning a whole recording for every shuffled planning-window query.
        self.arrivals, self.latest_prefix = [], []
        newest = None
        for packet in sorted(self.packets):
            if newest is None or packet[2] > newest[2]:
                newest = packet
            self.arrivals.append(packet[0])
            self.latest_prefix.append(newest)

    def query(self, frame):
        return [(generation_frame, meta) for _, generation_frame, meta in self.query_records(frame)]

    def query_records(self, frame):
        now = frame * self.frame_ms
        index = bisect_right(self.arrivals, now)-1
        if index < 0:
            return []
        _, route, latest, previous = self.latest_prefix[index]
        maximum = self.config_max_aoi_ms
        if maximum is not None and now-latest*self.frame_ms > maximum:
            return []
        return [(r, f, [float(i == 0), now - f * self.frame_ms])
                for i, (r, f) in enumerate([(route, latest)] + ([previous] if previous is not None else []))]


def retrieve_causal_data(dataset, scene, frame, tpe):
    config = dataset.params['paup']
    cache = getattr(dataset, '_paup_timelines', None)
    if cache is None:
        cache = dataset._paup_timelines = OrderedDict()
    visible = dataset.get_visible_actors(scene, frame)
    data = OrderedDict()
    ego = dataset.get_one_record(scene['ego'], frame, agent='ego', visible_actors=visible['car_0'], tpe=tpe)
    ego.update(paup_meta=[1., 0.], time_delay=0, route_dir=scene['ego'], scene_dict=scene, frame_id=frame)
    data['car_0'] = ego
    agents = []
    if dataset.max_cav > 1:
        agents.extend(('car_%d' % (i+1), route, 'other_ego') for i, route in enumerate(scene['other_egos']))
    if dataset.max_cav > 2:
        agents.extend(('rsu_%d' % i, route, 'rsu') for i, route in enumerate(scene['rsu']))
    agents = agents[:config.get('physical_agent_limit', dataset.max_cav)-1]
    for name, route, kind in agents:
        key = (scene['ego'], route)
        if key not in cache:
            cache[key] = OfflineTimeline(dataset.root_dir, scene['ego'], route, config['channel'],
                                         config.get('offline_frame_ms', dataset.params.get('frame_gap', 200)))
            while len(cache) > config.get('timeline_cache_size', 64):
                cache.popitem(last=False)
        cache.move_to_end(key)
        records = cache[key].query_records(frame)
        if not records:
            continue
        # Current labels are targets only. Current supporter sensor features never
        # enter data. Keep target time fixed even when a past sensor view is used.
        current = dataset.get_one_record(route, frame, agent=kind, visible_actors=visible[name], tpe=tpe)
        for i, (packet_route, generation_frame, meta) in enumerate(records):
            record = dataset.get_one_record(packet_route, generation_frame, agent=kind, visible_actors=visible[name], tpe=tpe)
            record['params']['vehicles'] = copy.deepcopy(current['params']['vehicles'])
            record.update(paup_meta=meta, time_delay=0, route_dir=packet_route)
            data[name if i == 0 else name + '_paup_history'] = record
    return data
