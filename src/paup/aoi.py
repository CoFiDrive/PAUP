"""Causal per-link delivery queues. All timestamps are simulation milliseconds."""
import heapq
import itertools
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReceivedMessage:
    generation_ms: float
    receive_ms: float
    message: object


class AoIManager:
    def __init__(self):
        self.pending = []
        self.last_valid = {}
        self._serial = itertools.count()
        self.now_ms = -math.inf

    def enqueue(self, sender, generation_ms, receive_ms, message, dropped=False):
        if not all(math.isfinite(x) for x in (generation_ms, receive_ms)) or receive_ms < generation_ms:
            raise ValueError('Invalid packet timestamps')
        if not dropped:
            heapq.heappush(self.pending, (receive_ms, next(self._serial), sender,
                                        ReceivedMessage(generation_ms, receive_ms, message)))

    def advance(self, now_ms):
        if not math.isfinite(now_ms) or now_ms < self.now_ms:
            raise ValueError('AoI clock must be finite and nondecreasing; reset between episodes')
        self.now_ms = now_ms
        while self.pending and self.pending[0][0] <= now_ms:
            _, _, sender, packet = heapq.heappop(self.pending)
            old = self.last_valid.get(sender)
            if old is None or packet.generation_ms > old.generation_ms:
                self.last_valid[sender] = packet

    def age_ms(self, sender):
        packet = self.last_valid.get(sender)
        return math.inf if packet is None else self.now_ms - packet.generation_ms

    def forget(self, sender):
        self.last_valid.pop(sender, None)
        self.pending = [entry for entry in self.pending if entry[2] != sender]
        heapq.heapify(self.pending)
