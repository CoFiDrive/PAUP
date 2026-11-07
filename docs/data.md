# Training data contract

## Frozen-feature windows

A manifest is an external JSON file:

```json
{
  "schema": "paup_feature_windows_v1",
  "occupancy_source": "observations",
  "pose_aligned": true,
  "windows": ["window_000001.npz", "window_000002.npz"]
}
```

Each NPZ contains float32 arrays, without Python pickle objects:

| Key | Shape | Role |
| --- | --- | --- |
| `features` |5×128×H×W | Frozen perception features aligned to the current ego pose |
| `occupancy` |5×5×H×W | Observed/perception-derived maps, not future GT occupancy |
| `target` |2 | Current navigation target |
| `future_waypoints` |10×2 | Supervision used only by the final loss |

The paper-compatible grid is192×96. The planner also accepts smaller spatial
grids for software tests. Use five native historical states and ten future
waypoints over2s for the method configuration. Preserve the underlying
timestamp and pose alignment when generating caches.

All window paths are relative to the manifest. Training and validation window
paths must be disjoint; route-disjoint splits should be enforced when preparing
the data. The reader checks shape, finiteness, observation provenance flags and
exact-path overlap, but it cannot infer a route split from a renamed file.

There is no released real dataset or fitted calibration in this repository.
Do not use synthetic test fixtures as experimental data.

## Causal live windows

The live dataset returns a dictionary with `window` and `future_waypoints`.
`window` contains1..5 chronologically ordered observations. Each frame has:

```text
time_ms: simulation/sensor time in milliseconds
pose: {gps_x, gps_y, theta}
observation: backend-native observed sensor data only
```

Pose convention is the planner's world frame `q=(gps_y,-gps_x)` with local
coordinates transformed by `R(theta)`. Requests retain their generation times
and poses; elapsed future points are removed, and expired requests are discarded.
The live loop does not pass the `future_waypoints` supervision to the frontend,
request generation or intermediate draft planner calls.
