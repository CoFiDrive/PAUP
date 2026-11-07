# PAUP

**Planning-Aware Uncertainty Propagation**.

PAUP uses information age and an aligned prior plan to extend communication
selection around predictive perception responses. A consistency gate and a
bounded additional-cell budget regulate the increment while retaining the
baseline selection. The PAUP operator itself adds no learned parameters.

## Included

- Causal message queues, age tracking and received-history prediction.
- Matched-residual calibration and AoI-dependent separable Gaussian expansion.
- Planning-request geometry, consistency gating and deterministic bounded masks.
- Per-pixel agent attention and a five-observation spatiotemporal waypoint planner.
- Perception-training adapter, fixed-feature planner training, and causal
  window-local planner fine-tuning interfaces.
- Optimizer / scheduler / checkpoint handling and CPU tests.

This repository contains source code and configuration templates.

## Install

Use Python3.8+ and install the appropriate PyTorch build for your CUDA version.

```bash
git clone https://github.com/CoFiDrive/PAUP.git
cd PAUP
python -m pip install -e '.[dev]'
pytest -q
```

The CPU tests use temporary synthetic fixtures to verify source behavior; they
are not driving-performance experiments.

## Training

All paths are user-supplied, relative to the working directory. Store private
data, backend installations and initial checkpoints under `external_paths/`.
Training outputs are ignored by Git.

### 1. Perception

Install a compatible perception backend separately and configure its package
roots, architecture YAML and dataset splits in `configs/perception.yaml`.
See [backend integration](docs/backends.md).

```bash
python -m paup.train --config configs/perception.yaml --check-config
python -m paup.train --config configs/perception.yaml --device cuda
```

The template uses Adam,160epochs, learning rate0.002 and weight decay0.0001.
Its data reader, detection loss and predictor architecture come from the
external backend; they are not replaced by an unrelated network in this release.

### 2. Planner

Prepare pose-aligned five-observation feature windows according to the
[data contract](docs/data.md). Features and occupancy maps must come from
observations / frozen perception, with future waypoints reserved for the loss.

```bash
python -m paup.train --config configs/planner.yaml --device cuda
```

The template uses128-channel features,50epochs, AdamW, learning rate0.0002,
weight decay0.0005 and batch size4. The model predicts10waypoints.

### 3. Fine-tune

For fine-tuning on fixed perception caches:

```bash
python -m paup.train --config configs/finetune.yaml \
  --initialize external_paths/checkpoints/planner.ckpt --device cuda
```

For **dynamic PAUP feedback**, implement the small frozen-frontend contract in
[docs/backends.md](docs/backends.md), then set its factories and the calibration
path in `configs/live_finetune.yaml`:

```bash
python -m paup.train --config configs/live_finetune.yaml \
  --initialize external_paths/checkpoints/planner.ckpt --device cuda
```

The live trainer resets state per window, generates detached requests from
earlier predictions, refreshes selection from current observed inputs and
backpropagates only through the final planner pass. The perception frontend
stays frozen and in evaluation mode. This path currently requires batch size1.
The10-epoch template uses AdamW with learning rate0.00002. Cached fine-tuning
supports batch size2 but does not refresh masks when the plan changes; use the
live backend when studying that feedback mechanism.

Outputs contain `last.ckpt`, `best.ckpt` and a training log. Initialization is
separate from resuming optimizer state:

```bash
python -m paup.train --config configs/planner.yaml \
  --resume runs/planner/last.ckpt --device cuda
```

Only load checkpoints from a trusted source. Training checkpoints contain
optimizer and random-generator state as well as model parameters.

## Residual calibration

Collect class-matched predictor/GT residuals on a reserved calibration split.
CSV columns are `aoi_ms,dx_px,dy_px`; optional identity columns are permitted.
Then fit a scale before using the planning-aware mode:

```bash
python -m paup.calibration \
  --residuals external_paths/calibration/residuals.csv \
  --split calibration --sensor-frame-ms 200 \
  --output external_paths/calibration/aoi_sigma.json
```

The sensor-frame option records the native sampling interval of the data.

The fitted scale is the component RMS
`sqrt(mean((dx²+dy²)/2))`, including bias and conditional on successful matching.
It is not a guarantee of recovering unmatched targets. The mapping must use
the same predictor, resolution and sampling protocol as training.

## Algorithm defaults

`configs/paup.yaml` contains the communication threshold0.05, expansion weight1,
maximum sigma12heatmap pixels, prediction cap4steps, request expiry600ms,
age decay800ms, drift weight2, foreground threshold0.1, and additional-cell
budget `clip(floor(0.25*N_base),4,32)`. No fitted scale values are shipped.

## Scope and attribution

This source release does not bundle the external LiDAR encoder, DPP checkpoint,
simulator, data preprocessing package or benchmark results. The live-backend
template requires the user's licensed frontend adapter; it is an integration
contract, not a claim that raw-sensor training works without those dependencies.
See [NOTICE.md](NOTICE.md) for upstream project attribution and external terms.
The code distributed here is licensed under MIT.
