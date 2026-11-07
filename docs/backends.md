# External backend integration

The PAUP package does not copy separately licensed perception or dataset code.
Install a compatible backend under `external_paths/` or as a Python package.
Use its actual configuration, model and dataset; no legacy project-specific
architecture name is hard-coded into PAUP's model factory.

## Perception training

The supplied perception adapter uses independently installed `opencood`
interfaces: YAML loading, `build_dataset`, `create_model` and `create_loss`.
Dataset objects expose `collate_batch_train`; their batches contain the native
`ego` input and label dictionaries. Predictor training follows that backend's
own architecture, pretrained initialization and loss configuration.

Set package roots and an external architecture YAML in
`configs/perception.yaml`. The adapter requires that backend's GPU operators
and dataset dependencies. Its CPU availability does not imply that sparse
convolution extensions or full dataset training are installed.

## Dynamic PAUP planner fine-tuning

`paup.backends.live` owns the causal rollout and the final planner training loss.
Its frontend factory returns a frozen `torch.nn.Module` with three methods:

### `encode(observation)`

Return an encoded dictionary containing:

| Key | Meaning |
| --- | --- |
| `predicted` |N×C×H×W causal DPP probabilities |
| `anchors` |N×C×H×W received-anchor probabilities before prediction |
| `aoi_ms` |N generation-time ages; index0 is the current ego with age0 |
| `history_available` |N received-history validity flags |
| `horizon_valid` |N forecast-horizon validity flags |
| `request_sampling_affines` |N×2×3 grid_sample transforms: supporter output grid → ego input grid |
| backend feature fields | Frozen per-scale BEV features required by `fuse` |

The encoder must use only delivered anchors and their available earlier
history. It must not access a current remote sensor frame that has not arrived.
Use `paup.causal.causal_forecast` to implement the received-history prediction
path. It retains anchor features and returns predictive probabilities/logits
under an explicit input-domain flag. Class-wise maximum is performed by the
PAUP communication layer. Fresh local ego features stay selected.

### `fuse(encoded, mask, frame)`

Apply the supplied exact communication mask to frozen BEV features, pool it
for coarser scales, warp selected features and run the backbone's learned
multi-scale fusion/decode. Return one observed history record, including any
current perception-derived occupancy and measured pose needed below.
`paup.models.communication.scaled_agent_attention` provides the standard
parameter-free agent-axis attention operator if the backend needs it.

### `align_history(records)`

Return exactly `features`, `occupancy`, `target` tensors for the current
planner call, with a batch dimension of1. Pad early histories with the first
observation until five states exist. Warp previous features/maps into the
current ego coordinate frame. Target is the observed navigation target, not
a future GT waypoint. The map tensor contains the five observed planner
channels and must not contain future GT occupancy.

The dataset factory receives `(backend_config, split='training'|'validation')`.
The frontend factory receives `backend_config` and loads the user's private
perception/DPP checkpoints. These factories are intentionally external: a
generic repository cannot infer your dataset reader, sparse backend build,
coordinate conventions or checkpoint locations.

State resets at each training window. Intermediate planner drafts and frozen
perception computations run under no_grad; the final planner pass has normal
training gradients. Frozen frontend parameters remain outside the optimizer.
Current support / gate / budget defaults are configured in the live YAML.

## Other backends

A custom `module:create_task(config)` may return
`paup.training.TrainingTask(model, train_loader, validation_loader, loss)`.
The common runner handles optimizer state, cosine scheduling, gradient checks,
validation loss and checkpoint resume. This interface also supports a native
predictor-pretraining task without vendoring its implementation.
