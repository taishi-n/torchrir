# Datasets

This page documents the `torchrir.datasets` helpers for CMU ARCTIC and
LibriSpeech, including accepted options, expected directory structures, and
error handling for invalid inputs.

## Scope

Covered APIs:

- `torchrir.datasets.CmuArcticDataset`
- `torchrir.datasets.LibriSpeechDataset`
- `torchrir.datasets.load_dataset_sources`
- `torchrir.datasets.collate_dataset_items`
- `torchrir.datasets.build_dynamic_cmu_arctic_dataset`

## Quick start (local data, no download)

```python
from pathlib import Path

from torchrir.datasets import CmuArcticDataset, LibriSpeechDataset

cmu = CmuArcticDataset(Path("datasets/cmu_arctic"), speaker="bdl", download=False)
libri = LibriSpeechDataset(
    Path("datasets/librispeech"),
    subset="train-clean-100",
    speaker="103",
    download=False,
)
```

When `download=False`, dataset loaders never start network download. They only
read local files and raise an error if required paths do not exist.

## CMU ARCTIC

### Accepted options

`CmuArcticDataset(root, speaker="bdl", download=False)`

| Option | Accepted values | Default | Notes |
|---|---|---|---|
| `root` | `Path` or path-like | required | Dataset root directory managed by the caller. |
| `speaker` | One of `aew`, `ahw`, `aup`, `awb`, `axb`, `bdl`, `clb`, `eey`, `fem`, `gka`, `jmk`, `ksp`, `ljm`, `lnh`, `rms`, `rxr`, `slp`, `slt` | `"bdl"` | Valid IDs are defined in `VALID_SPEAKERS`. |
| `download` | `True` or `False` | `False` | If `True`, download/extract missing archive for the selected speaker. |

You can also enumerate valid speakers with:

```python
from torchrir.datasets import cmu_arctic_speakers

print(cmu_arctic_speakers())
```

### Expected local structure

For `root = datasets/cmu_arctic` and `speaker = bdl`:

```text
datasets/cmu_arctic/
  ARCTIC/
    cmu_us_bdl_arctic/
      etc/txt.done.data
      wav/*.wav
```

### Invalid input handling

| Condition | Behavior |
|---|---|
| `speaker` not in supported set | Raises `ValueError` (`unsupported speaker: ...`). |
| Dataset path missing and `download=False` | Raises `FileNotFoundError` with guidance to use `download=True`. |
| Archive download/extract fails with `download=True` | Retries once, then propagates the original error. |
| Corrupt/unsafe archive member (path traversal, link entry) | Raises `ValueError` from safe extraction guard. |
| `load_audio(utterance_id)` for missing audio | Propagates backend audio I/O exception (for example, `soundfile` errors). |

## LibriSpeech

### Accepted options

`LibriSpeechDataset(root, subset="train-clean-100", speaker=None, download=False)`

| Option | Accepted values | Default | Notes |
|---|---|---|---|
| `root` | `Path` or path-like | required | Dataset root directory managed by the caller. |
| `subset` | One of `dev-clean`, `dev-other`, `test-clean`, `test-other`, `train-clean-100`, `train-clean-360`, `train-other-500` | `"train-clean-100"` | Valid values are defined in `VALID_SUBSETS`. |
| `speaker` | `None` or speaker directory name string (for example, `"103"`) | `None` | If set, loader is restricted to that speaker only. |
| `download` | `True` or `False` | `False` | If `True`, download/extract missing subset archive. |

### Expected local structure

For `root = datasets/librispeech` and `subset = train-clean-100`:

```text
datasets/librispeech/
  LibriSpeech/
    train-clean-100/
      <speaker_id>/
        <chapter_id>/
          <utt_id>.flac
          <speaker_id>-<chapter_id>.trans.txt
```

The loader finds utterances by scanning `*.trans.txt` files and keeping entries
whose `*.flac` files exist.

### Invalid input handling

| Condition | Behavior |
|---|---|
| `subset` not in supported set | Raises `ValueError` (`unsupported subset: ...`). |
| Subset path missing and `download=False` | Raises `FileNotFoundError` with guidance to use `download=True`. |
| `speaker` provided but speaker directory missing | Raises `FileNotFoundError` (`speaker directory not found: ...`). |
| `load_audio(utterance_id)` with malformed ID (not `spk-chapter-utt`) | Raises `ValueError` from tuple unpacking. |
| `load_audio(utterance_id)` for non-existing file | Propagates backend audio I/O exception (for example, `soundfile` errors). |
| Archive download/extract fails with `download=True` | Retries once, then propagates the original error. |
| Corrupt/unsafe archive member (path traversal, link entry) | Raises `ValueError` from safe extraction guard. |

## Shared utility behavior

### `load_dataset_sources`

`load_dataset_sources(dataset_factory, num_sources, duration_s, rng)` builds
fixed-duration signals by sampling speakers and concatenating utterances.

Accepted options and key constraints:

| Parameter | Accepted values | Invalid handling |
|---|---|---|
| `dataset_factory` | Callable that returns a `BaseDataset` | Runtime error if returned dataset has no speakers/sentences. |
| `num_sources` | Positive int, and must be `<=` available speakers | Raises `ValueError` if too large; may raise `RuntimeError` if no speakers. |
| `duration_s` | Positive float expected by caller | Non-positive values may produce empty/zero target behavior; treat as unsupported input in production pipelines. |
| `rng` | `random.Random` instance | Required for deterministic sampling behavior. |

Additional runtime checks:

- Raises `ValueError` if sampled speakers produce mixed sample rates.
- Raises `RuntimeError` if no audio can be loaded.

### `collate_dataset_items`

`collate_dataset_items(items, pad_value=0.0, keep_metadata=False)` builds a
padded batch for `DataLoader`.

| Condition | Behavior |
|---|---|
| Empty `items` | Raises `ValueError` (`collate_dataset_items received an empty batch`). |
| Mixed sample rates in a batch | Raises `ValueError` (`sample_rate must be consistent within a batch`). |
| Valid mixed lengths | Pads to `max_len` with `pad_value`. |

## Interaction with example scripts

The library loaders follow strict `download` behavior: `download=False` means
no network access.

`examples/build_dynamic_dataset.py` adds convenience fallback logic:

- If a dataset is missing and `--download` is not set, the script retries with
  download enabled.
- `--dataset` is argparse-constrained to `cmu_arctic` or `librispeech`.
- `--subset` is validated later by `LibriSpeechDataset` (`ValueError` on
  unsupported values).

Use `--dataset-dir` to point at a fully prepared local dataset tree when you
need predictable offline execution.

## Attribution and redistribution

For licensing and redistribution guidance, see
[`THIRD_PARTY_DATASETS.md`](https://github.com/taishi-n/torchrir/blob/main/THIRD_PARTY_DATASETS.md).

## Dynamic CMU ARCTIC builder (oobss-compatible)

`build_dynamic_cmu_arctic_dataset(...)` generates dynamic scenes in the layout
expected by `oobss` loader type `torchrir_dynamic`.

### Python API

```python
from pathlib import Path
from torchrir.datasets import build_dynamic_cmu_arctic_dataset

build_dynamic_cmu_arctic_dataset(
    cmu_root=Path("datasets/cmu_arctic"),
    dataset_root=Path("outputs/cmu_arctic_torchrir_dynamic_dataset"),
    n_scenes=2,
    overwrite=True,
)
```

Default behavior preserved from the legacy oobss builder:

- `n_sources=3`
- moving speed range `0.3-0.8` m/s
- motion profile: `0-35%` static, `35-65%` moving, `65-100%` static

### CLI usage

Module entrypoint:

```bash
python -m torchrir.datasets.dynamic_cmu_arctic \
  --cmu-root datasets/cmu_arctic \
  --dataset-root outputs/cmu_arctic_torchrir_dynamic_dataset \
  --n-scenes 10 \
  --save-layout-mp4 \
  --save-layout-mp4-3d \
  --overwrite-dataset
```

Console script entrypoint:

```bash
torchrir-build-dynamic-cmu-arctic \
  --cmu-root datasets/cmu_arctic \
  --dataset-root outputs/cmu_arctic_torchrir_dynamic_dataset \
  --n-scenes 10 \
  --layout-video-fps 12 \
  --overwrite-dataset
```

Useful video flags:

- `--no-save-layout-mp4`: disable MP4 rendering
- `--no-save-layout-mp4-3d`: skip `room_layout_3d.mp4`
- `--layout-video-fps <float>`: override frame rate
- `--layout-video-no-audio`: disable mixture-audio mux into MP4
- `--no-save-layout-images`: disable static layout images
- `--no-save-layout-images-3d`: skip `room_layout_3d.png`
- `--no-annotate-source-indices`: disable source index labels (`S0`, `S1`, ...)

### Output structure

```text
<dataset-root>/
  scene_0000/
    mixture.wav
    source_00.wav
    source_01.wav
    ...
    metadata.json
    source_info.json
    room_layout_2d.png
    room_layout_3d.png
    room_layout_2d.mp4
    room_layout_3d.mp4
  scene_0001/
    ...
```
