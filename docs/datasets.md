# Datasets

This page documents the `torchrir.datasets` helpers for CMU ARCTIC and
LibriSpeech, including accepted options, expected directory structures, and
error handling for invalid inputs.

For external corpora involving moving sources or microphone arrays, see
[Related Dynamic Speech and Acoustic Datasets](related-datasets.md). Those
datasets are research references and are not currently built-in integrations.

## Scope

Covered APIs:

- `torchrir.datasets.CmuArcticDataset`
- `torchrir.datasets.LibriSpeechDataset`
- `torchrir.datasets.load_dataset_sources`
- `torchrir.datasets.collate_dataset_items`
- `torchrir.datasets.DynamicCmuArcticBuildConfig`
- `torchrir.datasets.build_dynamic_cmu_arctic`

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

When `download=False`, dataset loaders never start a network download. They may
finish recovery of an interrupted local publication before checking readiness,
then raise an error if the requested tree is missing or incomplete. `download`
must be an actual `bool`; truthy integers and strings are rejected.

Secure filesystem access for these corpus loaders and the dynamic builder is
implemented only on Linux and macOS. It requires POSIX directory-descriptor
walking plus atomic no-replace and exchange renames. An unsupported platform or
missing descriptor primitive raises `NotImplementedError` before local dataset
I/O; a filesystem that refuses the required atomic rename fails at that rename
instead of using a weaker fallback.

## CMU ARCTIC

### Accepted options

`CmuArcticDataset(root, speaker="bdl", download=False)`

| Option | Accepted values | Default | Notes |
|---|---|---|---|
| `root` | `Path` or path-like | required | Dataset root directory managed by the caller. |
| `speaker` | One of `aew`, `ahw`, `aup`, `awb`, `axb`, `bdl`, `clb`, `eey`, `fem`, `gka`, `jmk`, `ksp`, `ljm`, `lnh`, `rms`, `rxr`, `slp`, `slt` | `"bdl"` | Valid IDs are defined in `VALID_SPEAKERS`. |
| `download` | `True` or `False` | `False` | If `True`, download/extract missing archive for the selected speaker. |

Enumerate every supported speaker, or only speakers installed below a root,
with:

```python
from torchrir.datasets import cmu_arctic_speakers

print(cmu_arctic_speakers())
print(cmu_arctic_speakers(Path("datasets/cmu_arctic")))
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

A speaker tree is usable only when every non-empty line in `txt.done.data`
parses with the CMU ARCTIC transcript syntax and contains a canonical
`arctic_<letter><four-digits>` ID. At least one of those IDs must have a
same-named WAV file; audio is not required for every otherwise valid transcript
entry. This same predicate drives construction, `cmu_arctic_speakers(root)`,
and the download fast path. The speaker dataset root, intermediate `etc`/`wav`
directories, and final transcript/audio paths must not be symlinks. Each path
component is opened relative to an already-open directory descriptor with
`O_NOFOLLOW`; transcripts and audio are parsed or decoded through that same
final file descriptor instead of reopening a validated pathname.

### Invalid input handling

| Condition | Behavior |
|---|---|
| `speaker` is not a string | Raises `TypeError`. |
| String `speaker` not in supported set | Raises `ValueError` (`unsupported speaker: ...`). |
| Dataset tree missing, partial, or without a canonical transcript/audio pair and `download=False` | Raises `FileNotFoundError` with guidance to use `download=True`. |
| Usable tree and `download=True` | Returns without checking an archive or accessing the network. |
| HTTP 408/429/5xx, transport, malformed/mismatched response length, or digest failure during download | Uses HTTPS, verifies the pinned per-speaker SHA-256, and retries once by default before propagating the error. |
| Declared or streamed body exceeds the 64 GiB safety limit | Raises `ValueError` without retrying. |
| Local filesystem, extraction, validation, or publication failure | Propagates immediately without a download retry or re-downloading an already verified archive. |
| Cached archive path is a symlink or another non-regular entry | A symlink is replaced only through a verified atomic exchange without following its target; any other non-regular cache entry is rejected. |
| Corrupt/unsafe or excessive archive (path traversal, link/special-file entry, too many members, or declared size over a bound) | Raises `ValueError` before any member is extracted. |
| Malformed or path-like `utterance_id` | Raises `ValueError` before filesystem access. |
| `load_audio(utterance_id)` for missing audio or a directory/special file in place of audio | Raises `FileNotFoundError` before SoundFile is called. |
| `load_audio(utterance_id)` detects a symlink at the speaker dataset root or final audio entry, or a resolved audio path outside the speaker tree | Raises `ValueError` during path validation, before SoundFile is called. |

## LibriSpeech

### Accepted options

`LibriSpeechDataset(root, subset="train-clean-100", speaker=None, download=False)`

| Option | Accepted values | Default | Notes |
|---|---|---|---|
| `root` | `Path` or path-like | required | Dataset root directory managed by the caller. |
| `subset` | One of `dev-clean`, `dev-other`, `test-clean`, `test-other`, `train-clean-100`, `train-clean-360`, `train-other-500` | `"train-clean-100"` | Valid values are defined in `VALID_SUBSETS`. |
| `speaker` | `None` or a numeric speaker ID string (for example, `"103"`) | `None` | If set, loader is restricted to that speaker only. |
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

The loader scans regular `*.trans.txt` files directly below non-symlink numeric
chapter directories of non-symlink numeric speaker trees. A speaker tree is
usable only when every non-empty entry in every transcript found by that scan
contains a canonical `speaker-chapter-utterance` ID whose speaker and chapter
components agree with its directories. At least one such entry must have a
same-named regular FLAC; audio is not required for every otherwise valid
transcript entry. Unrelated transcript-like files outside this canonical tree
and unusable stray speaker trees are ignored. The same predicate drives
speaker/subset discovery, construction, and the download fast path. The subset
root, numeric speaker/chapter directories, and final transcript/audio paths must
not be symlinks. Each final entry must be regular and is consumed through the
same descriptor obtained by a component-wise `O_NOFOLLOW` walk.

### Invalid input handling

| Condition | Behavior |
|---|---|
| `subset` is not a string, or `speaker` is neither `None` nor a string | Raises `TypeError`. |
| String `subset` not in supported set, or string `speaker` is empty/non-numeric | Raises `ValueError`. |
| Subset tree missing, partial, or without a canonical transcript/audio pair and `download=False` | Raises `FileNotFoundError` with guidance to use `download=True`. |
| `speaker` provided but its directory is missing or incomplete | Raises `FileNotFoundError` (`speaker directory is missing or incomplete: ...`). |
| Usable subset and `download=True` | Returns without checking an archive or accessing the network. With `speaker=...`, this fast path requires that specific speaker to be usable; another ready speaker does not mask a partial requested tree. |
| `load_audio(utterance_id)` with malformed ID (not numeric `spk-chapter-utt`) | Raises `ValueError` before filesystem access. |
| `utterance_id` belongs to a different configured speaker | Raises `ValueError`. |
| Transcript ID conflicts with its speaker/chapter directory | Raises `ValueError` while discovering sentences. |
| `load_audio(utterance_id)` for missing audio or a directory/special file in place of audio | Raises `FileNotFoundError` before SoundFile is called. |
| `load_audio(utterance_id)` detects a symlink at the subset, speaker, or chapter directory or final audio entry, or a resolved audio path outside the subset tree | Raises `ValueError` during path validation, before SoundFile is called. |
| HTTP 408/429/5xx, transport, malformed/mismatched response length, or digest failure during download | Verifies the pinned subset MD5 and retries once by default before propagating the error. |
| Declared or streamed body exceeds the 64 GiB safety limit | Raises `ValueError` without retrying. |
| Local filesystem, extraction, validation, or publication failure | Propagates immediately without a download retry or re-downloading an already verified archive. |
| Cached archive path is a symlink or another non-regular entry | A symlink is replaced only through a verified atomic exchange without following its target; any other non-regular cache entry is rejected. |
| Corrupt/unsafe or excessive archive (path traversal, link/special-file entry, too many members, or declared size over a bound) | Raises `ValueError` before any member is extracted. |

## Shared archive, concurrency, and recovery contract

CMU ARCTIC and LibriSpeech use the same verified transfer implementation:

- Dataset filesystem operations require Linux or macOS with POSIX `dir_fd`,
  `O_DIRECTORY`, and `O_NOFOLLOW`, plus Linux `renameat2` or macOS
  `renameatx_np` no-replace/exchange support. The implementation raises
  `NotImplementedError` rather than weakening the contract on unsupported
  platforms or filesystems.
- Each connection and response-body read has a 60-second I/O timeout. A
  six-hour total transfer deadline starts before connection setup, and its
  remaining time is applied to each connection/read timeout. The deadline
  bounds network progress; hashing, filesystem scheduling, and cleanup are not
  hard real-time operations. The response is streamed into an exclusively
  created `archive.part` inside a fresh marker-owned sibling workspace; no
  shared partial-download path is reused or followed.
- A transfer may contain at most 64 GiB. If supplied, `Content-Length` must be
  a valid non-negative integer no greater than that limit. The body is still
  read through EOF: a short or overlong response is rejected, as is a body
  without `Content-Length` that crosses the byte limit. The pinned digest must
  always match before publication.
- The default single retry is limited to HTTP 408, HTTP 429, HTTP 5xx,
  connection/response-body transport failures, malformed response length, and
  declared-length/digest integrity failures. The 64 GiB safety limit, other
  HTTP statuses, and local filesystem errors are not retried. Archive opening,
  extraction, readiness validation, and publication are also outside the retry
  operation, so a verified cached archive remains available when one of those
  local steps fails.
- An absent archive destination is published with an atomic no-replace rename.
  An existing regular file or symlink is replaced only by an atomic
  exchange/swap after its device, inode, and type have been snapshotted. The
  displaced entry must match that snapshot; otherwise the exchange is reversed
  and the racing third-party entry is retained. Symlink targets are never
  followed, and other non-regular destination types are rejected.
- Local transcript, audio, and archive files are opened by walking every path
  component from an already-open root directory descriptor with `O_NOFOLLOW`.
  Parsing, SoundFile decoding, hashing, and tar parsing consume that same final
  file descriptor. A pathname cannot be swapped between validation and use.
- Every member is prevalidated before extraction. The allowlist contains only
  regular files and directories below the staging root; traversal, links, and
  special files are rejected. Declared content is bounded to 1,000,000 members,
  8 GiB for one file, and 512 GiB in total. Archive payloads are extracted below
  a child directory so they cannot overwrite the workspace ownership marker.

Dataset repair has two lock scopes. A CMU speaker lock or LibriSpeech subset
lock is acquired before stale-workspace cleanup and the readiness check. While
holding it, the loader removes only empty or marker-verified workspaces reserved
for that archive/target and, only if repair is still necessary, performs cache
inspection, download, extraction, staged validation, and publication. Unsafe,
symlinked, or unowned matching paths are retained and reported. Cleanup failure
is a warning and does not replace the primary operation result. Lock acquisition
has a strict 1800-second default deadline, checked before and after non-blocking
`flock`; the acquired path is then revalidated against the descriptor to prevent
split-brain locking. The operating system releases the advisory lock if a
process exits. Persistent hidden regular lock files are not deleted after each
writer.

Staged publication has a separate sibling lock for each final target. That lock
is held only while recovering a prior transaction or committing the staged
tree: it covers the final target/type and create-only checks, the old-target
backup rename, the staged-tree rename, immediate rollback, and transaction
cleanup. Every target/backup/staged rename uses the platform's atomic
no-replace primitive, so an entry that races into a previously absent name is
never overwritten.

Replacing an existing target records its previous tree in a deterministic
sibling transaction directory. A fsynced schema-versioned manifest records the
device/inode identities of the staged and previous trees plus the
`initialized`, `backed_up`, or `published` phase. A normal publication failure
attempts immediate identity-checked rollback. If an atomic rename completed but
then reported an error, the implementation identifies the actual outcome
before deciding whether to return, roll back, or retain the transaction.

After interruption, the next loader or build restores a missing target only
from the manifest-recorded previous identity, and cleans state only when the
target is one of the recorded identities. A third-party target, changed backup,
unsafe symlink, malformed manifest, or unknown transaction entry stops recovery
without deleting the retained backup. Only recognized owned manifest temporary
files may be reclaimed automatically.

The dynamic builder adds a broader per-output build lock. It acquires that lock
before transaction recovery, the `overwrite`/existence check, marker-owned
stale build-workspace cleanup, corpus loading, or scene generation. Builds for
the same target are serialized for their complete expensive operation; the
target-specific publication lock remains a narrower nested lock for the final
transaction. Builds for different output targets remain independent.

## Shared utility behavior

### Dataset interfaces and records

`BaseDataset` is an abstract `torch.utils.data.Dataset` contract. Concrete
subclasses must implement speaker discovery, sentence discovery, audio loading,
and attribution; an incomplete subclass cannot be instantiated.
`CmuArcticSentence`, `LibriSpeechSentence`, and `DatasetAttribution` are frozen,
slotted, keyword-only records. Sentence construction validates canonical corpus
identifiers and, for LibriSpeech, agreement among utterance, speaker, and
chapter IDs. Attribution construction validates non-empty fields and an actual
boolean attribution flag.

### `load_dataset_sources`

`load_dataset_sources(*, dataset_factory, speakers, num_sources, duration_s,
rng)` builds fixed-duration signals by sampling an explicit speaker catalog and
concatenating utterances. All arguments are keyword-only. The factory receives
only selected string IDs; `None` is not used as a hidden catalog sentinel.

Accepted options and key constraints:

| Parameter | Accepted values | Invalid handling |
|---|---|---|
| `dataset_factory` | Callable from one speaker ID string to a `BaseDataset` | Raises `TypeError` if non-callable or if its result is not a `BaseDataset`; runtime error if a selected dataset has no sentences. |
| `speakers` | Non-empty sequence of unique, non-empty speaker ID strings | Raises `TypeError`/`ValueError`; use `cmu_arctic_speakers(root)` or a LibriSpeech catalog's `list_speakers()` for local data. |
| `num_sources` | Positive non-boolean int, and must be `<=` unique available speakers | Raises `TypeError`/`ValueError` for invalid values. |
| `duration_s` | Finite positive real number | Raises `TypeError`/`ValueError`; target length is `ceil(duration_s * sample_rate)`. |
| `rng` | `random.Random` instance | Raises `TypeError` otherwise. |

Additional runtime checks:

- Every loaded utterance must be a non-empty, finite, mono floating-point Tensor
  with a positive integer sample rate.
- Raises `ValueError` if sampled utterances/speakers mix sample rates, dtypes, or
  devices.

### `DatasetItem` and `collate_dataset_items`

`DatasetItem` is a frozen, slotted, keyword-only record. `audio` must be a
non-empty finite mono floating-point Tensor, `sample_rate` a positive integer,
and `utterance_id` a non-empty string. The optional `metadata` field is an
explicit pass-through payload. Because Tensor contents remain shallow-mutable,
collation calls `DatasetItem.validate()` again at the consumer boundary.

Sample rates accept non-boolean Python or NumPy integer scalars and are
normalized to Python `int`; the supported range is `1..2**31-1`. The same
range is used by corpus audio validation and dynamic build results.

`collate_dataset_items(items, pad_value=0.0, keep_metadata=False)` builds a
padded, frozen `CollateBatch` for `DataLoader`. Identifier/text/speaker
sequences and retained item metadata are normalized to tuples.

| Condition | Behavior |
|---|---|
| Empty `items` | Raises `ValueError` (`collate_dataset_items received an empty batch`). |
| Invalid or mutated `DatasetItem` | Revalidation raises before padding. |
| Mixed sample rates, dtypes, or devices in a batch | Raises `ValueError`. |
| Non-finite `pad_value` | Raises `ValueError`. |
| Valid mixed lengths | Pads to `max_len` with `pad_value`. |
| `keep_metadata=True` | Preserves one explicit metadata payload per item as a tuple. |

## Interaction with example scripts

The library loaders follow strict `download` behavior: `download=False` means
no network access.

`examples/build_dynamic_dataset.py` preserves the same strict policy:

- If a dataset is missing or incomplete and `--download` is not set, the
  script raises `FileNotFoundError` without network access.
- Passing `--download` explicitly authorizes fetching missing archives.
- `--dataset` is argparse-constrained to `cmu_arctic` or `librispeech`.
- `--subset` is validated later by `LibriSpeechDataset` (`ValueError` on
  unsupported values).

Use `--dataset-dir` to point at a fully prepared local dataset tree when you
need predictable offline execution.

## Attribution and redistribution

For licensing and redistribution guidance, see
[`THIRD_PARTY_DATASETS.md`](https://github.com/taishi-n/torchrir/blob/main/THIRD_PARTY_DATASETS.md).

## Dynamic CMU ARCTIC builder (oobss-compatible)

`build_dynamic_cmu_arctic(config)` generates dynamic scenes in the file format
expected by the `oobss` loader type `torchrir_dynamic`. This describes the
published artifact layout; `oobss` is not required to build the dataset.

The builder accepts one `DynamicCmuArcticBuildConfig` and returns a
`DynamicDatasetBuildResult` containing the output root, sample rate,
microphone count, scene count, and scene paths. Builds are staged in a sibling
temporary directory. The target directory is created or replaced only after
every scene has completed successfully, so a failed build leaves an existing
dataset unchanged.

Configuration construction performs all checks that depend only on requested
values. This includes positive counts and durations, source and moving-source
counts, trajectory length, motion-ratio ordering, moving-speed bounds, room
and source-margin geometry, Sabine T60 feasibility, simulation image-grid
dimension, and whether the octahedral microphone array and source radius fit
around a fixed or randomized microphone center. It also
rejects equal or nested `cmu_root` and `dataset_root` paths in either direction,
preventing an overwrite from deleting the source corpus. Invalid requests
therefore fail before a staging directory or dataset loader is created.
Filesystem existence, available utterances, per-utterance audio validity and
sample-rate consistency are checked when the build runs because they depend on
external state. Once the sample rate is known, the builder additionally
requires `trajectory_steps <= ceil(duration_sec * sample_rate)`, so every frame
has a distinct integer start sample.

The frozen config owns an immutable snapshot of its inputs. `cmu_root` and
`dataset_root` are normalized to `Path`; `speakers`, `room_size`, `mic_center`,
and `source_margin` are copied to tuples. Later mutation of lists supplied by a
caller cannot change an existing build request.

The builder acquires `.<target>.torchrir-build.lock` before recovery and the
initial `overwrite` check. While holding it, the builder removes only
marker-owned stale `.<target>.torchrir-build.*` workspaces, writes the complete
output below a new managed sibling workspace, and invokes the shared
publication transaction only after every scene succeeds. Thus two builds for
the same target do not duplicate expensive generation; the final commit still
uses the nested publication lock and repeats its target/create-only checks. See
[Shared archive, concurrency, and recovery contract](#shared-archive-concurrency-and-recovery-contract)
for the exact lock and rollback scope.

### Python API

```python
from pathlib import Path
import torch

from torchrir.config import SimulationConfig
from torchrir.datasets import (
    DynamicCmuArcticBuildConfig,
    build_dynamic_cmu_arctic,
)

config = DynamicCmuArcticBuildConfig(
    cmu_root=Path("datasets/cmu_arctic"),
    dataset_root=Path("outputs/cmu_arctic_torchrir_dynamic_dataset"),
    n_scenes=2,
    simulation=SimulationConfig(
        max_order=6,
        nsample=4096,
        device="auto",
        dtype=torch.float32,
    ),
    overwrite=True,
)
result = build_dynamic_cmu_arctic(config)

print(result.dataset_root)
print(result.scene_dirs)
```

Default output parameters include:

- `n_sources=3`
- `trajectory_steps=256`
- `simulation=SimulationConfig(max_order=6, nsample=4096, dtype=torch.float32)`
- moving speed range `0.3-0.8` m/s
- motion profile: `0-35%` static, `35-65%` moving, `65-100%` static

The `simulation` object is the sole owner of RIR algorithm settings. The
builder creates `FrameSchedule.uniform(...)` in exact integer samples and uses
those same starts for trajectory evaluation, scene simulation, convolution,
and metadata; it does not construct a second endpoint-inclusive time grid.
The requested duration is mapped to `ceil(duration_sec * sample_rate)` samples.
The resulting sample-aligned duration drives motion and is recorded beside the
requested duration in metadata. Before writing 32-bit floating-point WAV,
every reverberant stem and their summed mixture are checked together and, when
necessary, multiplied by one common scale capped at 0.99. Relative stem gains
and additivity are therefore preserved without clipping or independent PCM
quantization.

### CLI usage

Run these commands from the repository root. The `datasets` extra supplies
the builder's audio I/O and visualization dependencies.

Module entrypoint:

```bash
uv run --extra datasets python -m torchrir.datasets.dynamic_cmu_arctic \
  --cmu-root datasets/cmu_arctic \
  --dataset-root outputs/cmu_arctic_torchrir_dynamic_dataset \
  --n-scenes 10 \
  --save-layout-mp4 \
  --save-layout-mp4-3d \
  --overwrite-dataset
```

Console script entrypoint:

```bash
uv run --extra datasets torchrir-build-dynamic-cmu-arctic \
  --cmu-root datasets/cmu_arctic \
  --dataset-root outputs/cmu_arctic_torchrir_dynamic_dataset \
  --n-scenes 10 \
  --layout-video-fps 12 \
  --overwrite-dataset
```

Useful video flags:

- `--no-save-layout-mp4`: disable MP4 rendering
- `--no-save-layout-mp4-3d`: skip `room_layout_3d.mp4`
- `--layout-video-fps <float>`: request a positive frame rate; the frame count is
  rounded up and the effective rate preserves the complete mixture duration.
  Layout videos use the exact scene schedule and hold the final geometry
  through the reverberation tail.
- `--layout-video-no-audio`: disable mixture-audio mux into MP4
- `--no-save-layout-images`: disable static layout images
- `--no-save-layout-images-3d`: skip `room_layout_3d.png`
- `--no-annotate-source-indices`: disable source index labels (`S0`, `S1`, ...)

Simulation flags map into the same `SimulationConfig` contract used by the
Python API: `--max-order`, `--rir-samples`, `--device`, and `--dtype`. Both the
CLI and Python simulation API support `float32` and `float64`; lower-precision
ISM geometry is rejected before execution.

### Output structure

PNG files are present when layout images are enabled. MP4 files are produced
only when requested; requested MP4 output requires a system `ffmpeg`, and audio
muxing also requires the `audio` extra. Rendering failures abort the build instead
of silently omitting requested artifacts. The dataset is published only after
all requested outputs are produced.

```text
<dataset-root>/
  scene_0000/
    mixture.wav
    source_00.wav
    source_01.wav
    ...
    metadata.json
    source_info.json
    room_layout_2d.png       # optional
    room_layout_3d.png       # optional
    room_layout_2d.mp4       # optional
    room_layout_3d.mp4       # optional
  scene_0001/
    ...
```

Each `metadata.json` uses `torchrir.scene` schema version 1. It stores the
integer `frame_schedule.starts_samples`, emission-time convolution semantics,
and compact RIR/signal sample counts; it does not duplicate starts in seconds
or serialize a complete RIR time-axis array.
