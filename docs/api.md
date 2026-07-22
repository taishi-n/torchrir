# API

## Public API

This reference contains implemented public modules only. Planned backends and
dataset integrations are documented as future work rather than exposed as
placeholder classes.

For the physical meaning of dynamic convolution time references, frame
schedules, and simulation settings,
read [Overview](overview.md). Released behavior changes are recorded in the
[Changelog](changelog.md).

## Modules

### `torchrir`

The top-level facade re-exports `Room`, `Source`, `MicrophoneArray`,
`StaticScene`, `DynamicScene`, and `RIRResult`. Their definitions appear once
under `torchrir.models` below.

### `torchrir.sim`

::: torchrir.sim
    options:
      members:
        - directivity_gain
        - simulate

### `torchrir.signal`

::: torchrir.signal
    options:
      members:
        - DynamicConvolver
        - FrameSchedule
        - convolve_rir
        - fft_convolve

`FrameSchedule` stores exact CPU `int64` starts. A scene-owned schedule is
consumed automatically from a dynamic `RIRResult`; otherwise pass it to
`DynamicConvolver.convolve`. `FrameSchedule.from_seconds` retains its
conversion sample rate and rejects use with a different room sample rate.

### `torchrir.geometry`

::: torchrir.geometry
    options:
      members:
        - binaural_array
        - circular_array
        - clamp_positions
        - eigenmike_em32
        - eigenmike_em64
        - linear_array
        - linear_trajectory
        - polyhedron_array
        - sample_positions
        - sample_positions_min_distance

### `torchrir.viz`

::: torchrir.viz
    options:
      members:
        - animate_scene_gif
        - animate_scene_mp4
        - plot_scene_dynamic
        - plot_scene_static
        - render_scene_plots
        - save_scene_gifs
        - save_scene_layout_images
        - save_scene_plots
        - save_scene_videos

### `torchrir.models`

::: torchrir.models
    options:
      members:
        - Room
        - Source
        - MicrophoneArray
        - StaticScene
        - DynamicScene
        - RIRResult

### `torchrir.io`

::: torchrir.io
    options:
      members:
        - AudioData
        - AudioInfo
        - build_metadata
        - build_result_metadata
        - info_audio
        - info_wav
        - load_audio
        - load_audio_data
        - load_wav
        - save_attribution_file
        - save_audio
        - save_audio_data
        - save_metadata_json
        - save_result_metadata
        - save_scene_audio
        - save_scene_metadata
        - save_wav

Audio save functions preserve gain by default (`normalize=False`).
`load_audio` and `load_audio_data` accept either a `Path` or a caller-owned
open, seekable binary stream. Strings and arbitrary objects raise `TypeError`;
closed or non-seekable streams raise `ValueError`. One SoundFile handle supplies
both metadata and samples, and a supplied stream remains open after the call.
`AudioInfo` normalizes Python/NumPy integer metadata: sample rate is limited to
`1..2**31-1`, frame count to non-negative `int64`, and channel count to positive
`int32`.
`save_audio_data` reuses its stored subtype only when the destination container
matches the loaded format; otherwise WAV output without an explicit subtype
uses `FLOAT`. Non-floating subtypes reject samples outside `[-1, 1]`. Metadata
builders use explicit `schedule` and `time_reference` arguments, emit
`torchrir.scene` schema version 1 with generator provenance and compact
RIR/sample axes, and reject a time reference inconsistent with scene motion;
JSON publication is finite-only and atomic. See
[Metadata schema version 1](overview.md#metadata-schema-version-1).

### `torchrir.logging`

::: torchrir.logging
    options:
      members:
        - LoggingConfig
        - get_logger
        - setup_logging

### `torchrir.config`

::: torchrir.config
    options:
      members:
        - RIRHighPassConfig
        - ResolvedSimulationConfig
        - SimulationConfig

`SimulationConfig` contains requested values. Scene-oriented simulation stores
the fully resolved `ResolvedSimulationConfig` in `RIRResult.config`, including
backend-effective LUT and compile flags.

### `torchrir.util`

::: torchrir.util
    options:
      members:
        - DeviceSpec
        - add_output_args
        - as_float_tensor
        - as_tensor
        - attenuation_db_to_time_sabine
        - ensure_dim
        - estimate_beta_from_t60
        - estimate_image_counts_from_tmax
        - estimate_t60_from_beta
        - extend_size
        - normalize_orientation
        - orientation_to_unit
        - resolve_device

`DeviceSpec` and `resolve_device` are the canonical device utilities.

### `torchrir.datasets`

::: torchrir.datasets
    options:
      members:
        - BaseDataset
        - CmuArcticDataset
        - CmuArcticSentence
        - CollateBatch
        - DatasetAttribution
        - DatasetItem
        - DynamicCmuArcticBuildConfig
        - DynamicDatasetBuildResult
        - LibriSpeechDataset
        - LibriSpeechSentence
        - SentenceLike
        - attribution_for
        - build_dynamic_cmu_arctic
        - choose_speakers
        - cmu_arctic_speakers
        - collate_dataset_items
        - default_modification_notes
        - load_dataset_sources

`DatasetItem` is a keyword-only validated mono-audio record with an explicit
metadata payload. `collate_dataset_items` revalidates each item, requires one
sample rate/dtype/device, and returns immutable metadata sequences. Dataset
downloads verify pinned checksums and extract only regular files/directories.
Secure corpus filesystem operations are available only on Linux and macOS with
the required POSIX descriptor-walk and atomic rename primitives; unsupported
platforms/filesystems raise `NotImplementedError`.
