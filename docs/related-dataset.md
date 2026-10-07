# Related dataset

TorchRIR corpus loaders provide individual speech utterances as inputs to RIR
convolution. CMU ARCTIC and LibriSpeech are implemented; VCTK and WSJ0 are
planned source corpus integrations. Current APIs and the source loader scope
are documented on the [Datasets](datasets.md) page.

This page describes the planned source corpora and their use in acoustic
datasets. The acoustic datasets and benchmarks below are research references
outside the source corpus loader roadmap. They provide context for speech
separation, enhancement, localization, and dynamic room acoustic experiments.

## Planned source corpora

### VCTK

- **Status:** Planned individual-utterance loader.
- **Source speech:** The CSTR VCTK Corpus version 0.92 contains read speech
  from 110 English speakers with varied accents. Recordings were made in a
  hemi-anechoic chamber using two microphones; the released audio is 48 kHz.
- **Reason for inclusion:** Individual speaker utterances provide source
  waveforms for applying a chosen TorchRIR room response. Speaker and accent
  diversity expands the material available for simulation experiments.
- **Loader scope:** Select the microphone explicitly, index available
  speaker/utterance recordings and transcripts, preserve the native sample
  rate, and record corpus attribution. Missing microphone recordings and
  transcripts require an explicit indexing policy.
- **Related corpora:** [LOCATA](#locata) replays VCTK audio in Tasks 1 and 2;
  live talkers read selected VCTK sentences in Tasks 3--6.
  [trajectoRIR](#trajectorir) uses VCTK material for its female-speech playback
  signal, as identified in Table 3 of its
  [paper](https://arxiv.org/html/2503.23004).
- **Official corpus:** [CSTR VCTK Corpus version 0.92](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928).

### WSJ0

- **Status:** Planned individual-utterance loader.
- **Source speech:** CSR-I (WSJ0) contains English read speech with
  transcriptions. The LDC release specifies single-channel, 16 kHz audio.
- **Reason for inclusion:** Original speaker utterances provide inputs for
  controlled RIR convolution and source mixing, and are used as source
  material in speech-separation experiments.
- **Loader scope:** Read individual utterances, speaker IDs, and transcripts
  from a user-supplied, licensed local corpus, preserving the native sample
  rate and attribution.
- **Related corpora:** [WSJ0-Demand-6ch-Move](#wsj0-demand-6ch-move) selects two
  WSJ0 utterances, propagates them through simulated moving-source RIRs, and
  adds measured environmental noise. WSJ0 supplies the speech before this
  propagation and mixing, as described in the
  [generation recipe](https://arxiv.org/html/2602.22487v1#S5.SS1).
- **Official corpus:** [CSR-I (WSJ0) Complete, LDC93S6A](https://catalog.ldc.upenn.edu/LDC93S6A).

## Terminology

- **Simulated synthesis** means that dry source recordings are propagated
  through a numerically simulated acoustic environment. The source recordings
  may themselves be real recordings, but the multichannel room response is not
  captured with physical microphones.
- **Hybrid synthesis** combines simulated propagation with components recorded
  using physical microphones, such as measured environmental noise.
- **Real recording** means that the propagation path and microphone signals
  are captured in a physical environment. Some real-recorded datasets use
  loudspeakers to reproduce source material instead of live talkers.
- **Publication year** refers to the formal publication year of the paper
  linked in the **Related paper** field.

## Acoustic datasets and benchmarks

### LOCATA

- **Intended use:** Acoustic source localization and tracking. Tasks 4 and 6
  also provide multiple moving talkers that can support separation-oriented
  evaluation.
- **Motion condition:** Task 4 uses multiple moving talkers and a static array;
  Task 6 uses multiple moving talkers and a moving, rotating array.
- **Acquisition type:** Real recording.
- **Playback or source emission:** Human talkers read VCTK sentences while
  moving in Tasks 3--6; static-source tasks use loudspeakers reproducing VCTK
  material.
- **Source datasets or recordings:** Tasks 1 and 2 replay speech from the
  [CSTR VCTK Corpus](https://datashare.ed.ac.uk/items/30e7453c-9ea8-48b4-8e18-f96d0dc62928).
  The live talkers in Tasks 3--6 read sentences selected from the same corpus.
- **Recording or capture method:** Recordings were made in a real reverberant
  room using the 15-channel DICIT array, 32-channel Eigenmike, 12-channel NAO
  head array, and hearing-aid arrays. Close-talk microphones and optical pose
  tracking provide references.
- **Official dataset:** [LOCATA corpus](https://www.locata.lms.tf.fau.de/corpus/).
- **Related paper:** [The LOCATA Challenge: Acoustic Source Localization and Tracking](https://doi.org/10.1109/TASLP.2020.2990485).
- **Publication year:** 2020.

### EasyCom

- **Intended use:** Wearable speech enhancement, speaker extraction,
  localization, and recognition in conversation.
- **Motion condition:** Conversation participants and the wearer of the AR
  glasses move and rotate.
- **Acquisition type:** Real recording.
- **Playback or source emission:** Participants engage in live conversation;
  restaurant-like background noise is reproduced through ten loudspeakers in
  the room.
- **Source datasets or recordings:** The conversation speech is produced live.
  The paper and official repository do not identify an external dataset for
  the restaurant-like noise replayed by the ten loudspeakers.
- **Recording or capture method:** A six-channel array mounted on the
  listener's AR glasses records the scene. Close-talk microphones, video,
  six-degree-of-freedom poses, voice activity, and transcripts are
  synchronized.
- **Relation to source loading:** Participant close-microphone recordings are
  available as mono references. Using them as source utterances requires
  segment selection and assessment of other-talker and background-noise
  leakage. The synchronized conversation dataset is an external workflow.
- **Official dataset:** [EasyCom dataset](https://github.com/facebookresearch/EasyComDataset).
- **Related paper:** [EasyCom: An Augmented Reality Dataset to Support Algorithms for Easy Communication in Noisy Environments](https://arxiv.org/abs/2107.04174).
- **Publication year:** 2021.

### RealMAN

- **Intended use:** Dynamic speech enhancement and source localization.
- **Motion condition:** Approximately 48.3 hours use a static source and
  35.4 hours use a moving source; the array is fixed.
- **Acquisition type:** Real recording with loudspeaker reproduction.
- **Playback or source emission:** Clean Mandarin speech is reproduced through
  a physical loudspeaker that is held static or moved; environmental noise is
  recorded separately in the target environments.
- **Source datasets or recordings:** The replay material is a close-sourced
  collection of nearly 35 hours of clean Mandarin speech recorded for RealMAN,
  rather than a named public source corpus. The paper does not provide an
  external dataset link for it.
- **Recording or capture method:** A physical 32-channel array records indoor,
  outdoor, semi-outdoor, and transportation scenes. Camera-based source
  positions and direct-path targets are provided.
- **Relation to source loading:** The `dp_speech` targets are obtained by
  filtering source speech with an estimated direct-path propagation filter.
  They are processed references rather than the original playback utterances.
  See the [official dataset description](https://github.com/Audio-WestlakeU/RealMAN#introduction).
- **Official dataset:** [RealMAN dataset](https://github.com/Audio-WestlakeU/RealMAN).
- **Related paper:** [RealMAN: A Real-Recorded and Annotated Microphone Array Dataset for Dynamic Speech Enhancement and Localization](https://proceedings.neurips.cc/paper_files/paper/2024/file/bf8f6f5b017dc60d0c4e28a7a9a4ee7b-Paper-Datasets_and_Benchmarks_Track.pdf).
- **Publication year:** 2024.

### SonicSet v2

- **Intended use:** Moving-speaker speech separation and speech enhancement.
- **Motion condition:** Multiple sources move through three-dimensional indoor
  scenes; microphones are normally fixed.
- **Acquisition type:** Simulated synthesis.
- **Simulation tools or libraries:** [SonicSim](https://github.com/JusperLee/SonicSim),
  built on [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) and
  its audio-rendering capabilities associated with
  [SoundSpaces 2.0](https://github.com/facebookresearch/sound-spaces).
- **Playback or source emission:** LibriSpeech and other source recordings are
  emitted as virtual sources and propagated by SonicSim. No physical
  loudspeaker is used for the main synthetic dataset.
- **Source datasets or recordings:** Speech comes from
  [LibriSpeech](https://www.openslr.org/12/), environmental noise from
  [FSD50K](https://zenodo.org/records/4060432), and musical noise from the
  [Free Music Archive (FMA)](https://github.com/mdeff/fma).
- **Recording or capture method:** Matterport3D scene models and bidirectional
  path tracing generate signals for mono, binaural, Ambisonics, and custom
  virtual microphone arrays.
- **Relation to source loading:** The existing LibriSpeech loader supplies the
  speech inputs. The per-speaker `moving_audio_*.wav` files are RIR-convolved
  recordings, as shown in the
  [generation code](https://github.com/JusperLee/SonicSim/blob/main/SonicSim-SonicSet/SonicSet.py#L72-L106).
- **Official dataset:** [SonicSet v2](https://huggingface.co/datasets/JusperLee/SonicSet-v2).
- **Related paper:** [SonicSim: A Customizable Simulation Platform for Speech Processing in Moving Sound Source Scenarios](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a8633d27d782f66fe660c2fb4bae446e-Abstract-Conference.html).
- **Publication year:** 2025.

### ASA_20k_4s_nspk2-4

- **Intended use:** Multichannel universal sound separation and polyphonic
  audio classification; speech is one of the included source classes.
- **Motion condition:** Two to four foreground sources; each source moves with
  75% probability at a speed from 0 to 3 m/s; the microphone array is fixed.
- **Acquisition type:** Simulated synthesis.
- **Simulation tools or libraries:** [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR)
  is used to generate the dynamic multichannel room responses.
- **Playback or source emission:** Recordings from LibriSpeech, FSD50K,
  MUSDB18, and other corpora are used as virtual sources and convolved with
  gpuRIR responses; diffuse background noise is added.
- **Source datasets or recordings:** Foreground sounds come from
  [Pixabay Sound Effects](https://pixabay.com/sound-effects/),
  [FSD50K](https://zenodo.org/records/4060432),
  [LibriSpeech](https://www.openslr.org/12/),
  [MUSDB18](https://zenodo.org/records/1117372), and
  [VocalSound](https://sls.csail.mit.edu/downloads/vocalsound/). The diffuse
  background noise comes from the
  [TAU Spatial Room Impulse Response Database and TAU-SNoise](https://zenodo.org/records/6408611).
- **Recording or capture method:** A virtual four-channel tetrahedral array
  with a 4.2 cm radius is placed in a shoebox room with an RT60 from 0.2 to
  0.6 s.
- **Relation to source loading:** Source material must be selected from the
  original corpora. The published generator applies a direct-path RIR even
  to signals saved as `anechoic`, distinguishing those references from its
  input waveforms. See the
  [generation code](https://github.com/donghoney0416/DeFTMamba/blob/main/AudioDataset2.py#L187-L216).
- **Official dataset:** [Auditory Scene Analysis dataset](https://zenodo.org/records/13749621).
- **Related paper:** [DeFT-Mamba: Universal Multichannel Sound Separation and Polyphonic Audio Classification](https://doi.org/10.1109/ICASSP49660.2025.10890324).
- **Publication year:** 2025.

### WSJ0-Demand-6ch-Move

- **Intended use:** Noisy two-speaker moving speech separation.
- **Motion condition:** Two speakers move independently along straight
  trajectories at speeds from 0 to 1 m/s; the six-channel array is fixed.
- **Acquisition type:** Hybrid synthesis.
- **Simulation tools or libraries:** [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR)
  is used to simulate the six-channel reverberant speech for each moving
  source. The paper does not name a separate dataset-generation framework.
- **Playback or source emission:** WSJ0 utterances are convolved with dynamic
  simulated RIRs; environmental noise from real DEMAND recordings is mixed
  into the result.
- **Source datasets or recordings:** Speech comes from the licensed
  [WSJ0 source corpus](#wsj0), and
  environmental noise comes from
  [DEMAND](https://zenodo.org/records/1227121).
- **Recording or capture method:** Speech propagation uses a virtual
  six-channel circular array based on the DEMAND microphone layout; the noise
  component comes from physical DEMAND array recordings.
- **Relation to source loading:** The planned WSJ0 loader exposes the original
  utterances. The generated reverberant mixtures and multichannel noise belong
  to the benchmark's synthesis workflow.
- **Dataset recipe:** [Experimental setup, Section V-A](https://arxiv.org/html/2602.22487v1#S5.SS1).
- **Related paper:** [Moving Speaker Separation via Parallel Spectral-Spatial Processing](https://doi.org/10.1109/TASLPRO.2026.3671599).
- **Publication year:** 2026.

### trajectoRIR

- **Intended use:** Dynamic RIR analysis, moving-microphone localization and
  tracking, sound-field reconstruction, auralization, and system
  identification.
- **Motion condition:** Sources are fixed; microphone arrays move along an
  L-shaped trajectory at 0.2, 0.4, or 0.8 m/s.
- **Acquisition type:** Real recording with loudspeaker reproduction.
- **Playback or source emission:** Two fixed loudspeakers reproduce sweeps,
  speech, music, and noise.
- **Source datasets or recordings:** The female-speech signal comes from
  [VCTK](#vctk), as identified in Table 3 of the
  [paper](https://arxiv.org/html/2503.23004). The piano, drum, female-speech,
  white-noise, and sweep source files are included under `audio/SRC` in the
  [trajectoRIR dataset archive](https://zenodo.org/records/15564430).
- **Recording or capture method:** A robotic cart moves physical dummy-head,
  first-order Ambisonics, 16- and 4-channel circular, and 12-channel linear
  arrays through a real room; stationary RIRs are also measured along the
  trajectory.
- **Relation to TorchRIR:** Measured RIRs and moving-microphone recordings can
  support dynamic-convolution validation. The included playback signals
  accompany that experiment; the planned speech corpus loader targets VCTK.
- **Official dataset:** [trajectoRIR dataset](https://zenodo.org/records/15564430).
- **Related paper:** [The trajectoRIR Database: Room Acoustic Recordings Along a Trajectory of Moving Microphones](https://doi.org/10.1186/s13636-026-00449-2).
- **Publication year:** 2026.

## Scope notes

SonicSet v2 also links a separately distributed real-world speech-separation
evaluation set. The classification above describes the main SonicSet synthetic
corpus rather than that additional recording.

LOCATA, RealMAN, and trajectoRIR are not purpose-built multi-talker speech
separation benchmarks. They are included because their measured dynamic
acoustics, source references, or trajectories are useful for evaluating
components of a moving-source or moving-array separation system.
