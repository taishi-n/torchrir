# Third-Party Dataset Notices

TorchRIR provides dataset loaders/examples and does not bundle CMU ARCTIC or
LibriSpeech by default. If you redistribute raw or derived data built from
these corpora, include attribution and license notices with your artifacts.

## CMU ARCTIC

- Source: http://www.festvox.org/cmu_arctic/
- License: permissive, attribution required (see upstream `COPYING`)
- Attribution: Carnegie Mellon University, Language Technologies Institute
- Redistribution: retain upstream notices and clearly mark modifications

## LibriSpeech (OpenSLR SLR12)

- Source: https://www.openslr.org/12
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- License URL: https://creativecommons.org/licenses/by/4.0/
- Attribution: LibriSpeech corpus contributors
- Redistribution: provide attribution, include license link, and indicate
  modifications

## Recommended Practice for Derived Dataset Releases

- Include an `ATTRIBUTION.txt` in your release root.
- Keep a machine-readable note in metadata (`dataset_license`,
  `modifications`).
- Preserve upstream license text/files whenever available.
