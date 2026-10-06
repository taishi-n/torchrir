"""Execute public example/CLI paths with local synthetic speech, without network."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import yaml

from torchrir.datasets import cmu_arctic_speakers

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def corpus(tmp_path):
    root = tmp_path / "corpus"
    for index, speaker in enumerate(("bdl", "slt", "clb")):
        directory = root / "ARCTIC" / f"cmu_us_{speaker}_arctic"
        (directory / "wav").mkdir(parents=True)
        (directory / "etc").mkdir()
        t = np.arange(1000) / 8000
        sf.write(
            directory / "wav/arctic_a0001.wav",
            0.1 * np.sin(2 * np.pi * (220 + 90 * index) * t),
            8000,
            subtype="FLOAT",
        )
        (directory / "etc/txt.done.data").write_text(
            '( arctic_a0001 "synthetic fixture" )\n'
        )
    return root


def run_python(tmp_path, *arguments):
    result = subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=tmp_path,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def check_outputs(directory, prefix, mode, samples, microphones=2):
    audio, rate = sf.read(directory / f"{prefix}.wav", always_2d=True)
    meta = json.loads((directory / f"{prefix}_metadata.json").read_text())
    assert sf.info(directory / f"{prefix}.wav").subtype == "FLOAT"
    assert rate == meta["room"]["fs"] == meta["signal"]["sample_rate"] == 8000
    assert meta["signal"]["sample_count"] == samples
    assert audio.shape == (samples + meta["rir"]["sample_count"] - 1, microphones)
    assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 0
    assert len(meta["mics"]["positions"]) == microphones
    assert len(meta["sources"]["positions"]) == 2
    if mode == "static":
        assert meta["dynamic"] is False
        assert meta["frame_schedule"] is None and meta["convolution"] is None
    else:
        assert meta["dynamic"] is True
        assert meta["convolution"]["time_reference"] == (
            "emission" if mode == "dynamic_src" else "observation"
        )
        assert meta["convolution"]["output_sample_count"] == len(audio)
        starts = meta["frame_schedule"]["starts_samples"]
        assert starts == [i * samples // 3 for i in range(3)]
        assert len(meta["trajectories"]["sources"]) == len(starts)
        assert len(meta["trajectories"]["mics"]) == len(starts)
    assert (directory / "ATTRIBUTION.txt").is_file()
    return audio, meta


@pytest.mark.parametrize("mode", ["static", "dynamic_src", "dynamic_mic"])
@pytest.mark.parametrize("suffix", ["json", "yaml"])
def test_unified_cli_config_roundtrip_and_overrides(tmp_path, corpus, mode, suffix):
    first = tmp_path / "first"
    config = tmp_path / f"config.{suffix}"
    run_python(
        tmp_path,
        ROOT / "examples/cli.py",
        "--mode",
        mode,
        "--dataset-dir",
        corpus,
        "--no-download",
        "--num-sources",
        2,
        "--duration",
        0.08,
        "--steps",
        3,
        "--order",
        0,
        "--tmax",
        0.04,
        "--device",
        "cpu",
        "--out-dir",
        first,
        "--config-out",
        config,
    )
    original, _ = check_outputs(first, f"{mode}_binaural", mode, 640)
    saved = (
        json.loads(config.read_text())
        if suffix == "json"
        else yaml.safe_load(config.read_text())
    )
    assert saved["mode"] == mode and saved["download"] is False
    replay = tmp_path / "replay"
    run_python(
        tmp_path, ROOT / "examples/cli.py", "--config-in", config, "--out-dir", replay
    )
    repeated, _ = check_outputs(replay, f"{mode}_binaural", mode, 640)
    np.testing.assert_array_equal(original, repeated)
    overridden = tmp_path / "override"
    run_python(
        tmp_path,
        ROOT / "examples/cli.py",
        "--config-in",
        config,
        "--out-dir",
        overridden,
        "--duration",
        0.1,
    )
    _, meta = check_outputs(overridden, f"{mode}_binaural", mode, 800)
    assert meta["extra"]["args"]["duration"] == 0.1
    assert meta["extra"]["args"]["out_dir"] == str(overridden)


@pytest.mark.parametrize("mode", ["static", "dynamic_src", "dynamic_mic"])
def test_standalone_examples_save_consistent_references(tmp_path, corpus, mode):
    output = tmp_path / "output"
    args = [] if mode == "static" else ["--steps", "3"]
    run_python(
        tmp_path,
        ROOT / f"examples/{mode}.py",
        "--dataset-dir",
        corpus,
        "--no-download",
        "--num-sources",
        2,
        "--num-mics",
        3,
        "--duration",
        0.08,
        "--order",
        0,
        "--tmax",
        0.04,
        "--device",
        "cpu",
        "--out-dir",
        output,
        *args,
    )
    audio, meta = check_outputs(output, mode, mode, 640, microphones=3)
    refs = [
        sf.read(output / item["filename"], always_2d=True)[0]
        for item in meta["extra"]["reference_audio"]
    ]
    assert len(refs) == 2
    np.testing.assert_allclose(audio, sum(refs), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("dataset", ["cmu_arctic", "librispeech"])
def test_example_builder_uses_audio_sample_rate(tmp_path, corpus, dataset):
    if dataset == "cmu_arctic":
        for speaker in cmu_arctic_speakers():
            destination = corpus / "ARCTIC" / f"cmu_us_{speaker}_arctic"
            if not destination.exists():
                shutil.copytree(corpus / "ARCTIC/cmu_us_bdl_arctic", destination)
    else:
        for speaker in ("103", "104"):
            chapter = corpus / "LibriSpeech" / "dev-clean" / speaker / "1240"
            chapter.mkdir(parents=True)
            utterance = f"{speaker}-1240-0000"
            (chapter / f"{speaker}-1240.trans.txt").write_text(
                f"{utterance} SYNTHETIC FIXTURE\n"
            )
            sf.write(
                chapter / f"{utterance}.flac",
                np.linspace(-0.1, 0.1, 1000),
                8000,
            )
    output = tmp_path / "output"
    run_python(
        tmp_path,
        ROOT / "examples/build_dynamic_dataset.py",
        "--dataset",
        dataset,
        "--dataset-dir",
        corpus,
        "--subset",
        "dev-clean",
        "--num-scenes",
        1,
        "--num-sources",
        2,
        "--duration",
        0.08,
        "--steps",
        3,
        "--order",
        0,
        "--tmax",
        0.04,
        "--device",
        "cpu",
        "--out-dir",
        output,
    )
    audio, meta = check_outputs(output, "scene_000", "dynamic_src", 640)
    assert meta["rir"]["sample_count"] == 320
    refs = [
        sf.read(output / item["filename"], always_2d=True)[0]
        for item in meta["extra"]["reference_audio"]
    ]
    np.testing.assert_allclose(audio, sum(refs), rtol=1e-5, atol=1e-7)


def test_builder_cli_generates_a_complete_local_scene(tmp_path, corpus):
    output = tmp_path / "dataset"
    run_python(
        tmp_path,
        "-m",
        "torchrir.datasets.dynamic_cmu_arctic",
        "--cmu-root",
        corpus,
        "--no-download-cmu",
        "--dataset-root",
        output,
        "--speakers",
        "bdl",
        "slt",
        "clb",
        "--n-scenes",
        1,
        "--n-sources",
        2,
        "--n-moving-sources",
        1,
        "--duration-sec",
        0.08,
        "--trajectory-steps",
        3,
        "--rir-samples",
        256,
        "--max-order",
        0,
        "--device",
        "cpu",
        "--no-save-layout-images",
        "--no-save-layout-mp4",
    )
    scene = output / "scene_0000"
    meta = json.loads((scene / "metadata.json").read_text())
    audio, rate = sf.read(scene / "mixture.wav", always_2d=True)
    assert rate == 8000
    assert audio.shape == (895, 6)
    refs = [
        sf.read(scene / f"source_{index:02d}.wav", always_2d=True)[0]
        for index in range(2)
    ]
    np.testing.assert_allclose(audio, sum(refs), rtol=1e-5, atol=1e-7)
    assert meta["signal"]["sample_count"] == 640
    assert meta["convolution"] == {
        "time_reference": "emission",
        "output_sample_count": 895,
    }
    assert meta["frame_schedule"]["starts_samples"] == [0, 213, 426]
    assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 0
