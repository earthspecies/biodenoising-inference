"""Synthetic CPU audio round trip, without a download or microphone access.

Run this script after installing the package and its runtime dependencies.
It exercises an untrained miniature model, not pretrained denoising quality.
"""

import importlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

# Regression for #1: unused imports must not require these extra dependencies.
sys.modules.update(librosa=None, norbert=None)

import torch
import torchaudio

from biodenoising.denoiser.audio import get_info
from biodenoising.denoiser.demucs import Demucs


def main():
    denoise = importlib.import_module("biodenoising.denoiser.denoise")
    torch.manual_seed(0)
    torch.set_num_threads(1)
    sample_rate = 16_000
    signal = 0.1 * torch.sin(2 * torch.pi * 440 * torch.arange(3200) / sample_rate)
    model = Demucs(hidden=4, depth=2, resample=1, sample_rate=sample_rate).eval()

    with TemporaryDirectory() as directory:
        directory = Path(directory)
        input_directory = directory / "input"
        input_directory.mkdir()
        input_path = input_directory / "synthetic.wav"
        torchaudio.save(str(input_path), signal.unsqueeze(0), sample_rate)
        info = get_info(str(input_path))
        assert (info.length, info.sample_rate, info.channels) == (3200, sample_rate, 1)
        output_directory = directory / "output"
        args = denoise.parser.parse_args([
            "--input", str(input_directory), "--output", str(output_directory),
            "--num_workers", "1",
        ])
        # Match the CLI's per-directory initialization, but supply our local
        # untrained model so no checkpoint is downloaded or deserialized.
        args.noisy_dir = args.input
        args.out_dir = args.output
        denoise.args = args
        denoise.denoise(args, model=model)
        output, output_rate = torchaudio.load(str(output_directory / "synthetic.wav"))
        assert output_rate == sample_rate
        assert output.shape == (1, 3200)
        assert torch.isfinite(output).all()
        assert output.abs().max() > 0
    print(f"Synthetic offline round trip passed: torch {torch.__version__}, "
          f"torchaudio {torchaudio.__version__}, 3200 mono samples at {sample_rate} Hz")


if __name__ == "__main__":
    main()
