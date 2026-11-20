"""Validate the published DNS48 weights on CPU without network or audio devices.

Requires Torch >= 2.6 and the runtime dependencies. Download the official
``pretrained.DNS_48_URL`` separately, then run from an installed source checkout:
``python tests/pretrained_smoke.py /path/to/model-16kHz-dns48.th``.

The public Hub loader runs unchanged against a temporary cache. This test forces
weights-only deserialization of the SHA256-pinned checkpoint and temporarily
allowlists the reviewed local Demucs class. Production policy is unchanged.
Synthetic audio checks numerical loading and the WAV pipeline, not audio quality.
"""

import argparse
import hashlib
import importlib
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from unittest import mock

import torch
import torchaudio

from biodenoising.denoiser import pretrained
from biodenoising.denoiser.demucs import Demucs


PUBLISHED_SHA256 = "e1f0da9dcecbba5c7fe497daa6e2e35d9f55af895ff893d56ff93409ce7bd24a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    checkpoint = parser.parse_args().checkpoint
    torch.set_num_threads(1)
    model_seed = 0
    previous_allowlist = torch.serialization.get_safe_globals()
    denoise = importlib.import_module("biodenoising.denoiser.denoise")
    with TemporaryDirectory() as directory, torch.serialization.safe_globals([Demucs]):
        directory = Path(directory)
        cache = directory / "hub" / "checkpoints"
        cache.mkdir(parents=True)
        cached = cache / "model-16kHz-dns48.th"
        shutil.copyfile(checkpoint, cached)
        if hashlib.sha256(cached.read_bytes()).hexdigest() != PUBLISHED_SHA256:
            raise ValueError("Checkpoint does not match the inspected published DNS48 file")
        package = torch.load(cached, map_location="cpu", weights_only=True)
        state = package["model"]["state"]

        with mock.patch.dict(os.environ, {"TORCH_HOME": str(directory),
                                          "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1"}), \
                mock.patch.object(torch.hub, "download_url_to_file",
                                  side_effect=AssertionError("network is forbidden")):
            assert Path(torch.hub.get_dir()) == cache.parent
            torch.manual_seed(model_seed)
            loaded = pretrained.dns48().eval()
            assert set(loaded.state_dict()) == set(state)
            for key, value in loaded.state_dict().items():
                assert torch.equal(value, state[key]), key

            torch.manual_seed(model_seed)
            reference = Demucs(hidden=48, sample_rate=16000).eval()
            reference.load_state_dict(state, strict=True)
            torch.manual_seed(model_seed)
            untrained = pretrained.dns48(pretrained=False).eval()
            signal = (0.1 * torch.sin(2 * torch.pi * 440 * torch.arange(3200) / 16000)
                      + 0.01 * torch.randn(3200)).unsqueeze(0)
            with torch.no_grad():
                expected = reference(signal.unsqueeze(0))
                actual = loaded(signal.unsqueeze(0))
                random_output = untrained(signal.unsqueeze(0))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert not torch.equal(actual, random_output)
            assert torch.isfinite(actual).all()

            inputs = directory / "input"
            inputs.mkdir()
            input_path = inputs / "synthetic.wav"
            torchaudio.save(str(input_path), signal, 16000)
            saved_input, _ = torchaudio.load(str(input_path))
            with torch.no_grad():
                expected_saved = reference(saved_input.unsqueeze(0))[0]
            assert expected_saved.abs().max() < 1
            outputs = directory / "output"
            args = denoise.parser.parse_args([
                "--input", str(inputs), "--output", str(outputs), "--num_workers", "1",
            ])
            args.noisy_dir, args.out_dir = args.input, args.output
            denoise.args = args
            # No supplied model: get_model -> dns48 -> actual cached hub load.
            denoise.denoise(args)
            output, output_rate = torchaudio.load(str(outputs / "synthetic.wav"))
            assert output_rate == 16000
            assert output.shape == (1, 3200)
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output, expected_saved, rtol=0, atol=1 / 32768 + 1e-7)

    assert torch.serialization.get_safe_globals() == previous_allowlist
    print(f"Published DNS48 smoke passed: {len(state)}/{len(state)} state entries equal; "
          "forward matches an independently strict-loaded reference and differs from random; "
          "3200-sample 16 kHz WAV pipeline matches reference within PCM16 quantization; "
          f"torch {torch.__version__}, torchaudio {torchaudio.__version__}; no network.")


if __name__ == "__main__":
    main()
