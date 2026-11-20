"""CPU-only pretrained-state regression checks; no downloads or audio devices.

Run ``python -m unittest discover -s tests -p test_pretrained.py -v`` after
installing the package's runtime dependencies.
"""

from collections import OrderedDict
from copy import deepcopy
import unittest
from unittest import mock

import torch
from torch import nn

from biodenoising.denoiser import pretrained
from biodenoising.denoiser.demucs import Demucs
from biodenoising.denoiser.utils import load_model_state_dict, serialize_model


class StateDictTests(unittest.TestCase):
    def make_model(self):
        model = nn.Linear(3, 2)
        model.register_buffer("offset", torch.ones(2))
        with torch.no_grad():
            model.weight.fill_(1)
            model.bias.fill_(1)
        return model

    def test_supplied_nonzero_weights_and_buffers_replace_initial_state(self):
        model = self.make_model()
        state = OrderedDict((key, torch.full_like(value, index + 2))
                            for index, (key, value) in enumerate(model.state_dict().items()))
        original = deepcopy(state)

        self.assertIs(load_model_state_dict(model, state), model)

        for key, value in model.state_dict().items():
            with self.subTest(key=key):
                self.assertTrue(torch.equal(value, original[key]))
                self.assertTrue(torch.equal(state[key], original[key]))

    def test_all_zero_state_is_not_ignored(self):
        model = self.make_model()
        state = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
        load_model_state_dict(model, state)
        for key, value in model.state_dict().items():
            with self.subTest(key=key):
                self.assertEqual(torch.count_nonzero(value).item(), 0)

    def test_partial_state_keeps_missing_values_and_ignores_unexpected_keys(self):
        model = self.make_model()
        state = {"weight": torch.full_like(model.weight, 5), "unused": torch.ones(1)}
        load_model_state_dict(model, state)
        self.assertTrue(torch.equal(model.weight, state["weight"]))
        self.assertTrue(torch.equal(model.bias, torch.ones_like(model.bias)))
        self.assertTrue(torch.equal(model.offset, torch.ones_like(model.offset)))

    def test_incompatible_shapes_raise_instead_of_being_filtered(self):
        model = self.make_model()
        with self.assertRaisesRegex(RuntimeError, "size mismatch for weight"):
            load_model_state_dict(model, {"weight": torch.ones(1, 1)})

    def test_invalid_parameter_values_raise(self):
        model = self.make_model()
        with self.assertRaisesRegex(RuntimeError, "weight"):
            load_model_state_dict(model, {"weight": "not a tensor"})

    def test_state_dict_metadata_is_forwarded_unchanged(self):
        model = self.make_model()
        state = model.state_dict()
        with mock.patch.object(model, "load_state_dict", wraps=model.load_state_dict) as load:
            load_model_state_dict(model, state)
        self.assertIs(load.call_args.args[0], state)
        self.assertEqual(load.call_args.kwargs, {"strict": False})


class PretrainedTests(unittest.TestCase):
    def make_model(self):
        return Demucs(hidden=4, depth=2, resample=1, sample_rate=16000)

    def assert_state_equal(self, model, state):
        self.assertEqual(set(model.state_dict()), set(state))
        for key, value in model.state_dict().items():
            with self.subTest(key=key):
                self.assertTrue(torch.equal(value, state[key]))

    def test_dns48_applies_raw_downloaded_weights(self):
        target = self.make_model()
        state = {key: torch.full_like(value, 0.125)
                 for key, value in target.state_dict().items()}
        with mock.patch.object(pretrained, "Demucs", return_value=target) as constructor, \
                mock.patch.object(pretrained.torch.hub, "load_state_dict_from_url",
                                  return_value=state) as download:
            model = pretrained.dns48()
        self.assertIs(model, target)
        constructor.assert_called_once_with(hidden=48, sample_rate=16000)
        download.assert_called_once_with(pretrained.DNS_48_URL, map_location="cpu")
        self.assert_state_equal(model, state)

    def test_dns48_applies_published_nested_package_state(self):
        source = self.make_model()
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(0.25)
        package = {"model": serialize_model(source)}
        # The published package includes old constructor options. Loading into
        # the selected DNS48 architecture must not invoke checkpoint classes.
        package["model"]["class"] = mock.Mock(side_effect=AssertionError("not a constructor"))
        package["model"]["kwargs"]["noisereduce"] = False
        original_kwargs = deepcopy(package["model"]["kwargs"])
        original_state = deepcopy(package["model"]["state"])
        original_keys = tuple(package["model"])
        target = self.make_model()
        with mock.patch.object(pretrained, "Demucs", return_value=target), \
                mock.patch.object(pretrained.torch.hub, "load_state_dict_from_url",
                                  return_value=package):
            model = pretrained.dns48()
        self.assert_state_equal(model, original_state)
        package["model"]["class"].assert_not_called()
        self.assertEqual(package["model"]["kwargs"], original_kwargs)
        self.assertEqual(tuple(package["model"]), original_keys)
        self.assertEqual(set(package), {"model"})
        for key, value in package["model"]["state"].items():
            with self.subTest(checkpoint_key=key):
                self.assertTrue(torch.equal(value, original_state[key]))

    def test_malformed_nested_package_does_not_silently_keep_random_weights(self):
        with mock.patch.object(pretrained, "Demucs", return_value=self.make_model()), \
                mock.patch.object(pretrained.torch.hub, "load_state_dict_from_url",
                                  return_value={"model": {}}):
            with self.assertRaises(KeyError):
                pretrained.dns48()

    def test_downloaded_incompatible_shape_is_reported(self):
        target = self.make_model()
        state = target.state_dict()
        state["encoder.0.0.weight"] = torch.ones(1, 1, 1)
        with mock.patch.object(pretrained, "Demucs", return_value=target), \
                mock.patch.object(pretrained.torch.hub, "load_state_dict_from_url",
                                  return_value={"model": {"state": state}}):
            with self.assertRaisesRegex(RuntimeError, "size mismatch for encoder.0.0.weight"):
                pretrained.dns48()

    def test_dns48_rejects_incomplete_or_unexpected_pretrained_state(self):
        target = self.make_model()
        state = target.state_dict()
        first = next(iter(state))
        extra = {"not_a_dns48_key": torch.ones(1)}
        cases = {
            "empty": {},
            "missing": {key: value for key, value in state.items() if key != first},
            "unrelated": extra,
            "unexpected": {**state, **extra},
        }
        for name, bad_state in cases.items():
            for wrapped in (False, True):
                payload = {"model": {"state": bad_state}} if wrapped else bad_state
                with self.subTest(case=name, wrapped=wrapped), \
                        mock.patch.object(pretrained, "Demucs", return_value=target), \
                        mock.patch.object(pretrained.torch.hub, "load_state_dict_from_url",
                                          return_value=payload):
                    with self.assertRaisesRegex(RuntimeError, r"key\(s\) in state_dict"):
                        pretrained.dns48()

    def test_pretrained_false_does_not_download(self):
        target = self.make_model()
        original = deepcopy(target.state_dict())
        with mock.patch.object(pretrained, "Demucs", return_value=target), \
                mock.patch.object(pretrained.torch.hub, "load_state_dict_from_url") as download:
            model = pretrained.dns48(pretrained=False)
        download.assert_not_called()
        self.assertIs(model, target)
        self.assert_state_equal(model, original)


if __name__ == "__main__":
    unittest.main()
