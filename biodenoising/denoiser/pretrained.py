# Copyright (c) Earth Species Project. This work is based on Facebook's denoiser.

#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import os
import logging

import torch.hub

from .cleanunet import CleanUNet
from .demucs import Demucs
from .htdemucs import HTDemucs
from .utils import deserialize_model, load_model_state_dict
from .states import set_state

logger = logging.getLogger(__name__)
ROOT = "https://storage.googleapis.com/esp-public-files/biodenoising/"
DNS_48_URL = ROOT + "model-16kHz-dns48.th"


def _demucs(pretrained, url, **kwargs):
    model = Demucs(**kwargs, sample_rate=kwargs.get('sample_rate', 16_000))
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        model = load_model_state_dict(model, state_dict)
    return model


def dns48(pretrained=True):
    return _demucs(pretrained, DNS_48_URL, hidden=48)


def add_model_flags(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-m", "--model_path", help="Path to local trained model.")
    group.add_argument("--dns48", action="store_true",
                       help="Use pre-trained real time H=48 model trained on biodenosing-datasets.")


def get_model(args):
    """
    Load local model package or torchhub pre-trained model.
    """
    if args.model_path:
        logger.info("Loading model from %s", args.model_path)
        pkg = torch.load(args.model_path, 'cpu')
        if 'model' in pkg:
            # if 'best_state' in pkg:
            #     logger.info("Loading best model state.")
            #     pkg['model']['state'] = pkg['best_state']
            model = deserialize_model(pkg['model'])
        else:
            model = deserialize_model(pkg)
    else:
        logger.info("Loading pre-trained real time H=48 model trained on biodenosing-datasets.")
        model = dns48()
    logger.debug(model)
    return model
