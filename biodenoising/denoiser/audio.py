# Copyright (c) Earth Species Project. This work is based on Facebook's denoiser.

#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

from collections import namedtuple
import json
from pathlib import Path
import math
import os
import sys
import inspect
import random
import torch
import torchaudio
from torch.nn import functional as F

from .dsp import convert_audio

Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    return build_meta(audio_files, progress=progress)
    
def build_meta(audio_files, progress=True):
    meta = []
    for idx, file in enumerate(audio_files):
        info = get_info(file)
        meta.append((file, info.length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta

def repeat_and_pad(audio, duration_samples, repeat_prob=0.5, random_repeat=False, random_pad=False, random_obj=None):
    assert random_obj is not None, "random_obj should be provided"
    if random_obj.random() > repeat_prob:
        ### repeat
        if random_repeat:
            initial_length = audio.shape[-1]
            nrepeats = random_obj.randint(2,math.ceil(duration_samples/audio.shape[-1]))
            for i in range(nrepeats):
                gain = random_obj.uniform(0.1,1.3)
                audio[...,i*initial_length:(i+1)*initial_length] *= gain
        else:
            nrepeats = int(math.ceil(duration_samples/audio.shape[-1]))
            audio = audio.repeat(1,nrepeats)
        audio =  audio[...,:duration_samples]
    if audio.shape[-1] < duration_samples:
        if random_pad:
            audio_final = torch.zeros((audio.shape[0],duration_samples), device=audio.device)
            ### pad
            pad_left = random_obj.randint(0,duration_samples - audio.shape[-1])
            audio_final[...,pad_left:pad_left+audio.shape[-1]] = audio
            audio = audio_final
            pad_right = duration_samples - audio.shape[-1] - pad_left
        else:
            pad_left = int((duration_samples - audio.shape[-1])//2)
            pad_right = int(duration_samples - audio.shape[-1] - pad_left)
            audio = F.pad(audio, (pad_left, pad_right), mode='constant', value=0)
            #out = F.pad(out, (0, self.length - out.shape[-1]))
    return audio


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                pad=True, with_path=False, sample_rate=None,
                channels=None, convert=False, repeat_prob=0.5, 
                random_repeat=False, random_pad=False, use_subset=False, random_obj=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.subsets = {}
        self.num_examples_subsets = {}
        self.use_subset = use_subset
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.convert = convert
        self.repeat_prob = repeat_prob
        self.random_repeat = random_repeat
        self.random_pad = random_pad
        self.random_obj = random_obj if random_obj is not None else random.Random(0)
        for file, file_length in self.files:
            subset = os.path.basename(file).split('_')[0]
            if subset not in self.subsets.keys():
                self.subsets[subset] = []
                self.num_examples_subsets[subset] = []
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 #if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)
            # self.num_frames.append(file_length)
            self.num_examples_subsets[subset].append(examples)
            self.subsets[subset].append((file,file_length))
        if len(self.subsets.keys()) > 1:
            self.subset = list(self.subsets.keys())[0]

    def __len__(self):
        if self.use_subset:
            return sum(self.num_examples_subsets[self.subset])
        else:
            return sum(self.num_examples)
        
    def set_subset(self, subset_id):
        subsets = list(self.subsets.keys())
        subset = subsets[subset_id]
        self.subset = subset

    def __getitem__(self, index):
        if self.use_subset:
            files = self.subsets[self.subset]
            num_examples = self.num_examples_subsets[self.subset]
        else:
            files = self.files
            num_examples = self.num_examples
        for (file, _), examples in zip(files, num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
                # if examples==1:
                #     num_frames = self.num_frames[index]
                # else:
                #     num_frames = self.length
            params = inspect.getfullargspec(torchaudio.load)[0]
            if 'frame_offset' in params:
                out, sr = torchaudio.load(str(file),
                                        frame_offset=offset,
                                        num_frames=num_frames or -1)
            else:
                out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
            target_sr = self.sample_rate or sr
            target_channels = self.channels or out.shape[0]
            if self.convert:
                out = convert_audio(out, sr, target_sr, target_channels)
            else:
                if sr != target_sr:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                    f"{target_sr}, but got {sr}")
                if out.shape[0] != target_channels:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                    f"{target_channels}, but got {sr}")
            if num_frames and self.length > out.shape[-1]:
                out = repeat_and_pad(out, self.length, repeat_prob=self.repeat_prob, random_repeat=self.random_repeat, random_pad=self.random_pad, random_obj=self.random_obj)
            if self.with_path:
                return out, file
            else:
                return out


if __name__ == "__main__":
    meta = []
    for path in sys.argv[1:]:
        meta += find_audio_files(path)
    json.dump(meta, sys.stdout, indent=4)
