import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import librosa as li
import crepe
import math


def safe_log(x):
    return torch.log(x + 1e-7)

def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)
