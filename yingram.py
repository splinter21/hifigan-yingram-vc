from typing import Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def m2l(sr: float, m: float) -> float:
    """Midi-to-lag converter.
    Args:
        sr: sample rate.
        m: midi-scale.
    Returns;
        corresponding time-lag.
    """
    return sr / (440 * 2 ** ((m - 69) / 12))


def l2m(sr: float, l: float) -> float:
    """Lag-to-midi converter.
    Args:
        sr: sample rate.
        l: time-lag.
    Returns:
        corresponding midi-scale value.
    """
    return 12 * np.log2(sr / (440 * l)) + 69


class Yingram(nn.Module):
    """Yingram, Midi-scale cumulative mean-normalized difference.
    """
    def __init__(self,
                 strides: int,
                 windows: int,
                 lmin: int,
                 lmax: int,
                 bins: int,
                 sr: int = 16000):
        """Initializer.
        Args:
            strides: the number of the frames between adjacent windows.
            windows: width of the window.
            lmin, lmax: bounds of time-lag,
                it could be `sr / fmax` or `sr / fmin`.
            bins: the number of the bins per semitone.
            sr: sample rate, default 16khz.
        """
        super().__init__()
        self.strides = strides
        self.windows = windows
        self.lmin, self.lmax = lmin, lmax
        self.bins = bins
        self.sr = sr
        # midi range
        self.mmin, self.mmax = Yingram.midi_range(sr, lmin, lmax)

    @staticmethod
    def midi_range(sr: int, lmin: int, lmax: int) -> Tuple[int, int]:
        """Convert time-lag range to midi-range.
        Args:
            sr: sample rate.
            lmin, lmax: bounds of time-lag.
        Returns:
            bounds of midi-scale range, closed interval.
        """
        return int(np.ceil(l2m(sr, lmax))), int(l2m(sr, lmin))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute the yingram from audio signal.
        Args:
            audio: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
            audiolen: [torch.long; [B]], length of signals.
        Returns:
            [torch.float32; [B, T / `strides`, `bins` x (M - m + 1)]], yingram,
                where M = l2m(`lmin`), m = l2m(`max`)
                      l2m(l) = 12 x log2(`sr` / (440 * l)) + 69
        """
        # d[tau]
        # = sum_{j=1}^{W-tau} (x[j] - x[j + tau])^2
        # = sum_{j=1}^{W-tau} (x[j] ** 2 - 2x[j]x[j + tau] + x[j + tau] ** 2)
        # = c[W - tau] - 2 * a[tau] + (c[W] - c[tau])
        #     where c[k] = sum_{j=1}^k (x[j] ** 2)
        #           a[tau] = sum_{j=1}^W x[j]x[j + tau]

        # alias
        w, tau_max = self.windows, self.lmax
        # [B, T / strides, windows]
        frames = F.pad(audio, [0, w]).unfold(-1, w, self.strides)
        # [B, T / strides, windows + 1]
        fft = torch.fft.rfft(frames, w * 2, dim=-1)
        # [B, T / strides, windows], symmetric
        corr = torch.fft.irfft(fft.abs().square(), dim=-1)
        # [B, T / strides, windows + 1]
        cumsum = F.pad(frames.square().cumsum(dim=-1), [1, 0])
        # [B, T / strides, lmax], difference function
        diff = (
            # c[W - tau]
            torch.flip(cumsum[..., w - tau_max + 1:w + 1], dims=[-1])
            # - 2 * a[tau]
            - 2 * corr[..., :tau_max]
            # + (c[W] - c[tau]])
            + cumsum[..., w, None] - cumsum[..., :tau_max])
        # [B, T / strides, lmax - 1]
        cumdiff = diff[..., 1:] / (diff[..., 1:].cumsum(dim=-1) + 1e-7)
        ## in NANSY, Eq(1), it does not normalize the cumulative sum with lag size
        ## , but in YIN, Eq(8), it normalize the sum with their lags
        cumdiff = cumdiff * torch.arange(1, tau_max, device=cumdiff.device)
        # [B, T / strides, lmax], cumulative mean normalized difference
        cumdiff = F.pad(cumdiff, [1, 0], value=1.)
        # [bins x (mmax - mmin + 1)]
        steps = self.bins ** -1
        lags = m2l(
            self.sr,
            torch.arange(self.mmin, self.mmax + 1, step=steps, device=cumdiff.device))
        lceil, lfloor = lags.ceil().long(), lags.floor().long()
        # [B, T / strides, bins x (mmax - mmin + 1)], yingram
        return (
            (cumdiff[..., lceil] - cumdiff[..., lfloor]) * (lags - lfloor)
            / (lceil - lfloor) + cumdiff[..., lfloor])


# yingram
yin_strides = 640  # targeting 50hz on 22050hz sr
yin_windows = 2048
yin_lmin = 22  # 1000.40hz, 83midi(floor)
yin_lmax = 2047  # 10.77hz,  5midi(ceil), 79-channel yingram

yin_hmin = 25.11  # 878 lag, 20midi, +15 from lmin
yin_hmax = 430.19  # 51 lag, 68midi, 49-channel

yin_bins = 2  # the number of the bins per semitone
sr = 32000

yingram = Yingram(
    yin_strides,
    yin_windows,
    yin_lmin,
    yin_lmax,
    yin_bins,
    sr)

def yingram_channels():
    # midi-range
    mmin, mmax = Yingram.midi_range(sr, yin_lmin, yin_lmax)
    # hertz to lag
    lmin, lmax = int(sr / yin_hmax), int(sr / yin_hmin)
    # for feeding synthesizer
    # WARNING: since integer quantization in `midi_range`
    # , it could occur the errors
    #  ex. 984-channels Yingram in paper, 980 in this repo
    s_mmin, s_mmax = Yingram.midi_range(sr, lmin, lmax)
    # for sampling
    bins = yin_bins
    delta, range_ = (s_mmin - mmin) * bins, (s_mmax - s_mmin + 1) * bins
    return delta, range_, {
        'yin-midi-min': mmin, 'yin-midi-max': mmax,
        'syn-midi-min': s_mmin, 'syn-midi-max': s_mmax}

yin_delta, yin_range, _ = yingram_channels()

def sample_yingram(
                   yingram: torch.Tensor,
                   start=None,
                   semitone: bool = False) \
        -> torch.Tensor:
    """Sample the yingram for synthesizer.
    Args:
        yingram: [torch.float32; [B, Y, T // yin_strides]], yingram,
            where Y = yin_bins x (mmax - mmin + 1)
        start: [torch.long; [B]], start position.
        semitone: treat the `start` as relative semitone steps
            otherwise absolute start index.
    Returns:
        [torch.float32; [B, Y', T // yin_strides]], sampled yingram,
            where Y' = yin_bins x (lmax - lmin + 1)
    """
    d, r = yin_delta, yin_range
    if start is None:
        return yingram[:, d:d + r]
    # semitone conversion
    if semitone:
        start = d + start * yin_bins
    # sampling
    if isinstance(start, int):
        return yingram[:, start:start + r]
    return torch.stack([y[s:s + r] for y, s in zip(yingram, start)], dim=0)


def calc_yingram(path, shift=0):
    wav, _ = librosa.load(path, sr=sr)
    wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
    x = yingram.forward(wav_tensor).transpose(1, 2)
    sp_gram = sample_yingram(x, torch.LongTensor([-shift]), True).squeeze(0)
    return sp_gram.transpose(0,1)


# print(calc_yingram("dataset_raw/kkk.wav").shape)
