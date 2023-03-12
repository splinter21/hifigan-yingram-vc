import time

import librosa
import matplotlib.pyplot as plt
import torch

from yingram import calc_yingram


def plot_yingram(path ,shift):
    sp_gram = calc_yingram(path, shift).transpose(0,1)[:, :400]
    print(sp_gram)

    plt.imshow(sp_gram,origin='lower')
    plt.show()


plot_yingram("/Volumes/Extend/下载/asdasd.wav", 0)
