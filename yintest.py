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






# plot_yingram("/Users/xingyijin/Documents/individualAudio.wav", 0)
# plot_yingram("kkk.wav", 0)
# plot_yingram("kkk.wav", 9)
# plot_yingram("testwav.wav", 0)
plot_yingram("/Volumes/Extend/下载/yintest.flac", -15)
plot_yingram("/Volumes/Extend/下载/yintest.flac", 0)
# plot_yingram("/Volumes/Extend/下载/yintest-11.flac", 0)
# plot_yingram("/Volumes/Extend/下载/yintest-11.flac", -11)

# sp_gram = sample_yingram(x, torch.LongTensor([-12]), True).squeeze(0)
#
# plt.imshow(sp_gram,origin='lower')
# plt.show()
#
# sp_gram = sample_yingram(x, torch.LongTensor([12]), True).squeeze(0)
#
# plt.imshow(sp_gram,origin='lower')
# plt.show()
