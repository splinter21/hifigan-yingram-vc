import os

import librosa

allwavs = []
allfeats = []

for wav in os.listdir("dataset/32k"):
    if wav.endswith("wav"):
        filepath = f"dataset/32k/{wav}"
        y, sr = librosa.load(filepath, None)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration>1:
            allwavs.append(wav)
            allfeats.append(wav.replace(".wav", ".pt"))

train_wavs = allwavs[:-3]
train_feats = allfeats[:-3]
val_wavs = allwavs[-3:]
val_feats = allfeats[-3:]

with open("data_splits/wavlm-hifigan-train.csv", "w") as f:
    f.write("audio_path,feat_path\n")
    for wav, feat in zip(train_wavs, train_feats):
        f.write(f"{wav},{feat}\n")

with open("data_splits/wavlm-hifigan-valid.csv", "w") as f:
    f.write("audio_path,feat_path\n")
    for wav, feat in zip(val_wavs, val_feats):
        f.write(f"{wav},{feat}\n")


