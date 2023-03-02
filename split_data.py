import os


allwavs = []
allfeats = []

for wav in os.listdir("dataset_raw"):
    if wav.endswith("wav"):
        allwavs.append(wav)
        allfeats.append(wav.replace("dataset_raw/", "dataset/").replace("wav", "pt"))

train_wavs = allwavs[:-3]
train_feats = allfeats[:-3]
val_wavs = allwavs[-3:]
val_feats = allfeats[-3:]

with open("data_splits/wavlm-hifigan-train.csv", "w") as f:
    f.write("audio_path,feat_path\n")
    for wav, feat in zip(train_wavs, train_feats):
        f.write(f"{wav}, {feat}\n")

with open("data_splits/wavlm-hifigan-valid.csv", "w") as f:
    f.write("audio_path,feat_path\n")
    for wav, feat in zip(val_wavs, val_feats):
        f.write(f"{wav}, {feat}\n")


