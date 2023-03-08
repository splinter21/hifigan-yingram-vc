import argparse
import gc
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import tqdm
from fastprogress.fastprogress import master_bar, progress_bar
from torch import Tensor

import hubconf
import yingram
from hubconf import wavlm_large

DOWNSAMPLE_FACTOR = 320

feature_cache = {}
synthesis_cache = {}

def make_librispeech_df(root_path: Path) -> pd.DataFrame:
    all_files = []
    all_files.extend(list((root_path).rglob('*.wav')))
    speakers = [None for f in all_files]
    df = pd.DataFrame({'path': all_files, 'speaker': speakers})
    return df


def main(args):
    device = torch.device(args.device)
    SYNTH_WEIGHTINGS = F.one_hot(torch.tensor(args.synthesis_layer), num_classes=25).float().to(device)[:, None]
    MATCH_WEIGHTINGS = F.one_hot(torch.tensor(args.matching_layer), num_classes=25).float().to(device)[:, None]

    print(f"Matching weightings: {MATCH_WEIGHTINGS.squeeze()}\nSynthesis weightings: {SYNTH_WEIGHTINGS.squeeze()}")
    ls_df = make_librispeech_df(Path(args.librispeech_path))

    print(f"Loading wavlm.")
    wavlm = hubconf.get_hubert_model().to(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    extract(ls_df, wavlm, args.device, Path(args.librispeech_path), Path(args.out_path), SYNTH_WEIGHTINGS, MATCH_WEIGHTINGS)
    print("All done!", flush=True)


def get_hubert_content(hmodel, wav_16k_tensor):
    feats = wav_16k_tensor
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    feats = feats.view(1, -1)
    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
    inputs = {
        "source": feats.to(wav_16k_tensor.device),
        "padding_mask": padding_mask.to(wav_16k_tensor.device),
        "output_layer": 9,  # layer 9
    }
    with torch.no_grad():
        logits = hmodel.extract_features(**inputs)
        feats = hmodel.final_proj(logits[0])
    return feats.transpose(1, 2)

@torch.inference_mode()
def get_full_features(path, hmodel, device):

    x, sr = librosa.load(path.parent.parent/ "16k" / path.name, sr=None)
    x = torch.FloatTensor(x).to(device)

    assert sr == 16000
    n_pad = DOWNSAMPLE_FACTOR - (x.shape[-1] % DOWNSAMPLE_FACTOR)
    x = F.pad(x, (0, n_pad), value=0)
    features = get_hubert_content(hmodel, x).transpose(1, 2)

    return features.squeeze(0)


def fast_cosine_dist(source_feats, matching_pool):
    source_norms = torch.norm(source_feats, p=2, dim=-1)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -torch.cdist(source_feats[None], matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
    dotprod /= 2

    dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
    return dists


@torch.inference_mode()
def extract(df: pd.DataFrame, wavlm: nn.Module, device, ls_path: Path, out_path: Path, synth_weights: Tensor, match_weights: Tensor):
    
    pb = progress_bar(df.iterrows(), total=len(df))
    # print(pb.wait_for)
    for i, row in pb:
        rel_path = Path(row.path).relative_to(ls_path)
        targ_path = (out_path/rel_path).with_suffix('.pt')
        if args.resume:
            if targ_path.is_file(): continue
        # if targ_path.is_file(): continue
        os.makedirs(targ_path.parent, exist_ok=True)

        source_feats = get_full_features(row.path, wavlm, device)

        out_feats = source_feats.cpu()

        gram = yingram.calc_yingram(row.path)
        assert abs(out_feats.shape[0]- gram.shape[0])<3
        l = min(out_feats.shape[0], gram.shape[0])
        out_feats = out_feats[:l, :]
        gram = gram[:l, :]
        out_feats = torch.cat((out_feats, gram), dim=1)
        # if not args.prematch:
        # else:
        #     matching_pool, synth_pool = path2pools(row.path, wavlm, match_weights, synth_weights, device)
        #     dists = fast_cosine_dist(source_feats.cpu(), matching_pool.cpu()).cpu()
        #     best = dists.topk(k=args.topk, dim=-1, largest=False) # (src_len, 4)
        #     out_feats = synth_pool[best.indices].mean(dim=1) # (N, dim)

        # save matched sequence
        if i < 3: print("Feature has shape: ", out_feats.shape, flush=True)
        # 3. save
        torch.save(out_feats.cpu().half(), str(targ_path))
        # if hasattr(pb, 'child'):
        #     pb.child.comment = str(rel_path)
        #     pb.child.wait_for = min(pb.child.wait_for, 10)
        #     pb.main_bar.comment = str(rel_path)
        # else:
        #     pb.wait_for = min(pb.wait_for, 10)
        # pb.comment = str(rel_path)
        #

        if i % 1000 == 0: 
            print(f"Done {i:,d}/{len(df):,d}", flush=True)
            # gc.collect()
            # time.sleep(4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute matched wavlm features for a librispeech dataset")

    parser.add_argument('--librispeech_path', default="dataset/32k", type=str)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--out_path', default="dataset/features", type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--topk', type=int, default=4)
    parser.add_argument('--matching_layer', type=int, default=6)
    parser.add_argument('--synthesis_layer', type=int, default=6)
    parser.add_argument('--prematch', action='store_true', help='prematch')
    parser.add_argument('--resume', default=True)

    args = parser.parse_args()
    main(args)

