import librosa

import yingram

dependencies = ['torch', 'torchaudio', 'numpy']

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import logging
import json
from pathlib import Path

from wavlm.WavLM import WavLM, WavLMConfig
from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from matcher import KNN_VC


def get_hubert_model():
    vec_path = "hubert/checkpoint_best_legacy_500.pt"
    print("load model(s) from {}".format(vec_path))
    from fairseq import checkpoint_utils
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [vec_path],
        suffix="",
    )
    model = models[0]
    model.eval()
    return model


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
def get_full_features(path, hmodel, device, shift):

    x, sr = librosa.load(path, sr=16000)
    x = torch.FloatTensor(x).to(device)

    assert sr == 16000
    n_pad = 320 - (x.shape[-1] % 320)
    x = F.pad(x, (0, n_pad), value=0)
    features = get_hubert_content(hmodel, x).transpose(1, 2).squeeze(0)
    out_feats = features.cpu()

    gram = yingram.calc_yingram(path, shift=shift)
    assert abs(out_feats.shape[0] - gram.shape[0]) < 3
    l = min(out_feats.shape[0], gram.shape[0])
    out_feats = out_feats[:l, :]
    gram = gram[:l, :]
    out_feats = torch.cat((out_feats, gram), dim=1)
    return out_feats


def knn_vc(pretrained=True, progress=True, prematched=True, device='cuda', model_path=None) -> KNN_VC:
    """ Load kNN-VC (WavLM encoder and HiFiGAN decoder). Optionally use vocoder trained on `prematched` data. """
    hifigan, hifigan_cfg = hifigan_wavlm(pretrained, progress, prematched, device, model_path)
    wavlm = wavlm_large(pretrained, progress, device)
    knnvc = KNN_VC(wavlm, hifigan, hifigan_cfg, device)
    return knnvc


def hifigan_wavlm(pretrained=True, progress=True, prematched=True, device='cuda', model_path=None) -> HiFiGAN:
    """ Load pretrained hifigan trained to vocode wavlm features. Optionally use weights trained on `prematched` data. """
    cp = Path(__file__).parent.absolute()

    with open(cp / 'hifigan' / 'config_v1_wavlm.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    device = torch.device(device)

    generator = HiFiGAN(h).to(device)

    state_dict_g = torch.load(model_path,device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    print(f"[HiFiGAN] Generator loaded with {sum([p.numel() for p in generator.parameters()]):,d} parameters.")
    return generator, h


def wavlm_large(pretrained=True, progress=True, device='cuda') -> WavLM:
    """Load the WavLM large checkpoint from the original paper. See https://github.com/microsoft/unilm/tree/master/wavlm for details. """
    if torch.cuda.is_available() == False:
        if str(device) != 'cpu':
            logging.warning(f"Overriding device {device} to cpu since no GPU is available.")
            device = 'cpu'
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/datasets/innnky/ft_vispeech/resolve/main/WavLM-Large.pt",
        map_location=device,
        progress=progress
    )

    cfg = WavLMConfig(checkpoint['cfg'])
    device = torch.device(device)
    model = WavLM(cfg)
    if pretrained:
        model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print(f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")
    return model
