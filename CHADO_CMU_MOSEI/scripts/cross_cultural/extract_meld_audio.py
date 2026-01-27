import os
import argparse
import numpy as np
import torch

from scripts.cross_cultural.meld_io import load_meld_split_df, find_media_file


def wav_to_logmel_74(wav_path: str, sr: int = 16000, n_mels: int = 74) -> torch.Tensor:
    """
    Returns [T, 74] log-mel. Then caller will resample T->400.
    """
    import librosa

    y, _sr = librosa.load(wav_path, sr=sr, mono=True)
    if y is None or len(y) == 0:
        return torch.zeros(1, n_mels)

    hop_length = int(0.010 * sr)  # 10ms
    win_length = int(0.025 * sr)  # 25ms
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=win_length, hop_length=hop_length, power=2.0
    )
    logS = librosa.power_to_db(S + 1e-10)
    feat = torch.from_numpy(logS.T).float()  # [T, 74]
    return feat


def resample_time(x: torch.Tensor, T: int = 400) -> torch.Tensor:
    """
    x: [t, d] -> [T, d] by linear interpolation
    """
    if x.ndim != 2:
        raise ValueError(f"Expected [t,d], got {x.shape}")
    t, d = x.shape
    if t == T:
        return x
    x_ = x.T.unsqueeze(0)  # [1,d,t]
    y_ = torch.nn.functional.interpolate(x_, size=T, mode="linear", align_corners=False)
    return y_.squeeze(0).T.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meld_root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "dev", "test", "val"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_items", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_meld_split_df(args.meld_root, args.split)
    if args.max_items > 0:
        df = df.head(args.max_items)

    kept = 0
    missing = 0
    for _, r in df.iterrows():
        utt_id = r["utt_id"]
        wav = find_media_file(args.meld_root, args.split, utt_id, kind="audio")
        if wav is None:
            missing += 1
            continue

        feat = wav_to_logmel_74(wav)       # [t,74]
        feat = resample_time(feat, 400)    # [400,74]
        out_path = os.path.join(args.out_dir, f"{utt_id}.pt")
        torch.save(feat, out_path)
        kept += 1

    print(f"[AUDIO] split={args.split} saved={kept} missing_wav={missing} out={args.out_dir}")


if __name__ == "__main__":
    main()
