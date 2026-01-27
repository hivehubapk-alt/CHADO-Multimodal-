import os
import argparse
import numpy as np
import torch

from scripts.cross_cultural.meld_io import load_meld_split_df, find_media_file


def _get_proj(path: str, in_dim: int = 512, out_dim: int = 35, seed: int = 42) -> torch.Tensor:
    """
    Fixed random projection so results are deterministic across runs.
    """
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    g = torch.Generator()
    g.manual_seed(seed)
    W = torch.randn(in_dim, out_dim, generator=g) / (in_dim ** 0.5)
    torch.save(W, path)
    return W


@torch.no_grad()
def video_to_resnet_proj(video_path: str, proj_W: torch.Tensor, max_frames: int = 32) -> torch.Tensor:
    """
    Extract frame embeddings using ResNet18 -> project to 35 dims.
    Returns [t,35] then resampled to 400 by caller.
    """
    import cv2
    import torchvision
    import torchvision.transforms as T
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # backbone
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    m.fc = torch.nn.Identity()
    m.eval().to(device)

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return torch.zeros(1, proj_W.shape[1])

    # sample uniformly up to max_frames
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return torch.zeros(1, proj_W.shape[1])

    idxs = np.linspace(0, n - 1, num=min(max_frames, n), dtype=int)

    feats = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        x = tfm(img).unsqueeze(0).to(device)  # [1,3,224,224]
        z = m(x).squeeze(0).detach().cpu()     # [512]
        feats.append(z)

    cap.release()
    if not feats:
        return torch.zeros(1, proj_W.shape[1])

    Z = torch.stack(feats, dim=0)             # [t,512]
    X = Z @ proj_W                            # [t,35]
    return X.float()


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
    ap.add_argument("--proj_cache", default="outputs/cache/meld_video_proj.pt")
    ap.add_argument("--max_items", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.proj_cache), exist_ok=True)

    proj_W = _get_proj(args.proj_cache, in_dim=512, out_dim=35, seed=42)  # [512,35]

    df = load_meld_split_df(args.meld_root, args.split)
    if args.max_items > 0:
        df = df.head(args.max_items)

    kept = 0
    missing = 0
    for _, r in df.iterrows():
        utt_id = r["utt_id"]
        mp4 = find_media_file(args.meld_root, args.split, utt_id, kind="video")
        if mp4 is None:
            missing += 1
            continue

        feat = video_to_resnet_proj(mp4, proj_W)  # [t,35]
        feat = resample_time(feat, 400)           # [400,35]
        out_path = os.path.join(args.out_dir, f"{utt_id}.pt")
        torch.save(feat, out_path)
        kept += 1

    print(f"[VIDEO] split={args.split} saved={kept} missing_mp4={missing} out={args.out_dir}")
    print(f"[VIDEO] proj_cache={args.proj_cache}")


if __name__ == "__main__":
    main()
