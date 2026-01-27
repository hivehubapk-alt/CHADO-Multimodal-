import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import soundfile as sf
import librosa

LABEL2ID = {"neu": 0, "hap": 1, "ang": 2, "sad": 3}

class IEMOCAPTriModal(Dataset):
    def __init__(self, csv_path, tokenizer, image_processor,
                 max_text_len=96, audio_sr=16000, audio_sec=4.0, n_frames=8):
        self.df = pd.read_csv(csv_path)
        self.tok = tokenizer
        self.img_proc = image_processor
        self.max_text_len = int(max_text_len)
        self.audio_sr = int(audio_sr)
        self.audio_sec = float(audio_sec)
        self.n_frames = int(n_frames)

    def __len__(self):
        return len(self.df)

    def _load_audio_segment(self, wav_path: str, start: float, end: float) -> np.ndarray:
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != self.audio_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_sr)
            sr = self.audio_sr

        s = int(max(0.0, float(start)) * sr)
        e = int(max(0.0, float(end)) * sr)
        if e <= s:
            e = min(len(audio), s + int(self.audio_sec * sr))

        seg = audio[s:e]
        target_len = int(self.audio_sec * sr)
        if len(seg) < target_len:
            seg = np.pad(seg, (0, target_len - len(seg)))
        else:
            seg = seg[:target_len]
        return seg.astype(np.float32)

    def _sample_video_frames_opencv(self, avi_path: str, start: float, end: float) -> np.ndarray:
        import cv2
        cap = cv2.VideoCapture(avi_path)
        if not cap.isOpened():
            return np.zeros((self.n_frames, 224, 224, 3), dtype=np.uint8)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 1e-6:
            fps = 30.0

        start_f = int(max(0.0, float(start)) * fps)
        end_f = int(max(0.0, float(end)) * fps)
        if end_f <= start_f:
            end_f = start_f + int(fps)

        idxs = np.linspace(start_f, end_f, num=self.n_frames, dtype=np.int64)

        frames = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
        cap.release()
        return np.stack(frames, axis=0)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        y = LABEL2ID[str(r["label_4"]).strip()]

        enc = self.tok(
            str(r["transcript"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        wav = self._load_audio_segment(r["wav_path"], r["start"], r["end"])
        wav = torch.from_numpy(wav)

        frames = self._sample_video_frames_opencv(r["avi_path"], r["start"], r["end"])
        img = self.img_proc(list(frames), return_tensors="pt")
        pixel_values = img["pixel_values"]  # [T,3,224,224]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "wav": wav,
            "pixel_values": pixel_values,
            "label": torch.tensor(y, dtype=torch.long),
        }
