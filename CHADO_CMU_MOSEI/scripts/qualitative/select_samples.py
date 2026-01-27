# scripts/qualitative/select_samples.py
import torch
import pandas as pd

K_TRUE = 5
K_WRONG = 5

probs = torch.load("outputs/cache/test_probs.pt")
labels = torch.load("outputs/cache/test_labels.pt")
mad = torch.load("outputs/cache/test_mad.pt")

pred = (probs > 0.5).int()
true = (labels > 0.5).int()

correct = (pred == true).all(dim=1)
wrong = ~correct

# Prefer high ambiguity
wrong_idx = torch.where(wrong)[0]
wrong_idx = wrong_idx[torch.argsort(mad[wrong_idx], descending=True)[:K_WRONG]]

correct_idx = torch.where(correct)[0]
correct_idx = correct_idx[torch.argsort(mad[correct_idx], descending=True)[:K_TRUE]]

torch.save({
    "correct": correct_idx,
    "wrong": wrong_idx
}, "outputs/analysis/qualitative/selected_idx.pt")
