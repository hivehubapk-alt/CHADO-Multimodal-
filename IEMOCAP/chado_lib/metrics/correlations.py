import numpy as np
from scipy.stats import pearsonr, spearmanr

def pearson(x, y):
    r, p = pearsonr(x, y)
    return float(r), float(p)

def spearman(x, y):
    r, p = spearmanr(x, y)
    return float(r), float(p)
