import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data_dir  = ROOT / "Datasets"
fig_dir   = ROOT / "figures"
out_dir   = ROOT / "outputs"
fig_dir.mkdir(exist_ok=True)
out_dir.mkdir(exist_ok=True)

def loadbook(filepath):
    with open(filepath, 'r') as f:
        data = f.read().splitlines()
    return data

def VocabBuilder(data):
    """it'll return sorted unique character dictionary and the reverse of it"""
    unique_char = sorted(list(set(data)))
    k = len(unique_char)
    char2ind = {c:i for i, c in enumerate(unique_char)}
    ind2char = {i:c for i, c in enumerate(unique_char)}
    return unique_char, k, char2ind, ind2char

def Char2oneHot(chars, char2ind, k):
    """it'll return one-hot encoded vector of the given characters"""
    one_hot = np.zeros((k, len(chars)))
    for i, c in enumerate(chars):
        one_hot[char2ind[c], i] = 1.0
    return one_hot

