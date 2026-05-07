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
        data = f.read()
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

def oneHot2Char(one_hot, ind2char):
    """convert one hot matrix (kxt) to a string"""
    indices = np.argmax(one_hot, axis=0)
    return ''.join(ind2char[i] for i in indices)


if __name__ == "__main__":
    book_data = loadbook(data_dir / "goblet_book.txt")
    unique_chars, K, char_to_ind, ind_to_char = VocabBuilder(book_data)

    print(f"Book length : {len(book_data):,} characters")
    print(f"Unique chars: {K}")
    print(f"First 50 chars : {book_data[:50]!r}")
    print(f"Sample mapping: 'H' to {char_to_ind['H']}, {char_to_ind['H']} to '{ind_to_char[char_to_ind['H']]}'")

    test_str = "Harry"
    X_test = Char2oneHot(test_str, char_to_ind, K)
    recovered = oneHot2Char(X_test, ind_to_char)
    assert recovered == test_str, f"Round-trip failed: {recovered}"
    print(f"One-hot round-trip OK: '{test_str}' to matrix({X_test.shape}) to '{recovered}'")