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

def InitRNN(m, K, seed=42):
    """Initialize RNN parameters. m = hidden size, K = vocab size."""
    rng = np.random.default_rng(seed)
    RNN = {}
    RNN['U'] = (1/np.sqrt(2*K)) * rng.standard_normal((m, K))
    RNN['W'] = (1/np.sqrt(2*m)) * rng.standard_normal((m, m))
    RNN['V'] = (1/np.sqrt(m))   * rng.standard_normal((K, m))
    RNN['b'] = np.zeros((m, 1))
    RNN['c'] = np.zeros((K, 1))
    return RNN

def SynthesizeText(RNN, h0, x0, n, rng):
    """
    Generate n characters from the RNN. h0: (m×1) initial hidden state, x0: (K×1) one-hot seed character. Returns synthesized string.
    """
    h = h0.copy()
    x = x0.copy()
    indices = []

    for _ in range(n):
        a = RNN['W'] @ h + RNN['U'] @ x + RNN['b'] # (m×1)
        h = np.tanh(a) # (m×1)
        o = RNN['V'] @ h + RNN['c'] # (K×1)
        
        o_shifted = o - np.max(o)
        exp_o = np.exp(o_shifted)
        p = exp_o / np.sum(exp_o)  # (K×1)

        # sample a character
        cp = np.cumsum(p, axis=0)
        a_draw = rng.uniform()
        ii = np.argmax(cp - a_draw > 0)
        indices.append(ii)

        # sampled char becomes next input
        x = np.zeros_like(x0)
        x[ii, 0] = 1.0

    return ''.join(ind_to_char[i] for i in indices)

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

    # Hyperparameters
    m = 100
    seq_length = 25
    eta= 0.001

    RNN = InitRNN(m, K, seed=42)

    print(f"\n-- RNN parameter shapes --")
    for key, val in RNN.items():
        print(f"  {key}: {val.shape}")

    rng_synth = np.random.default_rng(42)

    # seed: first character of the book, zero hidden state
    h0 = np.zeros((m, 1))
    x0 = Char2oneHot(book_data[0], char_to_ind, K).reshape(K, 1)

    synth = SynthesizeText(RNN, h0, x0, n=200, rng=rng_synth)
    print(f"\n-- Synthesized text (random init) --\n{synth}")