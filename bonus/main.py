import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch_gradient_computations_column_wise import ComputeGradsWithTorch

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

def SynthesizeTemp(RNN, h0, x0, n, rng, T=1.0):
    """
    Temperature sampling. T<1 = more peaked (safer), T>1 = flatter (more random).
    """
    h = h0.copy()
    x = x0.copy()
    indices = []

    for _ in range(n):
        a = RNN['W'] @ h + RNN['U'] @ x + RNN['b']
        h = np.tanh(a)
        o = RNN['V'] @ h + RNN['c']

        # divide by T before softmax
        o_scaled = o / T
        o_scaled -= np.max(o_scaled)
        p = np.exp(o_scaled) / np.sum(np.exp(o_scaled))

        cp = np.cumsum(p, axis=0)
        ii = np.argmax(cp - rng.uniform() > 0)
        indices.append(ii)

        x = np.zeros_like(x0)
        x[ii, 0] = 1.0

    return ''.join(ind_to_char[i] for i in indices)

def SynthesizeNucleus(RNN, h0, x0, n, rng, theta=0.9):
    """
    Nucleus sampling. Only sample from the smallest top-k chars summing to >= theta.
    Low theta = small nucleus (safe), high theta = large nucleus (diverse).
    """
    h = h0.copy()
    x = x0.copy()
    indices = []

    for _ in range(n):
        a = RNN['W'] @ h + RNN['U'] @ x + RNN['b']
        h = np.tanh(a)
        o = RNN['V'] @ h + RNN['c']

        o -= np.max(o)
        p = np.exp(o) / np.sum(np.exp(o))  
        p_flat = p.flatten()


        sorted_inds = np.argsort(p_flat)[::-1]   
        sorted_probs = p_flat[sorted_inds]

        cumsum = np.cumsum(sorted_probs)
        k = np.argmax(cumsum >= theta) + 1
        nucleus_inds  = sorted_inds[:k]
        nucleus_probs = sorted_probs[:k]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()  # renormalize

        ii = rng.choice(nucleus_inds, p=nucleus_probs)

        indices.append(ii)
        x = np.zeros_like(x0)
        x[ii, 0] = 1.0

    return ''.join(ind_to_char[i] for i in indices)


def ForwardPass(RNN, X, Y, h0):
    """
    X:  K×τ  one-hot input matrix
    Y:  K×τ  one-hot target matrix
    h0: m×1  initial hidden state
    Returns: loss (scalar), P (K×τ), cache dict
    """
    K, tau = X.shape
    m = h0.shape[0]

    H = np.zeros((m, tau))   # hidden states at each timestep

    h = h0.copy()
    for t in range(tau):
        a = RNN['W'] @ h + RNN['U'] @ X[:, t:t+1] + RNN['b']  # (m×1)
        h = np.tanh(a) # (m×1)
        H[:, t:t+1] = h

    # output layer, vectorized over all timesteps
    O = RNN['V'] @ H + RNN['c']# (K×τ)

    # numerically stable softmax
    O -= np.max(O, axis=0, keepdims=True)
    exp_O = np.exp(O)
    P = exp_O / np.sum(exp_O, axis=0, keepdims=True) # (K×τ)

    # loss
    y_inds = np.argmax(Y, axis=0)
    loss = -np.mean(np.log(P[y_inds, np.arange(tau)]))

    cache = {'X': X, 'Y': Y, 'H': H, 'P': P, 'h0': h0}
    return loss, P, cache


def BackwardPass(RNN, cache):
    """
    Returns gradients for all 5 parameters.
    """
    X   = cache['X']
    Y   = cache['Y']
    H   = cache['H']
    P   = cache['P']
    h0  = cache['h0']
    tau = X.shape[1]

    G = (P - Y) / tau # K×τ

    grad_V = G @ H.T # K×m
    grad_c = np.sum(G, axis=1, keepdims=True)  # K×1

    grad_W = np.zeros_like(RNN['W'])  # m×m
    grad_U = np.zeros_like(RNN['U'])  # m×K
    grad_b = np.zeros_like(RNN['b'])  # m×1

    grad_a_next = np.zeros((H.shape[0], 1)) # m×1

    for t in reversed(range(tau)):
        h_t    = H[:, t:t+1]  # m×1
        h_prev = H[:, t-1:t] if t > 0 else h0  # m×1
        x_t    = X[:, t:t+1] # K×1
        g_t    = G[:, t:t+1] # K×1

        grad_h = RNN['V'].T @ g_t + RNN['W'].T @ grad_a_next   # m×1


        grad_a = grad_h * (1 - h_t**2)             # m×1

        grad_W += grad_a @ h_prev.T
        grad_U += grad_a @ x_t.T
        grad_b += grad_a

        grad_a_next = grad_a

    return {'V': grad_V, 'c': grad_c, 'W': grad_W, 'U': grad_U, 'b': grad_b}

def InitAdam(RNN):
    """Initialize first and second moment vectors to zero, same shape as each param."""
    m = {k: np.zeros_like(v) for k, v in RNN.items()}
    v = {k: np.zeros_like(v) for k, v in RNN.items()}
    return m, v


def AdamUpdate(RNN, grads, m, v, t, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam update step.
    t: current iteration (1-indexed, increment before calling)
    Updates RNN, m, v in place.
    """
    for k in RNN:
        # update moments
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * grads[k]**2

        # bias correction
        m_hat = m[k] / (1 - beta1**t)
        v_hat = v[k] / (1 - beta2**t)

        # parameter update
        RNN[k] -= eta * m_hat / (np.sqrt(v_hat) + eps)

def TrainRNN(RNN, book_data, char_to_ind, ind_to_char, K, m,
             seq_length=25, eta=0.001, n_epochs=2, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    m_adam, v_adam = InitAdam(RNN)
    t = 0
    smooth_loss = -np.log(1.0 / K)
    best_loss   = float('inf')
    best_synth  = ""

    for epoch in range(1, n_epochs + 1):
        e      = 0
        hprev  = np.zeros((m, 1))

        while e + seq_length + 1 <= len(book_data):
            X_seq = Char2oneHot(book_data[e : e+seq_length],     char_to_ind, K)
            Y_seq = Char2oneHot(book_data[e+1 : e+seq_length+1], char_to_ind, K)

            loss, P, cache = ForwardPass(RNN, X_seq, Y_seq, hprev)
            grads = BackwardPass(RNN, cache)

            for k in grads:
                grads[k] = np.clip(grads[k], -5, 5)

            t += 1
            AdamUpdate(RNN, grads, m_adam, v_adam, t, eta=eta)

            hprev = cache['H'][:, -1:]

            smooth_loss = 0.999 * smooth_loss + 0.001 * loss

            if smooth_loss < best_loss:
                best_loss  = smooth_loss
                best_synth = SynthesizeText(RNN, hprev, X_seq[:, 0:1], 1000, rng)

            if t % 100 == 0:
                print(f"  epoch {epoch} | step {t:6d} | smooth_loss = {smooth_loss:.4f}")

            if t % 1000 == 0:
                synth = SynthesizeText(RNN, hprev, X_seq[:, 0:1], 200, rng)
                print(f"\n  [step {t}]\n{synth}\n")
                with open(out_dir / f"synthesis_step{t}.txt", 'w') as f:
                    f.write(synth)

            e += seq_length

        print(f"\n  ── epoch {epoch} complete | step {t} | smooth_loss = {smooth_loss:.4f} ──\n")

    with open(out_dir / "best_synthesis_1000chars.txt", 'w') as f:
        f.write(best_synth)
    print(f"\nBest loss reached: {best_loss:.4f}")

    return RNN, smooth_loss

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

    seq = book_data[0:seq_length]
    X_seq = Char2oneHot(seq,          char_to_ind, K)   # K×25
    Y_seq = Char2oneHot(book_data[1:seq_length+1], char_to_ind, K)

    h0 = np.zeros((m, 1))
    loss, P, cache = ForwardPass(RNN, X_seq, Y_seq, h0)
    grads = BackwardPass(RNN, cache)

    print(f"\n-- Forward/Backward sanity --")
    print(f"  loss : {loss:.4f}  (expect ~log({K}) = {np.log(K):.4f} for random weights)")
    for k, v in grads.items():
        print(f"  grad_{k}: {v.shape}  max_abs={np.max(np.abs(v)):.4e}")

    m_small  = 10
    seq_small = 25
    rng_check = np.random.default_rng(0)

    small_RNN = {}
    small_RNN['U'] = (1/np.sqrt(2*K))    * rng_check.standard_normal((m_small, K))
    small_RNN['W'] = (1/np.sqrt(2*m_small)) * rng_check.standard_normal((m_small, m_small))
    small_RNN['V'] = (1/np.sqrt(m_small)) * rng_check.standard_normal((K, m_small))
    small_RNN['b'] = np.zeros((m_small, 1))
    small_RNN['c'] = np.zeros((K, 1))

    X_chk = Char2oneHot(book_data[0:seq_small],   char_to_ind, K)
    Y_chk = Char2oneHot(book_data[1:seq_small+1], char_to_ind, K)
    y_chk = np.argmax(Y_chk, axis=0)

    h0_chk = np.zeros((m_small, 1))

    _, _, cache_chk = ForwardPass(small_RNN, X_chk, Y_chk, h0_chk)
    my_grads    = BackwardPass(small_RNN, cache_chk)
    torch_grads = ComputeGradsWithTorch(X_chk, y_chk, h0_chk, small_RNN)

    print("\n-- Gradient check --")
    for k in my_grads:
        abs_err = np.max(np.abs(my_grads[k] - torch_grads[k]))
        rel_err = np.max(np.abs(my_grads[k] - torch_grads[k]) /
                        np.maximum(1e-10, np.abs(my_grads[k]) + np.abs(torch_grads[k])))
        print(f"  {k}:  max abs err = {abs_err:.2e}   max rel err = {rel_err:.2e}")

    print("\n-- Overfit sanity check --")
    overfit_RNN = InitRNN(m, K, seed=42)
    X_over = Char2oneHot(book_data[0:seq_length],   char_to_ind, K)
    Y_over = Char2oneHot(book_data[1:seq_length+1], char_to_ind, K)
    h0_over = np.zeros((m, 1))
    eta_over = 0.1

    for step in range(200):
        loss_over, _, cache_over = ForwardPass(overfit_RNN, X_over, Y_over, h0_over)
        grads_over = BackwardPass(overfit_RNN, cache_over)
        for k in overfit_RNN:
            overfit_RNN[k] -= eta_over * grads_over[k]
        if (step+1) % 50 == 0:
            print(f"  step {step+1:3d}  loss = {loss_over:.4f}")

    print("\n-- Overfit check with Adam --")
    adam_RNN    = InitRNN(m, K, seed=42)
    m_adam, v_adam = InitAdam(adam_RNN)
    h0_over     = np.zeros((m, 1))
    adam_step   = 0

    for step in range(200):
        loss_a, _, cache_a = ForwardPass(adam_RNN, X_over, Y_over, h0_over)
        grads_a = BackwardPass(adam_RNN, cache_a)
        adam_step += 1
        AdamUpdate(adam_RNN, grads_a, m_adam, v_adam, adam_step, eta=0.001)
        if (step+1) % 50 == 0:
            print(f"  step {step+1:3d}  loss = {loss_a:.4f}")

    print("\n-- Training --")
    RNN = InitRNN(m, K, seed=42)
    rng_train = np.random.default_rng(42)
    RNN, final_loss = TrainRNN(
        RNN, book_data, char_to_ind, ind_to_char, K, m,
        seq_length=seq_length, eta=eta, n_epochs=2, rng=rng_train
    )

    print("\n-- Bonus 2.3: Sampling strategies --")
    h0_synth = np.zeros((m, 1))
    x0_synth = Char2oneHot(book_data[0], char_to_ind, K).reshape(K, 1)
    rng_bonus = np.random.default_rng(42)

    # Temperature
    for T in [0.5, 0.75, 1.5]:
        synth = SynthesizeTemp(RNN, h0_synth, x0_synth, 200, rng_bonus, T=T)
        print(f"\n  [Temperature T={T}]\n{synth}")
        with open(out_dir / f"temp_T{T}.txt", 'w') as f:
            f.write(synth)

    # Nucleus
    for theta in [0.5, 0.75, 0.95]:
        synth = SynthesizeNucleus(RNN, h0_synth, x0_synth, 200, rng_bonus, theta=theta)
        print(f"\n  [Nucleus theta={theta}]\n{synth}")
        with open(out_dir / f"nucleus_theta{theta}.txt", 'w') as f:
            f.write(synth)