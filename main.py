# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from scipy.special import softmax
import sys
import argparse


def gemm(A, B, transA=False):
    return A @ B if False == transA else A.T @ B


def main(d, h, l, b):
    print("Starting the Transformer Layer Simulator")
    print("D = {}, H = {}, L={}, B = {}".format(d, h, l, b))
    f = 4 * d
    l = l * b
    ######################### Initial data
    EI = np.random.rand(d, l)
    Wq = np.random.rand(d, d)
    Wk = np.random.rand(d, d)
    Wv = np.random.rand(d, d)
    Wout = np.random.rand(d, d)

    W1 = np.random.rand(f, d)
    W2 = np.random.rand(d, f)
    runing_mean = np.random.rand(l)
    inv_std = np.random.rand(l)
    gamma = np.random.rand(l)
    beta = np.random.rand(l)

    E2 = np.empty((d, l), dtype=float)

    assert (d % h == 0)
    split = d // h
    E1s = []
    E2s = []
    ##########################
    Q = gemm(Wq, EI)
    K = gemm(Wk, EI)
    V = gemm(Wv, EI)

    assert (Q.shape == K.shape == V.shape == (d, l))

    for i in range(split):
        E1s.append(gemm(K[i * h:(i + 1) * h, 0:l], Q[i * h:(i + 1) * h, 0:l], transA=True))
        assert (E1s[i].shape == (l, l))
        E1s[i] = softmax(E1s[i])

        E2s.append(gemm(V[i * h:(i + 1) * h, 0:l], E1s[i]))
        assert (E2s[i].shape == (h, l))

    print(E2.shape)
    for i in range(split):
        E2[i * h:(i + 1) * h, 0:l] = E2s[i]
    #
    assert (E2.shape == (d, l))
    AO = gemm(Wout, E2)
    # AO layer norm (A0) + EI
    assert (AO.shape == (d, l))

    #AO = bn_inference_cython(AO, runing_mean, inv_std, gamma, beta)
    AO = AO + EI
    assert (AO.shape == (d, l))

    ########### FFN

    E4 = gemm(W1, AO)

    assert (E4.shape == (f, l))

    # E4 = gelu(E4)

    E0 = gemm(W2, E4) + AO
    # E0 layer norm (E0) + AO
    print(E0)
    assert (E0.shape == (d, l))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTransformer')
    parser.add_argument('--d', type=int, default=32, help='d')
    parser.add_argument('--h', type=int, default=8, help='h')
    parser.add_argument('--l', type=int, default=16, help='l')
    parser.add_argument('--b', type=int, default=1, help='b')
    args = parser.parse_args()

    main(args.d, args.h, args.l, args.b)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
