import torch

def bit_reversal_permutation(n):
    """
    Generate the bit-reversal permutation for n points, where n is a power of 2.
    """
    bits = int(torch.log2(torch.tensor(n)).item())
    perm = torch.arange(n).view(-1, 1)
    perm = perm.repeat(1, bits)
    mask = torch.tensor([1 << i for i in range(bits)]).view(1, -1)
    perm = torch.bitwise_and(perm, mask).ne(0).long()
    perm = perm.flip(1)
    weights = torch.tensor([2 ** i for i in range(bits)])
    perm = torch.matmul(perm, weights.view(-1, 1)).view(-1)
    return perm

def fft(x, twiddles):
    N = x.shape[0]
    if N <= 1: return x

    # Bit-reversal permutation
    reversed_indices = bit_reversal_permutation(N)
    x = x[reversed_indices]

    # Iterative FFT
    M = 1
    while M < N:
        for k in range(M):
            w = torch.exp(torch.tensor([-2j * torch.pi * k * twiddles[k] / (2 * M)]))
            for n in range(k, N, 2 * M):
                t = w * x[n + M]
                x[n + M] = x[n] - t
                x[n] = x[n] + t
        M *= 2

    return x

# dX[k] / d x
def differentiated_fft(x, twiddles):
    #equation of the dft is just
    # X[k] = sum_0^N-1 x[n] * e^{-2pi ink tidl[n] / N}
    # so differentiated wrt. x[n] gives:
    # e^{-2pi ink tidl[n] / N}
    N = x.shape[0]
    torch.tensor([torch.exp(-2j * torch.pi * n * k / N )]

            TODO -- how did I get an extra dimension?


def inverse_fft(x, twiddles):
    # inverrt sign of twiddles[k].
    neg_twiddles = [complex(-twiddles[k].real, -twiddles[k].imag) for k in range(len(twiddles))]
    N = x.shape[0]
    return (1 / N) * fft(x, neg_twiddles)

twids = torch.tensor([1 + 2.444j, 3 + 1j, -1, -1j], dtype=torch.complex64)
res = fft(torch.tensor([1,2, 3, 4], dtype=torch.complex64), twids)
print("res is ", res)
print("inverse is ", inverse_fft(res, twids))
