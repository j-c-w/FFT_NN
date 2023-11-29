import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import math

torch.autograd.set_detect_anomaly(True)

BATCH_SIZE = 64

class CustomLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(CustomLinear, self).__init__()
        # Initialize weights and bias
        self.bias = nn.Parameter(torch.randn(output_features))

        # div features by 2 internall to account for difference
        # between real and imaginary
        self.input_features = input_features // 2
        self.pad_to = 2 ** (math.ceil(math.log2(max(input_features // 2, output_features // 2))))
        self.output_features = output_features // 2

        self.weights = to_imag_no_batch(nn.Parameter(torch.randn(2 * self.pad_to)))

    def forward(self, x):
        # Perform the linear operation: y = xW^T + b
        if self.input_features < self.pad_to:
            x = torch.nn.functional.pad(x, (0, self.pad_to - self.input_features))

        results = []
        for i in range(x.size(0)):
            x_sub = x[i]
            x_fft = torch_fft(x_sub, self.weights)

            if self.output_features < self.pad_to:
                x_fft = x_fft[0: self.output_features]
            results.append(x_fft)

        x_fft = torch.stack(results)
        return x_fft

def to_real(imag):
    num = imag.shape[-1]
    return torch.view_as_real(imag).view(-1, num * 2, 1)

def to_imag_no_batch(real):
    last = real.shape[-1]
    return torch.view_as_complex(real.view(last // 2, 2))

def to_imag(real):
    last = real.shape[-1]
    return torch.view_as_complex(real.view(-1, last // 2, 2))

def torch_fft(x, twiddles):
    """
    Apply custom twiddle factors to input data and perform FFT using PyTorch.

    Parameters:
    x (Tensor): Input data for FFT.
    twiddles (Tensor): Custom twiddle factors, same length as x.

    Returns:
    Tensor: The FFT of the modified input data.
    """
    if x.shape != twiddles.shape:
        raise ValueError("Shape of twiddles must match shape of input data x")

    # Element-wise multiplication of input data with custom twiddles
    modified_input = torch.mul(x, twiddles)

    # Perform FFT on the modified input
    fft_result = torch.fft.fft(modified_input)

    return fft_result


class CustomFFTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, twiddle_factors):
        # Implement the forward pass of the FFT
        # Save necessary tensors for backward pass
        ctx.save_for_backward(input, twiddle_factors)
        output = fft(input, twiddle_factors)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Implement the backward pass
        input, twiddle_factors = ctx.saved_tensors
        grad_input = grad_twiddle_factors = None

        # Compute gradient w.r.t. input and twiddle factors
        if ctx.needs_input_grad[0]:
            # grad_input = ...  # Gradient w.r.t. input
            # grad at n is given by: e^{-i2pi k n / N}
            TODO
            pass
        if ctx.needs_input_grad[1]:
            # Grad wrt twiddle factors is more complex.  :q
            pass
            # grad_twiddle_factors = # ...  # Gradient w.r.t. twiddle factors

        return grad_input, grad_twiddle_factors

def fft(x, twiddle_factors):
    N = x.shape[0]
    # Bit-reversal reordering
    x = x[torch.bitwise_xor(torch.arange(N), (N // 2)).long()]

    # Iterative FFT
    step = 2
    iter = 0
    while step <= N:
        half_step = step // 2
        for i in range(0, N, step):
            k = 0
            for j in range(i, i + half_step):
                temp = torch.exp(-2j * torch.pi * k / N * twiddle_factors[k]) * x[j + half_step]
                x[j + half_step] = x[j] - temp
                x[j] = x[j] + temp
                k += N // step
            iter += 1
        step *= 2

    return x

def fft_recursive(signal, twiddles):
    N = signal.numel()
    if N <= 1:
        return signal

    # Cooley-Tukey FFT algorithm
    even = fft_recursive(signal[0::2], twiddles[0::2])
    odd = fft_recursive(signal[1::2], twiddles[1::2])

    T = torch.exp(twiddles * -2j * torch.pi * torch.arange(N) / N)
    return torch.cat([even + T[:N // 2] * odd, even + T[N // 2:] * odd])


def fft_iterative(signal, twiddles):
    N = signal.numel()
    if N & (N - 1):
        raise ValueError("Size of input must be a power of 2")

    # Bit-reversal permutation
    J = torch.bitwise_xor(torch.arange(N // 2), torch.arange(N // 2).flip([0]))
    signal_permuted = signal.clone()[J]

    # Iterative FFT
    L = 1
    while L < N:
        L2 = L * 2
        exp_factor = torch.exp(-2j * torch.pi * torch.arange(L) / L2)
        for k in range(0, N, L2):
            t = exp_factor * signal_permuted[k + L:k + L2]
            signal_permuted = torch.cat([signal_permuted[:k],
                                         signal_permuted[k:k + L] + t,
                                         signal_permuted[k + L:k + L2] - t,
                                         signal_permuted[k + L2:]])
        L = L2

    return signal_permuted


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.hidden = CustomLinear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = CustomLinear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(to_imag(x))
        x = torch.squeeze(self.relu(to_real(x)))
        x = self.output(to_imag(x))
        return to_real(x)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, and Optimizer
input_size = 784  # 28x28 pixels flattened
hidden_size = 128
output_size = 10  # 10 classes for digits 0-9
model = SimpleMLP(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    print("Epoch ", epoch)
    print("Running loss ", running_loss)
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.view(inputs.shape[0], -1)  # Flatten the images

        optimizer.zero_grad()

        outputs = model(inputs).view(-1, output_size)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
