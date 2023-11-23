import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import math

BATCH_SIZE = 64

class CustomLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(CustomLinear, self).__init__()
        # Initialize weights and bias
        self.bias = nn.Parameter(torch.randn(output_features))

        # div features by 2 internall to account for difference
        # between real and imaginary
        self.input_features = input_features // 2
        self.pad_to = 2 ** (math.ceil(math.log2(max(input_features, output_features))))
        self.output_features = output_features // 2

        self.weights = nn.Parameter(torch.randn(self.pad_to))

    def forward(self, x):
        # Perform the linear operation: y = xW^T + b
        if self.input_features < self.pad_to:
            x = torch.nn.functional.pad(x, (0, self.pad_to - self.input_features))

        results = []
        for i in range(x.size(0)):
            x_sub = x[i]
            x_fft = fft(x_sub, self.weights)
            results.append(x_fft)

        x_fft = torch.stack(results)
        if self.output_features < self.pad_to:
            x_fft = x_fft[0: self.output_features]

        return x_fft

def to_real(imag):
    num = imag.shape[-2]
    return torch.view_as_real(imag).view(BATCH_SIZE, num * 2, 1)

def to_imag(real):
    last = real.shape[-1]
    return torch.view_as_complex(real.view(BATCH_SIZE, last // 2, 2))


def fft(x, twiddle_factors):
    N = x.shape[0]
    # Bit-reversal reordering
    x = x[torch.bitwise_xor(torch.arange(N), (N // 2)).long()]

    # Iterative FFT
    step = 2
    while step <= N:
        half_step = step // 2
        for i in range(0, N, step):
            k = 0
            for j in range(i, i + half_step):
                temp = torch.exp(-2j * torch.pi * k / N * twiddle_factors[k]) * x[j + half_step]
                x[j + half_step] = x[j] - temp
                x[j] = x[j] + temp
                k += N // step
        step *= 2

    return x

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
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.view(inputs.shape[0], -1)  # Flatten the images

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
