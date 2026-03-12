import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        The Spectral Convolution Layer: 
        This is what allows the AI to 'see' the universe as waves.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to keep
        self.modes2 = modes2

        scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # Complex multiplication for the Fourier modes
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # 1. Transform to Fourier space
        x_ft = torch.fft.rfft2(x)

        # 2. Multiply relevant Fourier modes by learned weights
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # 3. Return to spatial domain (Inverse FFT)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1=24, modes2=24, width=64):
        super(FNO2d, self).__init__()
        """
        Neural-AdS Architecture (High Precision):
        A 4-layer Fourier Neural Operator.
        Defaults: modes=24, width=64 for <1% Error.
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        # Lifting: Maps input (1 channel) to High-Dim feature space (width)
        self.p = nn.Conv2d(1, self.width, 1)

        # Fourier Layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        # W Layers
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        # Projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Input shape: (Batch, 1, 64, 64)
        x = self.p(x) # -> (Batch, Width, 64, 64)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # Project back to output dimension
        x = x.permute(0, 2, 3, 1) # (Batch, 64, 64, Width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        # Permute back to (Batch, 1, 64, 64) to match Target shape
        x = x.permute(0, 3, 1, 2)
        return x
