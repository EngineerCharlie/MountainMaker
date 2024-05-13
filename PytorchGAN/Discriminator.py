import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels=16):
        super(Discriminator, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels * 2, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels * 4, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels * 8, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_channels * 8, 1, kernel_size=4, stride=2, padding=0),
            #nn.Sigmoid()
        )

    def forward(self, x):
        #print("SHAPE INPUIT: ", x.shape)
        x = self.encoder(x)
        return x.view(-1, 1)