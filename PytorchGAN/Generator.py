import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=128):
        super(Generator, self).__init__()
        #super().__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Tanh activation for output pixels in [-1, 1] range
        )

    def forward(self, x):
        #print("before encoding, the shape is: ", x.shape)
        x = self.encoder(x)
        #print("shape inbetween: ", x.shape)
        x = self.decoder(x)
        #print("after decoding, the shape is: ", x.shape)
        return x