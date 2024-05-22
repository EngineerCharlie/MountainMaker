import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_channels=16):
        super(Generator, self).__init__()

            
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            #nn.Dropout2d(0.5),  # Adding dropout after activation
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
            #nn.Dropout2d(0.5),  # Adding dropout after activation
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(),
            #nn.Dropout2d(0.5),  # Adding dropout after activation
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU(),
            #nn.Dropout2d(0.5),  # Adding dropout after activation
            #nn.Conv2d(hidden_channels * 8, hidden_channels * 16, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(hidden_channels * 16),
            #nn.ReLU(),
            #nn.Dropout2d(0.5)  # Adding dropout after activation
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(hidden_channels * 16, hidden_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(hidden_channels * 8),
            #nn.ReLU(),
            #nn.Dropout2d(0.5),  # Adding dropout after activation
            nn.ConvTranspose2d(hidden_channels * 8, hidden_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU(),
            #nn.Dropout2d(0.5),  # Adding dropout after activation
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(),
            #nn.Dropout2d(0.5),  # Adding dropout after activation
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            #nn.Dropout2d(0.5),  # Adding dropout after activation
            nn.ConvTranspose2d(hidden_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.Dropout2d(0.5),  # Adding dropout after activation
            #nn.ConvTranspose2d(output_channels, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Tanh activation for output pixels in [-1, 1] range
        )
        '''
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(input_channels, hidden_channels * 16, 4, 1, 0),  # img: 4x4
            self._block(hidden_channels * 16, hidden_channels * 8, 4, 2, 1),  # img: 8x8
            self._block(hidden_channels * 8, hidden_channels * 4, 4, 2, 1),  # img: 16x16
            self._block(hidden_channels * 4, hidden_channels * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                hidden_channels * 2, output_channels, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    '''
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        #x = self.net(x)
        return x
