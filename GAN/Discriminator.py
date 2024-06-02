import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,  # square 4x4 kernel
                stride=stride,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], size=256):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(
                    in_channels,
                    out_channels=feature,
                    stride=1 if feature == features[-1] else 2,
                ),
            )
            in_channels = feature

        self.model = nn.Sequential(
            *layers,
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            ),
        )

    def forward(self, x, y):
        # print("Batch size, channels, height, width")
        # print(f"Input x and y shape: {str(x.shape)}")
        # Joins the inputs and outputs to create a batch size * 6 channel * 256*256 array
        x = torch.cat([x, y], dim=1)
        # print(f"Joined x, y shape: {str(x.shape)}")
        x = self.initial(x)
        # print(f"Initial model output shape: {str(x.shape)}")
        x = self.model(x)
        # print(f"Final model output shape: {str(x.shape)}")
        return x


def test():
    size = 256
    x = torch.randn((1, 3, size, size))
    y = torch.randn((1, 3, size, size))
    model = Discriminator(in_channels=3, size=size)
    preds = model(x, y)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()
