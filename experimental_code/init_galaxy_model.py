'''
class GalaxyCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == "__main__":
    model = GalaxyCNN()

    test_images = torch.randn(32, 1, 128, 128)

    output = model(test_images)

    print("Input shape:", test_images.shape)
    print("Output shape:", output.shape)
'''