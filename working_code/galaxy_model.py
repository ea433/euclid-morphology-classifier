import torch
from torch import nn
import timm

class GalaxyCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Second convolutional block
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(use_pretrained=False):
    """Return either the pretrained Zoobot model or the prototype CNN (baseline)."""
    if use_pretrained:
        # Pretrained Zoobot encoder (greyscale ConvNeXt-Nano) + new head with 3 classes
        return timm.create_model(
            "hf_hub:mwalmsley/zoobot-encoder-greyscale-convnext_nano",
            pretrained=True,
            num_classes=3,
            in_chans=1,
            pretrained_cfg_overlay=dict(input_size=(1, 224, 224)),
        )

    # Prototype CNN (baseline), expects 128 x 128 images
    return GalaxyCNN()


if __name__ == "__main__":
    # Test the prototype CNN
    model = create_model(use_pretrained=False)
    test_images = torch.randn(32, 1, 128, 128)
    output = model(test_images)

    print("Prototype CNN")
    print("Input shape:", test_images.shape)
    print("Output shape:", output.shape)
    print("First galaxy scores:", output[0])

    # Test the pretrained Zoobot model
    model = create_model(use_pretrained=True)
    test_images = torch.randn(32, 1, 224, 224)
    output = model(test_images)

    print("\nPretrained Zoobot model")
    print("Input shape:", test_images.shape)
    print("Output shape:", output.shape)
    print("First galaxy scores:", output[0])