import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, random_split

from working_code.prepare_dataset import prep_dataset
from working_code.galaxy_dataset import GalaxyDataset
from working_code.galaxy_model import GalaxyCNN


def train_model():
    # Step 1: Load the galaxy catalogue
    df = prep_dataset()
    valid_images = df[df["image_exists"]]

    dataset = GalaxyDataset(valid_images)

    # Step 2: Split into training and validation datasets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Step 3: Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    # Step 4: Initialize the CNN
    model = GalaxyCNN()

    # Step 5: Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # Empty arrays to later visualize results with matplotlib.
    training_losses = []
    validation_accuracies = []

    # Step 6: Train for five epochs
    for epoch in range(5):
        model.train()

        total_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()

            predictions = model(images)

            loss = loss_function(predictions, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * images.size(0)

        average_loss = total_loss / len(train_dataset)

        # Step 7: Evaluate on validation data
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                predictions = model(images)

                predicted_classes = predictions.argmax(dim=1)

                correct += (predicted_classes == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total

        training_losses.append(average_loss)
        validation_accuracies.append(accuracy * 100)

        print(
            f"Epoch {epoch + 1}/5 | "
            f"Training loss: {average_loss:.4f} | "
            f"Validation accuracy: {accuracy:.2%}"
        )

        epochs = range(1, len(training_losses) + 1)

        # Graph 1: Training loss
        plt.figure()
        plt.plot(epochs, training_losses, marker="o")
        plt.title("CNN Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig("training_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Graph 2: Validation accuracy
        plt.figure()
        plt.plot(epochs, validation_accuracies, marker="o")
        plt.title("CNN Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.grid(True)
        plt.savefig("validation_accuracy.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    train_model()
