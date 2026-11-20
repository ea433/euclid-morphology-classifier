import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from prepare_dataset import prep_dataset
from galaxy_dataset import GalaxyDataset
from galaxy_model import create_model
USE_PRETRAINED = True # false for my own CNN


def train_model():
    # Step 1: Load the galaxy catalogue
    df = prep_dataset()
    valid_images = df[df["image_exists"]]

    dataset = GalaxyDataset(valid_images)

    class_names = ["Smooth", "Featured/Disk", "Problem"]
    class_counts = [0, 0, 0]

    for index in range(len(dataset)):
        label = int(
            valid_images.iloc[index][dataset.label_columns]
            .to_numpy()
            .argmax()
        )

        class_counts[label] += 1

    for name, count in zip(class_names, class_counts):
        print(f"{name}: {count}")

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

    images, labels = next(iter(train_loader))
    print("Batch image shape:", images.shape)
    return

    # Step 4: Initialize the CNN
    model = create_model(use_pretrained=USE_PRETRAINED)

    device = "cuda" if torch.cuda.is_available() else "cpu"  
    model = model.to(device)                                  
    print("Using device:", device)  

    # Step 5: Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()

    learning_rate = 0.0001 if USE_PRETRAINED else 0.001 # lower lr pretrained
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Step 6: Train for five epochs
    for epoch in range(5):
        model.train()

        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
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
                images, labels = images.to(device), labels.to(device)

                predictions = model(images)

                predicted_classes = predictions.argmax(dim=1)

                correct += (predicted_classes == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total

        print(
            f"Epoch {epoch + 1}/5 | "
            f"Training loss: {average_loss:.4f} | "
            f"Validation accuracy: {accuracy:.2%}"
        )

        # Step 8: Evaluate predictions for each morphology category
        model.eval()

        confusion = torch.zeros((3, 3), dtype=torch.int64)

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                predictions = model(images)
                predicted_classes = predictions.argmax(dim=1)

                for actual, predicted in zip(labels, predicted_classes):
                    confusion[actual.item(), predicted.item()] += 1 # added .item() 

        print("\nConfusion Matrix:")
        print(confusion.numpy())

        print("\nClassification Report:")

        for i, name in enumerate(class_names):
            tp = confusion[i, i].item()

            actual_total = confusion[i, :].sum().item()
            predicted_total = confusion[:, i].sum().item()

            precision = tp / predicted_total if predicted_total > 0 else 0
            recall = tp / actual_total if actual_total > 0 else 0

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0 else 0
            )

            print(
                f"{name}: "
                f"Precision={precision:.3f}, "
                f"Recall={recall:.3f}, "
                f"F1={f1:.3f}, "
                f"Support={actual_total}"
            )

if __name__ == "__main__":
    train_model()
