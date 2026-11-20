
from PIL import Image
from torchvision import transforms
from prepare_dataset import prep_dataset
from torch.utils.data import Dataset, DataLoader

# Define how every galaxy image should be processed
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def process_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert("L")
        image_tensor = transform(img)

    return image_tensor


# Step 2: Create a PyTorch Dataset
class GalaxyDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

        self.label_columns = [
            "smooth-or-featured_smooth_fraction",
            "smooth-or-featured_featured-or-disk_fraction",
            "smooth-or-featured_problem_fraction"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = process_image(row["image_path"])

        fractions = row[self.label_columns]
        label = int(fractions.to_numpy().argmax())

        return image, label


if __name__ == "__main__":
    df = prep_dataset()

    # Select only galaxies with existing images
    valid_images = df[df["image_exists"]]

    label_columns = [
        "smooth-or-featured_smooth_fraction",
        "smooth-or-featured_featured-or-disk_fraction",
        "smooth-or-featured_problem_fraction"
    ]

    print(valid_images[label_columns].head(10))

    print("\nMissing values:")
    print(valid_images[label_columns].isna().sum())

    # Retrieve the path of the first valid image
    first_path = valid_images["image_path"].iloc[0]

    # Pass that path to our function
    image_tensor = process_image(first_path)

    print(image_tensor.shape)
    print(image_tensor.dtype)

    # Step 3: Create the Dataset and DataLoader
    dataset = GalaxyDataset(valid_images)

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    # Retrieve the first batch
    images, labels = next(iter(dataloader))

    print("\nNumber of galaxies:", len(dataset))
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    print("First 10 labels:", labels[:10])
