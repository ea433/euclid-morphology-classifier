from working_code.prepare_dataset import prep_dataset

from collections import Counter

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



def img_loading():
    df = prep_dataset() # nrows = 1000

    valid_images = df[df["image_exists"]]
    first_path = valid_images["image_path"].iloc[12]

    with Image.open(first_path) as img:
        img = img.convert("L")
        image_array = np.array(img)

    return image_array

image_array = img_loading()

print("Image shape:", image_array.shape)
print("Data type:", image_array.dtype)

plt.imshow(image_array, cmap="gray")
plt.title("Euclid Galaxy Cutout")
plt.axis("off")
plt.show()

def inspect_image_sizes():
    df = prep_dataset()

    # Only keep rows with existing images
    valid_images = df[df["image_exists"]]

    image_sizes = []

    # Inspect the first 100 valid images
    for path in valid_images["image_path"].head(100):
        with Image.open(path) as img:
            image_sizes.append(img.size)

    # Count how frequently each image size occurs
    size_counts = Counter(image_sizes)

    print("Image dimensions:")
    for size, count in size_counts.items():
        print(f"{size}: {count} images")

print(inspect_image_sizes())
