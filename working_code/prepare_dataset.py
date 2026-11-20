from pathlib import Path
import pandas as pd

# Paths
CSV_PATH = Path(r"C:\Euclid_Data\morphology_catalogue.csv")

# Change this to the folder where you extracted the TAR
IMAGE_ROOT = Path(
    r"c:\Euclid_Data\extract_cutouts_jpg_gz_arcsinh_vis_only.tar"
)

def prep_dataset():
# Load a small sample
    df = pd.read_csv(CSV_PATH, nrows=1000)

    def get_image_path(row):
        tile = str(row["tile_index"])
        object_id = str(row["object_id"]).replace("-", "NEG")

        filename = (
            f"{tile}_{object_id}_gz_arcsinh_vis_only.jpg"
        )

        return IMAGE_ROOT / tile / filename

    # Construct image paths
    df["image_path"] = df.apply(get_image_path, axis=1)
    # Check which images actually exist
    df["image_exists"] = df["image_path"].apply(lambda path: path.exists())

    return df

