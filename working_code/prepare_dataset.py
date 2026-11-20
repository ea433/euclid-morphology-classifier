from pathlib import Path
import pandas as pd

PROJECT_ROOT = (Path(__file__).resolve().parent.parent)
DATA_DIR = PROJECT_ROOT / "data" 

CSV_PATH = DATA_DIR / "morphology_catalogue.csv"
#IMAGE_ROOT = DATA_DIR / "cutouts_jpg_gz_arcsinh_vis_only.tar" 
IMAGE_ROOT = DATA_DIR / "images"


def prep_dataset():
# Load a small sample
    df = pd.read_csv(CSV_PATH, nrows=10000)
    df.columns = df.columns.str.strip()

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

