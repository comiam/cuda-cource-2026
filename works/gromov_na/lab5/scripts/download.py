import os
import requests
from pathlib import Path
import zipfile
from concurrent.futures import ThreadPoolExecutor
import random

def download_coco_val2017_images(output_dir="calibration_images", num_images=400):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    zip_url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = output_dir / "val2017.zip"
    
    if not zip_path.exists():
        response = requests.get(zip_url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    extract_dir = output_dir / "val2017"
    if not extract_dir.exists() or len(list(extract_dir.glob("*.jpg"))) < 5000:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    
    all_images = list(extract_dir.glob("*.jpg"))
    
    selected = random.sample(all_images, min(num_images, len(all_images)))
    
    for img_path in selected:
        dest = output_dir / img_path.name
        if not dest.exists():
            os.rename(img_path, dest)


if __name__ == "__main__":
    download_coco_val2017_images(num_images=400)
