import requests
import zipfile
import os
from tqdm import tqdm

"""
    Télécharge les données COCO VAL 2017 automatiquement
    Tu voudras changer la variable DATASET_PATH dans .env pour ./data/coco
"""

os.makedirs("./data/coco", exist_ok=True)

urls = {
    "images": "http://images.cocodataset.org/zips/val2017.zip",
    "captions": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

def download_and_extract(url, dest_folder):
    filename = url.split("/")[-1]
    file_path = os.path.join(dest_folder, filename)
    
    print(f"Téléchargement de {filename}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(file_path, "wb") as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    
    os.remove(file_path)
    print("Done.")

download_and_extract(urls["images"], "./data/coco")
download_and_extract(urls["captions"], "./data/coco")
