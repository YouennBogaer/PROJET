import requests
import zipfile
import os
from tqdm import tqdm # pip install tqdm (pour la barre de progression)

# Création du dossier
os.makedirs("./data/coco", exist_ok=True)

urls = {
    "images": "http://images.cocodataset.org/zips/val2017.zip",
    "captions": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

def download_and_extract(url, dest_folder):
    filename = url.split("/")[-1]
    file_path = os.path.join(dest_folder, filename)
    
    print(f"📥 Téléchargement de {filename}...")
    
    # Téléchargement avec barre de progression
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
            
    print(f"📦 Extraction de {filename}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    
    # Supprimer le zip pour gagner de la place
    os.remove(file_path)
    print("✅ Terminé !")

# Lancer les téléchargements
download_and_extract(urls["images"], "./data/coco")
download_and_extract(urls["captions"], "./data/coco")
