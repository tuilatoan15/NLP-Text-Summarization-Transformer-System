import zipfile
from pathlib import Path

dataset_dir = Path("C:/Users/ASUS/Desktop/NLP-Text-Summarization-Transformer-System/data/vietnamese-summarization-dataset-0001")
zips = ["vit5-colab-finetuned.zip", "mt5-colab-finetuned.zip", "bartpho-colab-finetuned.zip"]

for z in zips:
    zip_path = dataset_dir / z
    if zip_path.exists():
        print(f"=== Inspecting {z} ===")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Print first 10 files in the zip
            namelist = zip_ref.namelist()
            print(f"Total files: {len(namelist)}")
            for name in namelist[:15]:
                print(f"  {name}")
    else:
        print(f"{z} does not exist!")
