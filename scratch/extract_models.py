import zipfile
import shutil
from pathlib import Path
import time

dataset_dir = Path("C:/Users/ASUS/Desktop/NLP-Text-Summarization-Transformer-System/data/vietnamese-summarization-dataset-0001")
models_dir = Path("C:/Users/ASUS/Desktop/NLP-Text-Summarization-Transformer-System/models")

zips = {
    "vit5-colab-finetuned.zip": ("vit5-colab-finetuned", "vit5-finetuned"),
    "mt5-colab-finetuned.zip": ("mt5-colab-finetuned", "mt5-finetuned"),
    "bartpho-colab-finetuned.zip": ("bartpho-colab-finetuned", "bartpho-finetuned")
}

print(f"Dataset directory: {dataset_dir}")
print(f"Models directory: {models_dir}")

for zip_name, (extracted_folder, target_folder) in zips.items():
    zip_path = dataset_dir / zip_name
    target_path = models_dir / target_folder
    extracted_path = models_dir / extracted_folder
    
    if target_path.exists() and any(target_path.iterdir()):
        print(f"Target folder {target_folder} already exists and is not empty, skipping.")
        continue
        
    if not zip_path.exists():
        print(f"Zip file {zip_name} does not exist at {zip_path}")
        continue
        
    print(f"\n--- Extracting {zip_name} to {models_dir} ---")
    t0 = time.time()
    
    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(models_dir)
        
    print(f"Extracted to {extracted_path} in {time.time() - t0:.2f} seconds.")
    
    # If the target path exists (e.g. was empty), remove it first
    if target_path.exists():
        print(f"Removing empty target path {target_path}")
        shutil.rmtree(target_path)
        
    if extracted_path.exists():
        shutil.move(str(extracted_path), str(target_path))
        print(f"Renamed {extracted_folder} to {target_folder}")
    else:
        print(f"Warning: {extracted_path} was not found after extraction.")

print("\nAll extractions completed!")
