import os
import zipfile
import shutil
from pathlib import Path

def extract_zip(zip_path: Path, target_dir: Path, expected_prefix: str):
    print(f"[*] Extracting: {zip_path.name}...")
    
    # Create target directory if it does not exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create backup directory if current target directory is not empty
    if any(target_dir.iterdir()):
        backup_dir = target_dir.parent / f"{target_dir.name}_backup"
        print(f"[!] Target directory {target_dir.name} is not empty. Creating backup at {backup_dir.name}...")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(target_dir, backup_dir)
        # Delete old directory for a clean extract
        shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

    # Perform extraction and clean folder structure
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files in zip
        namelist = zip_ref.namelist()
        
        for member in namelist:
            # Strip the top-level directory prefix (e.g. 'vit5-colab-finetuned/')
            # to extract files directly under target_dir
            member_path = Path(member)
            if member_path.parts[0] == expected_prefix:
                # If it's just the root folder itself, skip
                if len(member_path.parts) == 1:
                    continue
                # Create relative path from the directory
                relative_path = Path(*member_path.parts[1:])
            else:
                relative_path = member_path
                
            target_path = target_dir / relative_path
            
            # If it is a directory, create it
            if member.endswith('/'):
                target_path.mkdir(parents=True, exist_ok=True)
            else:
                # Create parent directories if they don't exist
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # Write file
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                    
    print(f"[+] Successfully extracted {zip_path.name} to {target_dir.relative_to(Path.cwd())}!\n")

def main():
    workspace_dir = Path.cwd()
    data_dir = workspace_dir / "data" / "vietnamese-summarization-dataset-0001"
    models_dir = workspace_dir / "models"
    
    # Define models to look for
    models_to_check = ["vit5", "mt5", "bartpho"]
    
    found_any = False
    for model_id in models_to_check:
        # Check both naming conventions: model-finetuned.zip and model-colab-finetuned.zip
        candidates = [
            f"{model_id}-finetuned.zip",
            f"{model_id}-colab-finetuned.zip"
        ]
        
        zip_path = None
        for candidate in candidates:
            temp_path = data_dir / candidate
            if temp_path.exists():
                zip_path = temp_path
                break
                
        target_dir = models_dir / f"{model_id}-finetuned"
        
        if zip_path:
            found_any = True
            # The expected prefix folder inside the zip matches the zip file name without extension
            expected_prefix = zip_path.stem
            try:
                extract_zip(zip_path, target_dir, expected_prefix)
            except Exception as e:
                print(f"[-] Error extracting {zip_path.name}: {str(e)}")
        else:
            print(f"[i] No zip file found for {model_id} (tried {', '.join(candidates)}) in {data_dir.relative_to(workspace_dir)} - Skipping.")

    if not found_any:
        print("[-] No zip files found to extract in data/vietnamese-summarization-dataset-0001.")
    else:
        print("[*] All model updates completed! You can now restart your backend server to apply the new weights.")

if __name__ == "__main__":
    main()
