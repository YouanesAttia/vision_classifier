import os
import warnings
from PIL import Image, ImageFile
from pathlib import Path

warnings.filterwarnings('error', message='Truncated File Read')
ImageFile.LOAD_TRUNCATED_IMAGES = False 

data_dir = Path("data")

print("Scanning for corrupt and truncated images (Strict Mode)...")

count = 0
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.load()
            except BaseException as e:
                print(f"Removing bad file: {file_path}")
                try:
                    os.remove(file_path)
                    count += 1
                except:
                    pass

print(f"Cleanup complete. Removed {count} bad files.")