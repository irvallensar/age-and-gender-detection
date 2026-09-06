from datasets import load_dataset
import os

os.makedirs("utkface_data", exist_ok=True)
ds = load_dataset("py97/UTKFace-Cropped", split="train")

saved = 0
skipped = 0
for sample in ds:
    image = sample["jpg.chip.jpg"]
    if image is None:
        skipped += 1
        continue

    key = sample["__key__"]
    filename = os.path.basename(key)
    image.save(f"utkface_data/{filename}.jpg")
    saved += 1

print(f"Saved {saved} images to utkface_data/ ({skipped} skipped — missing image data)")
