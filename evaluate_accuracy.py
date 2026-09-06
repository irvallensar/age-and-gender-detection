import os
import re
from deepface import DeepFace

DATASET_DIR = "utkface_data"  

age_errors = []
gender_correct = 0
gender_total = 0

for filename in os.listdir(DATASET_DIR):
    match = re.match(r"(\d+)_(\d+)_", filename)
    if not match:
        continue
    true_age = int(match.group(1))
    true_gender = "Man" if match.group(2) == "0" else "Woman"  
    filepath = os.path.join(DATASET_DIR, filename)
    try:
        result = DeepFace.analyze(
            filepath,
            actions=['age', 'gender'],
            detector_backend='retinaface',
            enforce_detection=False
        )
        if isinstance(result, list):
            result = result[0]

        predicted_age = result['age']
        predicted_gender = result['dominant_gender']

        age_errors.append(abs(predicted_age - true_age))
        gender_total += 1
        if predicted_gender == true_gender:
            gender_correct += 1

    except Exception as e:
        print(f"Skipped {filename}: {e}")

mae = sum(age_errors) / len(age_errors)
gender_accuracy = (gender_correct / gender_total) * 100

print(f"Age MAE: {mae:.2f} years over {len(age_errors)} images")
print(f"Gender accuracy: {gender_accuracy:.2f}% over {gender_total} images")
